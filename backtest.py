from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .config import SystemConfig
from .indicators import _time_window_mask, compute_features
from .models import BacktestResult, PositionState, Side, TradeRecord
from .news import NewsCalendar
from .risk import RiskManager
from .selector import RuleBasedStrategySelector
from .strategies import BacktestContext, BreakoutStrategy, MeanReversionStrategy, MicroScalpStrategy


def load_market_csv(
    path: str | Path,
    config: SystemConfig,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Market data CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    data = compute_features(frame, config)
    if start:
        start_ts = pd.Timestamp(start, tz=config.data.timezone)
        data = data.loc[data.index >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz=config.data.timezone) + pd.Timedelta(days=1)
        data = data.loc[data.index < end_ts]
    return data


def coverage_summary(data: pd.DataFrame) -> dict[str, object]:
    if data.empty:
        return {"rows": 0, "start": None, "end": None}
    return {"rows": len(data), "start": data.index[0], "end": data.index[-1]}


def requested_coverage_ok(
    data: pd.DataFrame,
    start: str | None,
    end: str | None,
    timezone: str,
) -> tuple[bool, str]:
    if data.empty:
        return False, "market data is empty"
    messages: list[str] = []
    if start:
        start_ts = pd.Timestamp(start, tz=timezone)
        if data.index[0] > start_ts + pd.Timedelta(days=1):
            messages.append(f"data starts at {data.index[0]}, requested {start_ts}")
    if end:
        end_ts = pd.Timestamp(end, tz=timezone)
        if data.index[-1] < end_ts - pd.Timedelta(days=1):
            messages.append(f"data ends at {data.index[-1]}, requested {end_ts}")
    return not messages, "; ".join(messages)


def _mark_price(row: pd.Series, position: PositionState | None) -> float:
    if position is None:
        return float(row["close"])
    spread = float(row.get("spread", 0.0))
    if position.side is Side.LONG:
        return float(row["close"]) - spread / 2
    return float(row["close"]) + spread / 2


def _entry_fill_price(row: pd.Series, side: Side, slippage_abs: float) -> float:
    spread = float(row.get("spread", 0.0))
    close = float(row["close"])
    if side is Side.LONG:
        return close + spread / 2 + slippage_abs
    return close - spread / 2 - slippage_abs


def _exit_fill_price(base_price: float, side: Side, slippage_abs: float) -> float:
    if side is Side.LONG:
        return base_price - slippage_abs
    return base_price + slippage_abs


def _breakeven_stop_price(
    position: PositionState,
    config: SystemConfig,
) -> float:
    commission = config.risk.commission_per_lot * position.volume * 2
    commission_buffer = commission / (position.volume * config.data.contract_size)
    buffer_abs = (
        config.risk.slippage_abs
        + commission_buffer
        + config.risk.breakeven_buffer_points * config.data.point
    )
    return position.entry_price + position.side.sign * buffer_abs


def _maybe_move_stop_to_breakeven(
    position: PositionState,
    row: pd.Series,
    config: SystemConfig,
) -> None:
    trigger_pct = config.risk.breakeven_trigger_price_pct
    if trigger_pct <= 0.0 or position.breakeven_activated:
        return

    high = float(row["high"])
    low = float(row["low"])
    if position.side is Side.LONG:
        trigger_price = position.entry_price * (1 + trigger_pct)
        if high < trigger_price:
            return
        new_stop = min(_breakeven_stop_price(position, config), position.take_profit)
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop
            position.breakeven_activated = True
        return

    trigger_price = position.entry_price * (1 - trigger_pct)
    if low > trigger_price:
        return
    new_stop = max(_breakeven_stop_price(position, config), position.take_profit)
    if new_stop < position.stop_loss:
        position.stop_loss = new_stop
        position.breakeven_activated = True


def _exit_position(
    position: PositionState,
    row: pd.Series,
    contract_size: float,
    commission_per_lot: float,
    slippage_abs: float,
) -> TradeRecord | None:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    spread = float(row.get("spread", 0.0))
    timestamp = row.name.to_pydatetime() if hasattr(row.name, "to_pydatetime") else row.name
    exit_price: float | None = None
    exit_reason: str | None = None

    if position.side is Side.LONG:
        if low <= position.stop_loss:
            exit_price = position.stop_loss
            exit_reason = "breakeven_stop" if position.breakeven_activated else "stop_loss"
        elif high >= position.take_profit:
            exit_price = position.take_profit
            exit_reason = "take_profit"
        elif timestamp - position.opened_at >= timedelta(minutes=position.max_holding_minutes):
            exit_price = close - spread / 2
            exit_reason = "max_holding_time"
    else:
        if high >= position.stop_loss:
            exit_price = position.stop_loss
            exit_reason = "breakeven_stop" if position.breakeven_activated else "stop_loss"
        elif low <= position.take_profit:
            exit_price = position.take_profit
            exit_reason = "take_profit"
        elif timestamp - position.opened_at >= timedelta(minutes=position.max_holding_minutes):
            exit_price = close + spread / 2
            exit_reason = "max_holding_time"

    if exit_price is None or exit_reason is None:
        return None
    exit_price = _exit_fill_price(exit_price, position.side, slippage_abs)

    gross = (
        (exit_price - position.entry_price)
        * position.side.sign
        * position.volume
        * contract_size
    )
    commission = commission_per_lot * position.volume * 2
    return TradeRecord(
        trade_id=position.position_id,
        symbol=position.symbol,
        side=position.side,
        strategy=position.strategy,
        volume=position.volume,
        entry_time=position.opened_at,
        entry_price=position.entry_price,
        exit_time=timestamp,
        exit_price=exit_price,
        pnl=gross - commission,
        exit_reason=exit_reason,
    )


def _gap_exit_position(
    position: PositionState,
    row: pd.Series,
    contract_size: float,
    commission_per_lot: float,
    slippage_abs: float,
) -> TradeRecord:
    spread = float(row.get("spread", 0.0))
    open_price = float(row["open"])
    timestamp = row.name.to_pydatetime() if hasattr(row.name, "to_pydatetime") else row.name
    exit_price = open_price - spread / 2 if position.side is Side.LONG else open_price + spread / 2
    exit_price = _exit_fill_price(exit_price, position.side, slippage_abs)
    gross = (
        (exit_price - position.entry_price)
        * position.side.sign
        * position.volume
        * contract_size
    )
    commission = commission_per_lot * position.volume * 2
    return TradeRecord(
        trade_id=position.position_id,
        symbol=position.symbol,
        side=position.side,
        strategy=position.strategy,
        volume=position.volume,
        entry_time=position.opened_at,
        entry_price=position.entry_price,
        exit_time=timestamp,
        exit_price=exit_price,
        pnl=gross - commission,
        exit_reason="quote_gap",
    )


def _calculate_metrics(
    trades: list[TradeRecord], equity_curve: list[dict[str, object]], initial_equity: float
) -> dict[str, float]:
    ending_equity = float(equity_curve[-1]["equity"]) if equity_curve else initial_equity
    pnl_values = [trade.pnl for trade in trades]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    equity_values = pd.Series([float(point["equity"]) for point in equity_curve])
    if not equity_values.empty:
        running_max = equity_values.cummax()
        drawdowns = (running_max - equity_values) / running_max.replace(0.0, pd.NA)
        max_drawdown = float(drawdowns.max(skipna=True) or 0.0)
    else:
        max_drawdown = 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf") if wins else 0.0
    return {
        "initial_equity": float(initial_equity),
        "ending_equity": ending_equity,
        "total_return": (ending_equity - initial_equity) / initial_equity,
        "trade_count": float(len(trades)),
        "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
        "max_drawdown": max_drawdown,
        "profit_factor": float(profit_factor),
        "net_pnl": ending_equity - initial_equity,
    }


def _candidate_indices(data: pd.DataFrame, config: SystemConfig) -> list[int]:
    toxic = data["vpin_quantile"].to_numpy(dtype=float) >= config.filters.toxic_vpin_quantile

    breakout_cfg = config.strategies.breakout
    if breakout_cfg.enabled:
        breakout_window = _time_window_mask(data.index, breakout_cfg.trade_start, breakout_cfg.trade_end)
        close = data["close"].to_numpy(dtype=float)
        asia_high = data["asia_high"].to_numpy(dtype=float)
        asia_low = data["asia_low"].to_numpy(dtype=float)
        asia_width = data["asia_width"].to_numpy(dtype=float)
        rsi = data["rsi"].to_numpy(dtype=float)
        ofi = data["ofi"].to_numpy(dtype=float)
        volume_spike = data["volume_spike"].to_numpy(dtype=bool)
        breakout_signal = (
            breakout_window
            & (asia_width >= breakout_cfg.min_asia_width)
            & (rsi >= breakout_cfg.min_rsi)
            & (rsi <= breakout_cfg.max_rsi)
            & volume_spike
            & (((close > asia_high) & (ofi > 0.0)) | ((close < asia_low) & (ofi < 0.0)))
        )
    else:
        breakout_signal = pd.Series(False, index=data.index).to_numpy(dtype=bool).copy()

    mean_cfg = config.strategies.mean_reversion
    if mean_cfg.enabled:
        mean_window = pd.Series(False, index=data.index).to_numpy(dtype=bool).copy()
        for start, end in mean_cfg.windows:
            mean_window |= _time_window_mask(data.index, start, end)
        atr = data["atr"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        fair_value = data["fair_value"].to_numpy(dtype=float)
        ofi_ratio = data["ofi_ratio"].to_numpy(dtype=float)
        vpin_quantile = data["vpin_quantile"].to_numpy(dtype=float)
        min_deviation = mean_cfg.fair_value_atr_deviation * atr
        mean_signal = (
            mean_window
            & (atr > 0.0)
            & (vpin_quantile <= mean_cfg.max_vpin_quantile)
            & (
                ((ofi_ratio >= mean_cfg.ofi_ratio_threshold) & (close > fair_value + min_deviation))
                | ((ofi_ratio <= -mean_cfg.ofi_ratio_threshold) & (close < fair_value - min_deviation))
            )
        )
    else:
        mean_signal = pd.Series(False, index=data.index).to_numpy(dtype=bool).copy()

    scalp_cfg = config.strategies.micro_scalp
    if scalp_cfg.enabled:
        scalp_window = pd.Series(False, index=data.index).to_numpy(dtype=bool).copy()
        for start, end in scalp_cfg.windows:
            scalp_window |= _time_window_mask(data.index, start, end)
        close = data["close"].to_numpy(dtype=float)
        micro_high = data["micro_high"].to_numpy(dtype=float)
        micro_low = data["micro_low"].to_numpy(dtype=float)
        rsi = data["rsi"].to_numpy(dtype=float)
        ofi = data["ofi"].to_numpy(dtype=float)
        atr = data["atr"].to_numpy(dtype=float)
        vpin_quantile = data["vpin_quantile"].to_numpy(dtype=float)
        asia_width = data["asia_width"].to_numpy(dtype=float)
        volume_ok = (
            data["volume_spike"].to_numpy(dtype=bool)
            if scalp_cfg.require_volume_spike
            else pd.Series(True, index=data.index).to_numpy(dtype=bool)
        )
        scalp_signal = (
            scalp_window
            & (atr > 0.0)
            & (asia_width >= scalp_cfg.min_asia_width)
            & (vpin_quantile <= scalp_cfg.max_vpin_quantile)
            & volume_ok
            & (
                (
                    (close > micro_high)
                    & (rsi >= scalp_cfg.min_rsi_long)
                    & (ofi >= scalp_cfg.min_ofi_abs)
                )
                | (
                    (close < micro_low)
                    & (rsi <= scalp_cfg.max_rsi_short)
                    & (ofi <= -scalp_cfg.min_ofi_abs)
                )
            )
        )
    else:
        scalp_signal = pd.Series(False, index=data.index).to_numpy(dtype=bool).copy()

    return ((breakout_signal | mean_signal | scalp_signal) & ~toxic).nonzero()[0].tolist()


class BacktestEngine:
    def __init__(self, config: SystemConfig, news_calendar: NewsCalendar | None = None) -> None:
        self.config = config
        self.news_calendar = news_calendar
        self.selector = RuleBasedStrategySelector(config)
        self.breakout = BreakoutStrategy(config)
        self.mean_reversion = MeanReversionStrategy(config)
        self.micro_scalp = MicroScalpStrategy(config)
        self.risk = RiskManager(config)

    def run(
        self,
        market_data: pd.DataFrame,
        run_id: str | None = None,
        features_ready: bool = False,
    ) -> BacktestResult:
        data = market_data.copy() if features_ready else compute_features(market_data, self.config)
        data = data.sort_index()
        run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        equity = self.config.risk.initial_equity
        day_start_equity = equity
        current_day: date | None = None
        position: PositionState | None = None
        trades: list[TradeRecord] = []
        equity_curve: list[dict[str, object]] = []
        context = BacktestContext()
        candidates = _candidate_indices(data, self.config)
        candidate_pointer = 0
        row_count = len(data)
        previous_timestamp = None

        i = 0
        while i < row_count:
            if position is None:
                while candidate_pointer < len(candidates) and candidates[candidate_pointer] < i:
                    candidate_pointer += 1
                if candidate_pointer >= len(candidates):
                    break
                i = candidates[candidate_pointer]
                candidate_pointer += 1
            row = data.iloc[i]
            timestamp = row.name.to_pydatetime() if hasattr(row.name, "to_pydatetime") else row.name
            if current_day != timestamp.date():
                current_day = timestamp.date()
                day_start_equity = equity

            if position is not None:
                gap_minutes = (
                    (timestamp - previous_timestamp).total_seconds() / 60
                    if previous_timestamp is not None
                    else 0
                )
                if gap_minutes > self.config.risk.max_quote_gap_minutes:
                    trade = _gap_exit_position(
                        position,
                        row,
                        self.config.data.contract_size,
                        self.config.risk.commission_per_lot,
                        self.config.risk.slippage_abs,
                    )
                else:
                    trade = _exit_position(
                        position,
                        row,
                        self.config.data.contract_size,
                        self.config.risk.commission_per_lot,
                        self.config.risk.slippage_abs,
                    )
                if trade is not None:
                    trades.append(trade)
                    equity += trade.pnl
                    position = None
                else:
                    _maybe_move_stop_to_breakeven(position, row, self.config)

            opened_position = False
            if position is None:
                mode = self.selector.choose(row)
                signal = None
                if mode == "trend":
                    signal = self.breakout.generate(row, context)
                    if signal is not None and not signal.is_entry:
                        scalp_signal = self.micro_scalp.generate(row, context)
                        if scalp_signal.is_entry:
                            signal = scalp_signal
                elif mode == "range":
                    signal = self.mean_reversion.generate(row, context)
                    if signal is not None and not signal.is_entry:
                        scalp_signal = self.micro_scalp.generate(row, context)
                        if scalp_signal.is_entry:
                            signal = scalp_signal
                elif mode == "scalp":
                    signal = self.micro_scalp.generate(row, context)
                if signal is not None:
                    decision = self.risk.approve(
                        signal,
                        row,
                        equity,
                        day_start_equity,
                        position,
                        self.news_calendar,
                        paper_mode=False,
                    )
                    if decision.allowed and decision.order is not None:
                        order = decision.order
                        fill = _entry_fill_price(row, order.side, self.config.risk.slippage_abs)
                        position = PositionState(
                            position_id=uuid4().hex[:12],
                            symbol=order.symbol,
                            side=order.side,
                            volume=order.volume,
                            entry_price=fill,
                            stop_loss=order.stop_loss,
                            take_profit=order.take_profit,
                            opened_at=order.timestamp,
                            strategy=order.strategy,
                            max_holding_minutes=signal.max_holding_minutes,
                        )
                        context.increment(order.strategy, timestamp.date())
                        opened_position = True

            total_equity = equity
            if position is not None:
                total_equity += position.unrealized_pnl(
                    _mark_price(row, position), self.config.data.contract_size
                )
            equity_curve.append({"time": timestamp, "equity": float(total_equity)})
            i += 1

            previous_timestamp = timestamp
            if opened_position and i < row_count:
                continue

        if position is not None and not data.empty:
            last = data.iloc[-1]
            timestamp = last.name.to_pydatetime() if hasattr(last.name, "to_pydatetime") else last.name
            exit_price = _mark_price(last, position)
            exit_price = _exit_fill_price(
                exit_price,
                position.side,
                self.config.risk.slippage_abs,
            )
            gross = (
                (exit_price - position.entry_price)
                * position.side.sign
                * position.volume
                * self.config.data.contract_size
            )
            commission = self.config.risk.commission_per_lot * position.volume * 2
            trade = TradeRecord(
                trade_id=position.position_id,
                symbol=position.symbol,
                side=position.side,
                strategy=position.strategy,
                volume=position.volume,
                entry_time=position.opened_at,
                entry_price=position.entry_price,
                exit_time=timestamp,
                exit_price=exit_price,
                pnl=gross - commission,
                exit_reason="end_of_backtest",
            )
            trades.append(trade)
            equity += trade.pnl
            if equity_curve:
                equity_curve[-1]["equity"] = float(equity)

        if not equity_curve and not data.empty:
            timestamp = data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else data.index[-1]
            equity_curve.append({"time": timestamp, "equity": float(equity)})
        elif equity_curve and not data.empty and equity_curve[-1]["time"] != data.index[-1]:
            timestamp = data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else data.index[-1]
            equity_curve.append({"time": timestamp, "equity": float(equity)})

        metrics = _calculate_metrics(trades, equity_curve, self.config.risk.initial_equity)
        return BacktestResult(
            run_id=run_id,
            symbol=self.config.data.symbol,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )
