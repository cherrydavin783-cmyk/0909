from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from .config import SystemConfig
from .cross_assets import cross_asset_allows
from .models import OrderIntent, PositionState, RiskDecision, Signal
from .news import NewsCalendar


def _finite_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


class RiskManager:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def _risk_per_trade(self, signal: Signal) -> float:
        if signal.strategy == "micro_scalp":
            return self.config.strategies.micro_scalp.risk_per_trade
        return self.config.risk.risk_per_trade

    def _profit_target_equity_pct(self, signal: Signal) -> float:
        strategy_targets = self.config.risk.strategy_profit_target_equity_pct
        if signal.strategy in strategy_targets:
            return float(strategy_targets[signal.strategy])
        return self.config.risk.profit_target_equity_pct

    def _profit_target_price_pct(self, signal: Signal) -> float:
        strategy_targets = self.config.risk.strategy_profit_target_price_pct
        if signal.strategy in strategy_targets:
            return float(strategy_targets[signal.strategy])
        return self.config.risk.profit_target_price_pct

    def _round_volume(self, raw_volume: float) -> float:
        risk = self.config.risk
        if raw_volume <= 0:
            return 0.0
        steps = math.floor(raw_volume / risk.volume_step)
        volume = steps * risk.volume_step
        volume = max(risk.min_volume, min(risk.max_volume, volume))
        return round(volume, 8)

    def size_for_signal(self, signal: Signal, equity: float) -> tuple[float, float]:
        if signal.entry_price is None or signal.stop_loss is None:
            return 0.0, 0.0
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance <= 0.0:
            return 0.0, 0.0
        risk_amount = equity * self._risk_per_trade(signal)
        raw_volume = risk_amount / (stop_distance * self.config.data.contract_size)
        return self._round_volume(raw_volume), risk_amount

    def _profit_target_price(
        self,
        signal: Signal,
        row: pd.Series,
        equity: float,
        volume: float,
    ) -> float:
        assert signal.side is not None
        assert signal.entry_price is not None
        assert signal.take_profit is not None
        spread = float(row.get("spread", 0.0))
        slippage = self.config.risk.slippage_abs

        target_price_pct = self._profit_target_price_pct(signal)
        if target_price_pct > 0.0:
            if signal.side.name == "LONG":
                entry_fill = signal.entry_price + spread / 2 + slippage
                return entry_fill * (1 + target_price_pct)
            entry_fill = signal.entry_price - spread / 2 - slippage
            return entry_fill * (1 - target_price_pct)

        target_pct = self._profit_target_equity_pct(signal)
        if target_pct <= 0.0 or volume <= 0.0:
            return signal.take_profit
        commission = self.config.risk.commission_per_lot * volume * 2
        target_net = equity * target_pct
        required_move = (target_net + commission) / (
            volume * self.config.data.contract_size
        )
        if signal.side.name == "LONG":
            entry_fill = signal.entry_price + spread / 2 + slippage
            return entry_fill + required_move + slippage
        entry_fill = signal.entry_price - spread / 2 - slippage
        return entry_fill - required_move - slippage

    def approve(
        self,
        signal: Signal,
        row: pd.Series,
        equity: float,
        day_start_equity: float,
        current_position: PositionState | None,
        news_calendar: NewsCalendar | None,
        paper_mode: bool = False,
    ) -> RiskDecision:
        if not signal.is_entry or signal.side is None:
            return RiskDecision(False, signal.reason or "no entry signal")
        timestamp: datetime = signal.timestamp
        filters = self.config.filters

        if current_position is not None and current_position.side == signal.side:
            return RiskDecision(False, "same-side position already open")

        if day_start_equity > 0:
            drawdown = max(0.0, (day_start_equity - equity) / day_start_equity)
            if drawdown >= self.config.risk.max_daily_drawdown:
                return RiskDecision(False, "daily drawdown circuit breaker")

        spread = float(row.get("spread", 0.0))
        spread_median = float(row.get("spread_median", spread))
        if spread > filters.max_spread_abs:
            return RiskDecision(False, "spread above absolute limit")
        if spread_median > 0 and spread > spread_median * filters.max_spread_multiple:
            return RiskDecision(False, "spread above rolling limit")

        if float(row.get("vpin_quantile", 0.0)) >= filters.toxic_vpin_quantile:
            return RiskDecision(False, "toxic vpin filter")

        atr = _finite_float(row, "atr")
        if filters.max_atr_abs > 0.0 and atr >= filters.max_atr_abs:
            return RiskDecision(False, "atr above absolute limit")
        atr_quantile = _finite_float(row, "atr_quantile")
        if (
            0.0 < filters.max_atr_quantile < 1.0
            and atr_quantile >= filters.max_atr_quantile
        ):
            return RiskDecision(False, "atr above quantile limit")

        if news_calendar is None or not news_calendar.available:
            if paper_mode:
                return RiskDecision(False, "news calendar required for paper mode")
        elif news_calendar.in_blackout(
            timestamp, filters.news_before_minutes, filters.news_after_minutes
        ):
            return RiskDecision(False, "news blackout")

        allowed_by_cross_assets, cross_asset_reason = cross_asset_allows(
            signal.side, row, self.config
        )
        if not allowed_by_cross_assets:
            return RiskDecision(False, cross_asset_reason)

        volume, risk_amount = self.size_for_signal(signal, equity)
        if volume < self.config.risk.min_volume:
            return RiskDecision(False, "position size below minimum")
        assert signal.entry_price is not None
        assert signal.stop_loss is not None
        assert signal.take_profit is not None
        take_profit = self._profit_target_price(signal, row, equity, volume)
        order = OrderIntent(
            timestamp=timestamp,
            symbol=self.config.data.symbol,
            side=signal.side,
            volume=volume,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=take_profit,
            strategy=signal.strategy,
            risk_amount=risk_amount,
            reason=f"{signal.reason}; {cross_asset_reason}",
        )
        return RiskDecision(True, "approved", order)
