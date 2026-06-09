from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from .backtest import _exit_position, _maybe_move_stop_to_breakeven
from .config import SystemConfig
from .models import OrderIntent, PositionState, Side, TradeRecord


def _timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value)


def _paper_position_id(
    opened_at,
    symbol: str,
    strategy: str,
    side: Side,
    entry_price: float,
) -> str:
    key = (
        f"{_timestamp(opened_at).isoformat()}|{symbol}|{strategy}|"
        f"{side.value}|{float(entry_price):.5f}"
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _strategy_max_holding_minutes(config: SystemConfig, strategy: str) -> int:
    strategy_config = {
        "breakout": config.strategies.breakout,
        "mean_reversion": config.strategies.mean_reversion,
        "micro_scalp": config.strategies.micro_scalp,
    }.get(strategy)
    return int(getattr(strategy_config, "max_holding_minutes", 5))


def position_from_order(order: OrderIntent, max_holding_minutes: int) -> PositionState:
    return PositionState(
        position_id=_paper_position_id(
            order.timestamp,
            order.symbol,
            order.strategy,
            order.side,
            order.entry_price,
        ),
        symbol=order.symbol,
        side=order.side,
        volume=order.volume,
        entry_price=order.entry_price,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
        opened_at=order.timestamp,
        strategy=order.strategy,
        max_holding_minutes=max_holding_minutes,
    )


def evaluate_position_exit(
    position: PositionState,
    data: pd.DataFrame,
    config: SystemConfig,
) -> TradeRecord | None:
    if data.empty:
        return None

    opened_at = _timestamp(position.opened_at)
    rows = data.loc[data.index > opened_at]
    for _, row in rows.iterrows():
        trade = _exit_position(
            position,
            row,
            config.data.contract_size,
            config.risk.commission_per_lot,
            config.risk.slippage_abs,
        )
        if trade is not None:
            return trade
        _maybe_move_stop_to_breakeven(position, row, config)
    return None


class PaperPositionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> PositionState | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return PositionState(
            position_id=str(payload["position_id"]),
            symbol=str(payload["symbol"]),
            side=Side(str(payload["side"])),
            volume=float(payload["volume"]),
            entry_price=float(payload["entry_price"]),
            stop_loss=float(payload["stop_loss"]),
            take_profit=float(payload["take_profit"]),
            opened_at=_timestamp(payload["opened_at"]).to_pydatetime(),
            strategy=str(payload["strategy"]),
            max_holding_minutes=int(payload.get("max_holding_minutes", 5)),
            breakeven_activated=bool(payload.get("breakeven_activated", False)),
        )

    def save(self, position: PositionState) -> None:
        payload = {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "side": position.side.value,
            "volume": position.volume,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "opened_at": _timestamp(position.opened_at).isoformat(),
            "strategy": position.strategy,
            "max_holding_minutes": position.max_holding_minutes,
            "breakeven_activated": position.breakeven_activated,
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


class PaperTradeLog:
    header = [
        "trade_id",
        "symbol",
        "strategy",
        "side",
        "volume",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "pnl",
        "exit_reason",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(self.header)

    def append(self, trade: TradeRecord) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    trade.trade_id,
                    trade.symbol,
                    trade.strategy,
                    trade.side.value,
                    trade.volume,
                    _timestamp(trade.entry_time).isoformat(),
                    trade.entry_price,
                    _timestamp(trade.exit_time).isoformat(),
                    trade.exit_price,
                    trade.pnl,
                    trade.exit_reason,
                ]
            )

    def closed_trade_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            return {
                row["trade_id"]
                for row in csv.DictReader(handle)
                if row.get("trade_id")
            }


def recover_latest_open_position(
    signal_path: str | Path,
    trade_log: PaperTradeLog,
    config: SystemConfig,
) -> PositionState | None:
    path = Path(signal_path)
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    closed_ids = trade_log.closed_trade_ids()
    for row in reversed(rows):
        opened_at = _timestamp(row["time"]).to_pydatetime()
        side = Side(row["side"])
        entry_price = float(row["entry"])
        position_id = _paper_position_id(
            opened_at,
            row["symbol"],
            row["strategy"],
            side,
            entry_price,
        )
        if position_id in closed_ids:
            continue
        return PositionState(
            position_id=position_id,
            symbol=row["symbol"],
            side=side,
            volume=float(row["volume"]),
            entry_price=entry_price,
            stop_loss=float(row["stop_loss"]),
            take_profit=float(row["take_profit"]),
            opened_at=opened_at,
            strategy=row["strategy"],
            max_holding_minutes=_strategy_max_holding_minutes(config, row["strategy"]),
        )
    return None
