from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1


class SignalType(str, Enum):
    ENTRY = "entry"
    NONE = "none"


@dataclass
class MarketSnapshot:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float
    session: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    timestamp: datetime
    strategy: str
    signal_type: SignalType
    side: Side | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    max_holding_minutes: int = 5
    reason: str = ""
    confidence: float = 1.0

    @classmethod
    def none(cls, timestamp: datetime, strategy: str, reason: str = "") -> "Signal":
        return cls(timestamp=timestamp, strategy=strategy, signal_type=SignalType.NONE, reason=reason)

    @property
    def is_entry(self) -> bool:
        return self.signal_type is SignalType.ENTRY and self.side is not None


@dataclass
class OrderIntent:
    timestamp: datetime
    symbol: str
    side: Side
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    risk_amount: float
    reason: str = ""


@dataclass
class PositionState:
    position_id: str
    symbol: str
    side: Side
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    strategy: str
    max_holding_minutes: int = 5
    breakeven_activated: bool = False

    def unrealized_pnl(self, mark_price: float, contract_size: float) -> float:
        return (mark_price - self.entry_price) * self.side.sign * self.volume * contract_size


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    side: Side
    strategy: str
    volume: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    pnl: float
    exit_reason: str


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    order: OrderIntent | None = None


@dataclass
class BacktestResult:
    run_id: str
    symbol: str
    trades: list[TradeRecord]
    equity_curve: list[dict[str, Any]]
    metrics: dict[str, float]
    report_dir: str | None = None
