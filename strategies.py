from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .config import SystemConfig
from .models import Side, Signal, SignalType
from .timeutils import time_in_any_range, time_in_range


@dataclass
class BacktestContext:
    daily_strategy_counts: dict[tuple[str, date], int] = field(default_factory=dict)

    def count_for_day(self, strategy: str, day: date) -> int:
        return self.daily_strategy_counts.get((strategy, day), 0)

    def increment(self, strategy: str, day: date) -> None:
        key = (strategy, day)
        self.daily_strategy_counts[key] = self.daily_strategy_counts.get(key, 0) + 1


def _timestamp(row: pd.Series):
    return row.name.to_pydatetime() if hasattr(row.name, "to_pydatetime") else row.name


def _value(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    if pd.isna(value):
        return default
    return value


class BreakoutStrategy:
    name = "breakout"

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def generate(self, row: pd.Series, context: BacktestContext | None = None) -> Signal:
        cfg = self.config.strategies.breakout
        timestamp = _timestamp(row)
        if not cfg.enabled:
            return Signal.none(timestamp, self.name, "disabled")
        if not time_in_range(row.name, cfg.trade_start, cfg.trade_end):
            return Signal.none(timestamp, self.name, "outside breakout window")
        if context and context.count_for_day(self.name, row.name.date()) >= cfg.max_daily_trades:
            return Signal.none(timestamp, self.name, "daily breakout trade limit")

        asia_high = _value(row, "asia_high")
        asia_low = _value(row, "asia_low")
        asia_width = _value(row, "asia_width")
        atr = _value(row, "atr")
        rsi = _value(row, "rsi", 50.0)
        close = _value(row, "close")
        ofi = _value(row, "ofi")

        if asia_width < cfg.min_asia_width:
            return Signal.none(timestamp, self.name, "asia range too narrow")
        if atr <= 0.0:
            return Signal.none(timestamp, self.name, "missing atr")
        if not (cfg.min_rsi <= rsi <= cfg.max_rsi):
            return Signal.none(timestamp, self.name, "rsi outside neutral band")
        if not bool(row.get("volume_spike", False)):
            return Signal.none(timestamp, self.name, "no volume spike")

        if close > asia_high and ofi > 0.0:
            stop = close - cfg.stop_atr_multiple * atr
            take = close + cfg.take_profit_atr_multiple * atr
            return Signal(
                timestamp=timestamp,
                strategy=self.name,
                signal_type=SignalType.ENTRY,
                side=Side.LONG,
                entry_price=close,
                stop_loss=stop,
                take_profit=take,
                max_holding_minutes=cfg.max_holding_minutes,
                reason="bullish asia-range breakout",
            )
        if close < asia_low and ofi < 0.0:
            stop = close + cfg.stop_atr_multiple * atr
            take = close - cfg.take_profit_atr_multiple * atr
            return Signal(
                timestamp=timestamp,
                strategy=self.name,
                signal_type=SignalType.ENTRY,
                side=Side.SHORT,
                entry_price=close,
                stop_loss=stop,
                take_profit=take,
                max_holding_minutes=cfg.max_holding_minutes,
                reason="bearish asia-range breakout",
            )
        return Signal.none(timestamp, self.name, "no confirmed breakout")


class MeanReversionStrategy:
    name = "mean_reversion"

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def generate(self, row: pd.Series, context: BacktestContext | None = None) -> Signal:
        del context
        cfg = self.config.strategies.mean_reversion
        timestamp = _timestamp(row)
        if not cfg.enabled:
            return Signal.none(timestamp, self.name, "disabled")
        if not time_in_any_range(row.name, cfg.windows):
            return Signal.none(timestamp, self.name, "outside mean-reversion window")

        atr = _value(row, "atr")
        close = _value(row, "close")
        fair_value = _value(row, "fair_value", close)
        ofi_ratio = _value(row, "ofi_ratio")
        vpin_quantile = _value(row, "vpin_quantile")
        if atr <= 0.0:
            return Signal.none(timestamp, self.name, "missing atr")
        if vpin_quantile > cfg.max_vpin_quantile:
            return Signal.none(timestamp, self.name, "vpin too high for fade")

        min_deviation = cfg.fair_value_atr_deviation * atr
        stop_distance = min(cfg.stop_atr_multiple * atr, cfg.max_stop_distance)
        if stop_distance <= 0.0:
            return Signal.none(timestamp, self.name, "invalid stop distance")

        if ofi_ratio >= cfg.ofi_ratio_threshold and close > fair_value + min_deviation:
            return Signal(
                timestamp=timestamp,
                strategy=self.name,
                signal_type=SignalType.ENTRY,
                side=Side.SHORT,
                entry_price=close,
                stop_loss=close + stop_distance,
                take_profit=fair_value,
                max_holding_minutes=cfg.max_holding_minutes,
                reason="buy-side imbalance fade",
            )
        if ofi_ratio <= -cfg.ofi_ratio_threshold and close < fair_value - min_deviation:
            return Signal(
                timestamp=timestamp,
                strategy=self.name,
                signal_type=SignalType.ENTRY,
                side=Side.LONG,
                entry_price=close,
                stop_loss=close - stop_distance,
                take_profit=fair_value,
                max_holding_minutes=cfg.max_holding_minutes,
                reason="sell-side imbalance fade",
            )
        return Signal.none(timestamp, self.name, "no fade setup")


class MicroScalpStrategy:
    name = "micro_scalp"

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def generate(self, row: pd.Series, context: BacktestContext | None = None) -> Signal:
        cfg = self.config.strategies.micro_scalp
        timestamp = _timestamp(row)
        if not cfg.enabled:
            return Signal.none(timestamp, self.name, "disabled")
        if not time_in_any_range(row.name, cfg.windows):
            return Signal.none(timestamp, self.name, "outside micro-scalp window")
        if context and context.count_for_day(self.name, row.name.date()) >= cfg.max_daily_trades:
            return Signal.none(timestamp, self.name, "daily micro-scalp trade limit")

        atr = _value(row, "atr")
        close = _value(row, "close")
        rsi = _value(row, "rsi", 50.0)
        ofi = _value(row, "ofi")
        micro_high = _value(row, "micro_high")
        micro_low = _value(row, "micro_low")
        asia_width = _value(row, "asia_width")
        vpin_quantile = _value(row, "vpin_quantile")
        if atr <= 0.0 or micro_high <= 0.0 or micro_low <= 0.0:
            return Signal.none(timestamp, self.name, "missing micro range")
        if asia_width < cfg.min_asia_width:
            return Signal.none(timestamp, self.name, "asia range too narrow for scalp")
        if vpin_quantile > cfg.max_vpin_quantile:
            return Signal.none(timestamp, self.name, "vpin too high for scalp")
        if cfg.require_volume_spike and not bool(row.get("volume_spike", False)):
            return Signal.none(timestamp, self.name, "no volume spike")

        stop_distance = cfg.stop_atr_multiple * atr
        take_distance = cfg.take_profit_atr_multiple * atr
        if stop_distance <= 0.0 or take_distance <= 0.0:
            return Signal.none(timestamp, self.name, "invalid scalp distances")

        if close > micro_high and rsi >= cfg.min_rsi_long and ofi >= cfg.min_ofi_abs:
            return Signal(
                timestamp=timestamp,
                strategy=self.name,
                signal_type=SignalType.ENTRY,
                side=Side.LONG,
                entry_price=close,
                stop_loss=close - stop_distance,
                take_profit=close + take_distance,
                max_holding_minutes=cfg.max_holding_minutes,
                reason="micro range bullish scalp",
            )
        if close < micro_low and rsi <= cfg.max_rsi_short and ofi <= -cfg.min_ofi_abs:
            return Signal(
                timestamp=timestamp,
                strategy=self.name,
                signal_type=SignalType.ENTRY,
                side=Side.SHORT,
                entry_price=close,
                stop_loss=close + stop_distance,
                take_profit=close - take_distance,
                max_holding_minutes=cfg.max_holding_minutes,
                reason="micro range bearish scalp",
            )
        return Signal.none(timestamp, self.name, "no micro scalp setup")
