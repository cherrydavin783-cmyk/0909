from __future__ import annotations

import pandas as pd

from .config import SystemConfig
from .timeutils import time_in_any_range, time_in_range


class RuleBasedStrategySelector:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def choose(self, row: pd.Series) -> str:
        if float(row.get("vpin_quantile", 0.0)) >= self.config.filters.toxic_vpin_quantile:
            return "toxic"
        timestamp = row.name
        breakout = self.config.strategies.breakout
        if breakout.enabled and time_in_range(timestamp, breakout.trade_start, breakout.trade_end):
            return "trend"
        mean_reversion = self.config.strategies.mean_reversion
        if mean_reversion.enabled and time_in_any_range(timestamp, mean_reversion.windows):
            return "range"
        micro_scalp = self.config.strategies.micro_scalp
        if micro_scalp.enabled and time_in_any_range(timestamp, micro_scalp.windows):
            return "scalp"
        return "idle"
