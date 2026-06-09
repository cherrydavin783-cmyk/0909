from __future__ import annotations

import pandas as pd

from gold_scalper.config import config_from_dict
from gold_scalper.indicators import compute_features
from gold_scalper.timeutils import session_label, time_in_range


def _sample_bars() -> pd.DataFrame:
    times = pd.date_range("2026-01-05 06:00", periods=40, freq="15min", tz="Asia/Shanghai")
    close = [2000 + (idx % 5) for idx in range(len(times))]
    return pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [10 + idx for idx in range(len(times))],
            "spread": [0.3] * len(times),
        }
    )


def test_time_ranges_and_sessions() -> None:
    assert time_in_range(pd.Timestamp("2026-01-05 22:00", tz="Asia/Shanghai"), "21:00", "01:00")
    assert session_label(pd.Timestamp("2026-01-05 07:00", tz="Asia/Shanghai")) == "asia"
    assert session_label(pd.Timestamp("2026-01-05 22:00", tz="Asia/Shanghai")) == "overlap"
    assert session_label(pd.Timestamp("2026-01-06 02:00", tz="Asia/Shanghai")) == "ny_tail"


def test_compute_features_adds_core_indicators() -> None:
    config = config_from_dict({})
    features = compute_features(_sample_bars(), config)
    assert {
        "atr",
        "atr_quantile",
        "rsi",
        "session",
        "ofi",
        "ofi_ratio",
        "vpin",
        "asia_width",
    }.issubset(features.columns)
    assert features["atr"].iloc[-1] > 0
    assert 0 <= features["rsi"].iloc[-1] <= 100
    assert features["asia_high"].max() > features["asia_low"].min()
    assert features["spread_median"].iloc[-1] == 0.3
