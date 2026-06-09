from __future__ import annotations

import pandas as pd

from gold_scalper.config import config_from_dict
from gold_scalper.cross_assets import (
    add_cross_asset_features,
    cross_asset_allows,
    sync_fred_cross_assets,
)
from gold_scalper.models import Side


def _market_frame() -> pd.DataFrame:
    index = pd.date_range("2026-05-19 20:00", periods=4, freq="1min", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "open": [0.0] * 4,
            "high": [0.0] * 4,
            "low": [0.0] * 4,
            "close": [0.0] * 4,
            "volume": [0.0] * 4,
            "spread": [0.0] * 4,
        },
        index=index,
    )


def test_cross_asset_bias_long_when_dxy_and_yield_fall() -> None:
    config = config_from_dict(
        {
            "cross_assets": {
                "block_on_conflict": True,
                "momentum_lookback": 1,
                "dxy_min_change": 0.05,
                "yield_min_change": 0.01,
            }
        }
    )
    cross = pd.DataFrame(
        {
            "dxy": [100.2, 100.1, 100.0, 99.9],
            "us10y": [4.50, 4.48, 4.46, 4.44],
        },
        index=pd.date_range("2026-05-19 20:00", periods=4, freq="1min", tz="Asia/Shanghai"),
    )
    merged = add_cross_asset_features(_market_frame(), config, cross)
    assert merged["cross_asset_bias"].iloc[-1] == "long"
    allowed, reason = cross_asset_allows(Side.LONG, merged.iloc[-1], config)
    assert allowed
    assert "long" in reason
    allowed, reason = cross_asset_allows(Side.SHORT, merged.iloc[-1], config)
    assert not allowed
    assert "conflict" in reason


def test_cross_asset_missing_is_neutral_by_default() -> None:
    config = config_from_dict({})
    merged = add_cross_asset_features(_market_frame(), config, pd.DataFrame())
    assert merged["cross_asset_bias"].iloc[-1] == "missing"
    allowed, reason = cross_asset_allows(Side.LONG, merged.iloc[-1], config)
    assert allowed
    assert "missing" in reason


def test_cross_asset_can_require_confirmation() -> None:
    config = config_from_dict({"cross_assets": {"require_confirmation": True}})
    row = pd.Series({"cross_asset_bias": "neutral"})
    allowed, reason = cross_asset_allows(Side.LONG, row, config)
    assert not allowed
    assert "confirmation" in reason


def test_sync_fred_cross_assets(monkeypatch, tmp_path) -> None:
    def fake_read_csv(url):
        if "DTWEXBGS" in url:
            return pd.DataFrame(
                {
                    "observation_date": ["2026-01-01", "2026-01-02"],
                    "DTWEXBGS": [120.0, 121.0],
                }
            )
        return pd.DataFrame(
            {
                "observation_date": ["2026-01-01", "2026-01-02"],
                "DGS10": [4.1, 4.2],
            }
        )

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    output = tmp_path / "cross_assets.csv"
    config = config_from_dict({"cross_assets": {"csv_path": str(output)}})
    result = sync_fred_cross_assets(config, "2026-01-01", "2026-01-02")
    assert result == output
    rows = output.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "time,dxy,us10y"
    assert "2026-01-02 06:00:00+08:00,120.0,4.1" in rows[1]
