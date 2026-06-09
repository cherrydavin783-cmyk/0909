from __future__ import annotations

import pandas as pd

from gold_scalper.backtest import BacktestEngine, requested_coverage_ok
from gold_scalper.config import config_from_dict
from gold_scalper.news import NewsCalendar
from gold_scalper.reporting import write_backtest_report


def _synthetic_breakout_data() -> pd.DataFrame:
    rows = []
    for ts in pd.date_range("2026-01-05 06:00", "2026-01-05 16:00", freq="1min", tz="Asia/Shanghai"):
        rows.append(
            {
                "time": ts,
                "open": 2008.0,
                "high": 2020.0,
                "low": 2000.0,
                "close": 2010.0,
                "volume": 10.0,
                "spread": 0.3,
            }
        )
    for ts in pd.date_range("2026-01-05 16:01", "2026-01-05 20:49", freq="1min", tz="Asia/Shanghai"):
        rows.append(
            {
                "time": ts,
                "open": 2010.0,
                "high": 2012.0,
                "low": 2008.0,
                "close": 2010.0,
                "volume": 10.0,
                "spread": 0.3,
            }
        )
    rows.append(
        {
            "time": pd.Timestamp("2026-01-05 20:50", tz="Asia/Shanghai"),
            "open": 2019.0,
            "high": 2021.2,
            "low": 2018.8,
            "close": 2021.0,
            "volume": 100.0,
            "spread": 0.3,
        }
    )
    rows.append(
        {
            "time": pd.Timestamp("2026-01-05 20:51", tz="Asia/Shanghai"),
            "open": 2021.0,
            "high": 2024.0,
            "low": 2020.8,
            "close": 2023.0,
            "volume": 80.0,
            "spread": 0.3,
        }
    )
    return pd.DataFrame(rows)


def test_backtest_generates_trade_and_report(tmp_path) -> None:
    config = config_from_dict(
        {
            "strategies": {
                "breakout": {"min_asia_width": 15.0, "min_rsi": 0.0, "max_rsi": 100.0}
            },
            "filters": {"toxic_vpin_quantile": 2.0},
            "report": {"output_dir": str(tmp_path)},
        }
    )
    result = BacktestEngine(config, NewsCalendar([])).run(_synthetic_breakout_data())
    assert result.metrics["trade_count"] >= 1
    assert result.trades[0].strategy == "breakout"
    report_dir = write_backtest_report(result, tmp_path)
    assert (report_dir / "index.html").exists()
    assert (report_dir / "trades.csv").exists()
    assert (report_dir / "equity.csv").exists()


def test_requested_coverage_detects_truncated_data() -> None:
    data = pd.DataFrame(
        {"close": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-02-01", tz="Asia/Shanghai")]),
    )
    ok, message = requested_coverage_ok(data, "2021-05-19", "2026-05-19", "Asia/Shanghai")
    assert not ok
    assert "data starts" in message


def test_backtest_exits_on_quote_gap() -> None:
    config = config_from_dict(
        {
            "risk": {"max_quote_gap_minutes": 10},
            "strategies": {
                "breakout": {"enabled": False},
                "mean_reversion": {
                    "enabled": True,
                    "windows": [["06:00", "07:00"]],
                    "max_vpin_quantile": 1.0,
                },
            },
            "filters": {"toxic_vpin_quantile": 2.0},
        }
    )
    rows = [
        {
            "time": pd.Timestamp("2026-01-05 06:10", tz="Asia/Shanghai"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
            "spread": 0.1,
            "atr": 1.0,
            "rsi": 50.0,
            "session": "asia",
            "volume_median": 10.0,
            "volume_spike": True,
            "fair_value": 99.0,
            "spread_median": 0.1,
            "ofi": 10.0,
            "ofi_ratio": 4.0,
            "vpin": 0.1,
            "vpin_quantile": 0.1,
            "asia_high": 101.0,
            "asia_low": 99.0,
            "asia_width": 2.0,
            "cross_asset_bias": "missing",
            "cross_asset_reason": "missing",
            "dxy_momentum": 0.0,
            "yield_momentum": 0.0,
            "cross_asset_score": 0,
        },
        {
            "time": pd.Timestamp("2026-01-05 06:30", tz="Asia/Shanghai"),
            "open": 100.5,
            "high": 100.6,
            "low": 100.4,
            "close": 100.5,
            "volume": 10.0,
            "spread": 0.1,
            "atr": 1.0,
            "rsi": 50.0,
            "session": "asia",
            "volume_median": 10.0,
            "volume_spike": False,
            "fair_value": 100.0,
            "spread_median": 0.1,
            "ofi": 0.0,
            "ofi_ratio": 0.0,
            "vpin": 0.1,
            "vpin_quantile": 0.1,
            "asia_high": 101.0,
            "asia_low": 99.0,
            "asia_width": 2.0,
            "cross_asset_bias": "missing",
            "cross_asset_reason": "missing",
            "dxy_momentum": 0.0,
            "yield_momentum": 0.0,
            "cross_asset_score": 0,
        },
    ]
    frame = pd.DataFrame(rows)
    frame = frame.set_index(pd.DatetimeIndex(frame.pop("time")))
    result = BacktestEngine(config, NewsCalendar([])).run(frame, features_ready=True)
    assert result.trades
    assert result.trades[0].exit_reason == "quote_gap"


def test_backtest_moves_stop_to_breakeven_after_trigger() -> None:
    config = config_from_dict(
        {
            "risk": {
                "risk_per_trade": 0.01,
                "breakeven_trigger_price_pct": 0.0015,
                "breakeven_buffer_points": 0.0,
                "commission_per_lot": 0.0,
                "slippage_abs": 0.0,
            },
            "strategies": {
                "breakout": {"enabled": False},
                "mean_reversion": {
                    "enabled": True,
                    "windows": [["06:00", "07:00"]],
                    "max_vpin_quantile": 1.0,
                    "max_holding_minutes": 10,
                },
            },
            "filters": {"toxic_vpin_quantile": 2.0},
        }
    )

    def row(timestamp: str, **overrides):
        data = {
            "time": pd.Timestamp(timestamp, tz="Asia/Shanghai"),
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 10.0,
            "spread": 0.0,
            "atr": 1.0,
            "atr_quantile": 0.1,
            "rsi": 50.0,
            "session": "asia",
            "volume_median": 10.0,
            "volume_spike": True,
            "fair_value": 101.0,
            "spread_median": 0.0,
            "ofi": -10.0,
            "ofi_ratio": -4.0,
            "vpin": 0.1,
            "vpin_quantile": 0.1,
            "asia_high": 101.0,
            "asia_low": 99.0,
            "asia_width": 2.0,
            "cross_asset_bias": "missing",
            "cross_asset_reason": "missing",
            "dxy_momentum": 0.0,
            "yield_momentum": 0.0,
            "cross_asset_score": 0,
        }
        data.update(overrides)
        return data

    frame = pd.DataFrame(
        [
            row("2026-01-05 06:10"),
            row("2026-01-05 06:11", high=100.2, low=100.05, ofi_ratio=0.0),
            row("2026-01-05 06:12", high=100.05, low=99.95, ofi_ratio=0.0),
        ]
    )
    frame = frame.set_index(pd.DatetimeIndex(frame.pop("time")))
    result = BacktestEngine(config, NewsCalendar([])).run(frame, features_ready=True)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "breakeven_stop"
    assert trade.exit_price == trade.entry_price
    assert trade.pnl == 0.0
