from __future__ import annotations

import json

import pandas as pd
import yaml

from gold_scalper.portfolio import run_portfolio_backtest


def _bars(start: str, close: float) -> pd.DataFrame:
    rows = []
    for idx, ts in enumerate(pd.date_range(start, periods=40, freq="1min", tz="Asia/Shanghai")):
        value = close + (idx % 3) * 0.1
        rows.append(
            {
                "time": ts.isoformat(),
                "open": value,
                "high": value + 0.2,
                "low": value - 0.2,
                "close": value,
                "volume": 10 + idx,
                "spread": 0.01,
            }
        )
    return pd.DataFrame(rows)


def _write_config(tmp_path, symbol: str, bars_csv: str) -> str:
    path = tmp_path / f"{symbol.lower()}.yaml"
    payload = {
        "data": {
            "symbol": symbol,
            "bars_csv": bars_csv,
            "news_csv": str(tmp_path / "missing_news.csv"),
            "timezone": "Asia/Shanghai",
            "input_timezone": "Asia/Shanghai",
            "default_spread": 0.01,
            "contract_size": 1.0,
        },
        "cross_assets": {"enabled": False},
        "strategies": {
            "breakout": {"enabled": False},
            "mean_reversion": {"enabled": False},
            "micro_scalp": {"enabled": False},
        },
        "risk": {"initial_equity": 1000.0},
        "report": {"output_dir": str(tmp_path / "reports")},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_portfolio_backtest_writes_combined_outputs(tmp_path) -> None:
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    _bars("2026-01-05 20:00", 100.0).to_csv(first_csv, index=False)
    _bars("2026-01-05 20:00", 50.0).to_csv(second_csv, index=False)

    first_config = _write_config(tmp_path, "AAA", str(first_csv))
    second_config = _write_config(tmp_path, "BBB", str(second_csv))
    run_dir = run_portfolio_backtest(
        [first_config, second_config],
        "2026-01-05",
        "2026-01-05",
        tmp_path / "reports",
        total_equity=10000.0,
    )

    assert (run_dir / "index.html").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "equity.csv").exists()
    assert (run_dir / "component_equity.csv").exists()
    payload = json.loads((run_dir / "portfolio_result.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["initial_equity"] == 10000.0
    summary = pd.read_csv(run_dir / "summary.csv")
    assert set(summary["symbol"]) == {"AAA", "BBB"}
    assert summary["allocation"].sum() == 10000.0
