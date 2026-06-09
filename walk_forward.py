from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .backtest import BacktestEngine
from .config import SystemConfig
from .models import Side
from .news import NewsCalendar


def yearly_periods(start: str, end: str, timezone: str) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start, tz=timezone)
    end_ts = pd.Timestamp(end, tz=timezone)
    periods: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []
    for year in range(start_ts.year, end_ts.year + 1):
        period_start = max(start_ts, pd.Timestamp(f"{year}-01-01", tz=timezone))
        period_end = min(end_ts, pd.Timestamp(f"{year}-12-31", tz=timezone))
        periods.append((year, period_start, period_end))
    return periods


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Side):
        return value.value
    return str(value)


def run_yearly_walk_forward(
    config: SystemConfig,
    data: pd.DataFrame,
    news_calendar: NewsCalendar,
    start: str,
    end: str,
    output_dir: str | Path,
) -> Path:
    run_id = datetime.now().strftime("walk-forward-%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    payload: list[dict[str, object]] = []
    for year, period_start, period_end in yearly_periods(start, end, config.data.timezone):
        subset = data.loc[
            (data.index >= period_start)
            & (data.index < period_end + pd.Timedelta(days=1))
        ]
        result = BacktestEngine(config, news_calendar).run(subset, features_ready=True)
        metrics = result.metrics
        row = {
            "year": year,
            "start": period_start.isoformat(),
            "end": period_end.isoformat(),
            "trades": int(metrics["trade_count"]),
            "net_pnl": float(metrics["net_pnl"]),
            "total_return": float(metrics["total_return"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "profit_factor": float(metrics["profit_factor"]),
            "win_rate": float(metrics["win_rate"]),
        }
        rows.append(row)
        payload.append(
            {
                "period": row,
                "trades": [asdict(trade) for trade in result.trades],
                "equity_curve": result.equity_curve,
            }
        )

    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    json_path = run_dir / "walk_forward.json"
    json_path.write_text(json.dumps(payload, default=_json_default, indent=2), encoding="utf-8")
    return run_dir
