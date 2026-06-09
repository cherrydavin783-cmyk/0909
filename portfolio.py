from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .backtest import BacktestEngine, _calculate_metrics, load_market_csv
from .config import SystemConfig, load_config
from .models import BacktestResult, TradeRecord
from .news import NewsCalendar
from .reporting import _equity_svg, _json_default


def _news_calendar(config: SystemConfig) -> NewsCalendar:
    return NewsCalendar.from_csv(
        config.data.news_csv,
        config.data.timezone,
        config.filters.news_currencies,
        config.filters.news_impacts,
    )


def _weights(values: str | None, count: int) -> list[float]:
    if count <= 0:
        return []
    if not values:
        return [1 / count] * count
    parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    if len(parsed) != count:
        raise ValueError(f"Expected {count} portfolio weights, got {len(parsed)}")
    total = sum(parsed)
    if total <= 0.0:
        raise ValueError("Portfolio weights must sum to a positive value")
    return [value / total for value in parsed]


def _allocated_config(config: SystemConfig, allocation: float) -> SystemConfig:
    config.risk.initial_equity = float(allocation)
    return config


def _equity_series(result: BacktestResult, initial_equity: float) -> pd.Series:
    if not result.equity_curve:
        return pd.Series(dtype=float)
    series = pd.Series(
        [float(point["equity"]) for point in result.equity_curve],
        index=pd.DatetimeIndex([point["time"] for point in result.equity_curve]),
        name=result.symbol,
    ).sort_index()
    if not series.empty:
        series.iloc[0] = series.iloc[0] if pd.notna(series.iloc[0]) else initial_equity
    return series


def _combine_equity(
    results: list[BacktestResult],
    allocations: list[float],
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    component_inputs = [
        (result, allocation, _equity_series(result, allocation))
        for result, allocation in zip(results, allocations, strict=True)
    ]
    component_inputs = [
        item for item in component_inputs
        if not item[2].empty
    ]
    if not component_inputs:
        return [], pd.DataFrame(), pd.DataFrame()
    index = component_inputs[0][2].index
    for _, _, series in component_inputs[1:]:
        index = index.union(series.index)
    components = pd.DataFrame(index=index.sort_values())
    for result, allocation, series in component_inputs:
        components[result.symbol] = series.reindex(components.index).ffill().fillna(allocation)
    combined = components.sum(axis=1)
    equity_curve = [
        {"time": timestamp.to_pydatetime(), "equity": float(value)}
        for timestamp, value in combined.items()
    ]
    daily_returns = components.resample("1D").last().ffill().pct_change().dropna(how="all")
    correlations = daily_returns.corr() if not daily_returns.empty else pd.DataFrame()
    return equity_curve, components, correlations


def run_portfolio_backtest(
    config_paths: list[str | Path],
    start: str,
    end: str,
    output_dir: str | Path,
    total_equity: float | None = None,
    weights: str | None = None,
) -> Path:
    if not config_paths:
        raise ValueError("At least one config path is required")

    configs = [load_config(path) for path in config_paths]
    weight_values = _weights(weights, len(configs))
    if total_equity is None:
        allocations = [config.risk.initial_equity for config in configs]
    else:
        allocations = [float(total_equity) * weight for weight in weight_values]
    configs = [
        _allocated_config(config, allocation)
        for config, allocation in zip(configs, allocations, strict=True)
    ]
    missing = [
        str(Path(config.data.bars_csv))
        for config in configs
        if not Path(config.data.bars_csv).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Market data CSV not found: {', '.join(missing)}")

    results: list[BacktestResult] = []
    for config in configs:
        data = load_market_csv(config.data.bars_csv, config, start, end)
        result = BacktestEngine(config, _news_calendar(config)).run(data, features_ready=True)
        results.append(result)

    all_trades: list[TradeRecord] = []
    for result in results:
        all_trades.extend(result.trades)
    all_trades.sort(key=lambda trade: trade.exit_time)

    equity_curve, components, correlations = _combine_equity(results, allocations)
    metrics = _calculate_metrics(all_trades, equity_curve, sum(allocations))
    run_id = datetime.now().strftime("portfolio-%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for path, allocation, result in zip(config_paths, allocations, results, strict=True):
        row = {
            "config": str(path),
            "symbol": result.symbol,
            "allocation": allocation,
            **result.metrics,
        }
        summary_rows.append(row)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    trades_rows = []
    for trade in all_trades:
        row = asdict(trade)
        row["side"] = trade.side.value
        trades_rows.append(row)
    with (run_dir / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(trades_rows[0].keys()) if trades_rows else [
            "trade_id",
            "symbol",
            "side",
            "strategy",
            "volume",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "pnl",
            "exit_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades_rows)

    pd.DataFrame(equity_curve).to_csv(run_dir / "equity.csv", index=False)
    if not components.empty:
        components.to_csv(run_dir / "component_equity.csv", index_label="time")
    if not correlations.empty:
        correlations.to_csv(run_dir / "correlations.csv")

    payload = {
        "run_id": run_id,
        "metrics": metrics,
        "summary": summary_rows,
        "correlations": correlations.to_dict() if not correlations.empty else {},
    }
    (run_dir / "portfolio_result.json").write_text(
        json.dumps(payload, default=_json_default, indent=2),
        encoding="utf-8",
    )

    metrics_rows = "\n".join(
        f"<tr><th>{name}</th><td>{value:.6g}</td></tr>"
        for name, value in metrics.items()
    )
    summary_html = "\n".join(
        "<tr>"
        f"<td>{row['symbol']}</td><td>{row['allocation']:.2f}</td>"
        f"<td>{int(row['trade_count'])}</td><td>{row['total_return']:.2%}</td>"
        f"<td>{row['max_drawdown']:.2%}</td><td>{row['profit_factor']:.4g}</td>"
        "</tr>"
        for row in summary_rows
    )
    corr_html = (
        correlations.round(4).to_html()
        if not correlations.empty
        else "<p>Not enough overlapping daily equity points for correlation.</p>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gold Scalper Portfolio {run_id}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #172026; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .chart {{ border: 1px solid #d8dee4; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Gold Scalper Portfolio Backtest</h1>
  <p>Run ID: <code>{run_id}</code></p>
  <h2>Portfolio Metrics</h2>
  <table>{metrics_rows}</table>
  <h2>Combined Equity Curve</h2>
  <div class="chart">{_equity_svg(equity_curve)}</div>
  <h2>Components</h2>
  <table>
    <tr><th>Symbol</th><th>Allocation</th><th>Trades</th><th>Return</th><th>Max DD</th><th>PF</th></tr>
    {summary_html}
  </table>
  <h2>Daily Equity Correlation</h2>
  {corr_html}
</body>
</html>
"""
    (run_dir / "index.html").write_text(html, encoding="utf-8")
    return run_dir
