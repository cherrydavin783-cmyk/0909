from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import BacktestResult, Side


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Side):
        return value.value
    return str(value)


def _equity_svg(equity_curve: list[dict[str, Any]], width: int = 900, height: int = 220) -> str:
    values = [float(point["equity"]) for point in equity_curve]
    if len(values) < 2:
        return "<p>Not enough equity points to draw a chart.</p>"
    low = min(values)
    high = max(values)
    span = high - low or 1.0
    points = []
    for idx, value in enumerate(values):
        x = idx * width / (len(values) - 1)
        y = height - ((value - low) / span * (height - 20)) - 10
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="equity curve">'
        f'<polyline fill="none" stroke="#0f766e" stroke-width="2" points="{" ".join(points)}" />'
        "</svg>"
    )


def write_backtest_report(result: BacktestResult, output_dir: str | Path) -> Path:
    run_dir = Path(output_dir) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trades_csv = run_dir / "trades.csv"
    with trades_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
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
        for trade in result.trades:
            row = asdict(trade)
            row["side"] = trade.side.value
            writer.writerow(row)

    equity_csv = run_dir / "equity.csv"
    with equity_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "equity"])
        writer.writeheader()
        for point in result.equity_curve:
            writer.writerow(point)

    result_json = run_dir / "backtest_result.json"
    payload = {
        "run_id": result.run_id,
        "symbol": result.symbol,
        "metrics": result.metrics,
        "trades": [asdict(trade) for trade in result.trades],
        "equity_curve": result.equity_curve,
    }
    result_json.write_text(json.dumps(payload, default=_json_default, indent=2), encoding="utf-8")

    metrics_rows = "\n".join(
        f"<tr><th>{name}</th><td>{value:.6g}</td></tr>"
        for name, value in result.metrics.items()
    )
    trade_rows = "\n".join(
        "<tr>"
        f"<td>{trade.entry_time}</td><td>{trade.strategy}</td><td>{trade.side.value}</td>"
        f"<td>{trade.volume:.2f}</td><td>{trade.entry_price:.2f}</td>"
        f"<td>{trade.exit_price:.2f}</td><td>{trade.pnl:.2f}</td>"
        f"<td>{trade.exit_reason}</td>"
        "</tr>"
        for trade in result.trades
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gold Scalper Report {result.run_id}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #172026; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .chart {{ border: 1px solid #d8dee4; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Gold Scalper Backtest</h1>
  <p>Run ID: <code>{result.run_id}</code> | Symbol: <code>{result.symbol}</code></p>
  <h2>Metrics</h2>
  <table>{metrics_rows}</table>
  <h2>Equity Curve</h2>
  <div class="chart">{_equity_svg(result.equity_curve)}</div>
  <h2>Trades</h2>
  <table>
    <tr><th>Entry Time</th><th>Strategy</th><th>Side</th><th>Volume</th><th>Entry</th><th>Exit</th><th>PNL</th><th>Reason</th></tr>
    {trade_rows}
  </table>
</body>
</html>
"""
    (run_dir / "index.html").write_text(html, encoding="utf-8")
    result.report_dir = str(run_dir)
    return run_dir
