from __future__ import annotations

from gold_scalper.cli import build_parser


def test_cli_parser_includes_mt5_smoke() -> None:
    parser = build_parser()
    args = parser.parse_args(["mt5-smoke", "--symbol", "XAUUSD", "--bars", "2"])
    assert args.command == "mt5-smoke"
    assert args.symbol == "XAUUSD"
    assert args.bars == 2


def test_cli_parser_export_mt5_chunk_days() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["export-mt5", "--from", "2021-01-01", "--to", "2021-01-31", "--chunk-days", "7"]
    )
    assert args.command == "export-mt5"
    assert args.chunk_days == 7


def test_cli_parser_includes_sync_calendar() -> None:
    parser = build_parser()
    args = parser.parse_args(["sync-calendar", "--provider", "fred_us_macro"])
    assert args.command == "sync-calendar"
    assert args.provider == "fred_us_macro"


def test_cli_parser_includes_walk_forward() -> None:
    parser = build_parser()
    args = parser.parse_args(["walk-forward", "--from", "2021-01-01", "--to", "2021-12-31"])
    assert args.command == "walk-forward"
    assert args.from_date == "2021-01-01"


def test_cli_parser_includes_ibkr_commands() -> None:
    parser = build_parser()
    smoke = parser.parse_args(["ibkr-smoke"])
    assert smoke.command == "ibkr-smoke"
    export = parser.parse_args(["export-ibkr-cross-assets", "--out", "data/out.csv"])
    assert export.command == "export-ibkr-cross-assets"
    assert export.output == "data/out.csv"


def test_cli_parser_includes_fred_cross_assets() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["sync-fred-cross-assets", "--from", "2021-01-01", "--to", "2026-05-19"]
    )
    assert args.command == "sync-fred-cross-assets"
    assert args.from_date == "2021-01-01"


def test_cli_parser_includes_portfolio() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "portfolio",
            "--configs",
            "configs/hybrid_scalping.yaml",
            "configs/xagusd_hybrid_scalping.yaml",
            "--from",
            "2021-05-19",
            "--to",
            "2026-05-19",
            "--total-equity",
            "10000",
            "--weights",
            "0.6,0.4",
        ]
    )
    assert args.command == "portfolio"
    assert args.configs == ["configs/hybrid_scalping.yaml", "configs/xagusd_hybrid_scalping.yaml"]
    assert args.total_equity == 10000
    assert args.weights == "0.6,0.4"


def test_cli_parser_includes_paper_check() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["paper-check", "--config", "configs/high_frequency_025_price_aggressive.yaml", "--skip-mt5"]
    )
    assert args.command == "paper-check"
    assert args.config == "configs/high_frequency_025_price_aggressive.yaml"
    assert args.skip_mt5
