from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .backtest import BacktestEngine, coverage_summary, load_market_csv, requested_coverage_ok
from .broker import PaperSignalLog
from .calendar_sync import CalendarSyncError, sync_calendar
from .config import load_config
from .cross_assets import add_cross_asset_features, load_cross_asset_csv, sync_fred_cross_assets
from .ibkr_adapter import (
    IBKRCrossAssetExporter,
    IBKRUnavailable,
    ib_gateway_path_exists,
    ibkr_port_open,
    ibkr_smoke,
)
from .indicators import compute_features
from .live_data import latest_bar_age_minutes, make_live_market_data_source
from .mt5_adapter import MT5MarketDataSource, MT5Unavailable, export_rates_to_csv
from .news import NewsCalendar
from .notifications import NotificationError, TelegramNotifier
from .paper import (
    PaperPositionStore,
    PaperTradeLog,
    evaluate_position_exit,
    position_from_order,
    recover_latest_open_position,
)
from .portfolio import run_portfolio_backtest
from .reporting import write_backtest_report
from .risk import RiskManager
from .selector import RuleBasedStrategySelector
from .strategies import BacktestContext, BreakoutStrategy, MeanReversionStrategy, MicroScalpStrategy
from .walk_forward import run_yearly_walk_forward


def _news_calendar(config):
    return NewsCalendar.from_csv(
        config.data.news_csv,
        config.data.timezone,
        config.filters.news_currencies,
        config.filters.news_impacts,
    )


def _parse_config_date(value: str | None, timezone: str):
    if not value:
        return None
    return pd.Timestamp(value, tz=timezone).to_pydatetime()


def _paper_signal_for_row(
    config,
    row,
    context: BacktestContext,
    selector: RuleBasedStrategySelector,
    breakout: BreakoutStrategy,
    mean_reversion: MeanReversionStrategy,
    micro_scalp: MicroScalpStrategy,
):
    mode = selector.choose(row)
    signal = None
    if mode == "trend":
        signal = breakout.generate(row, context)
        if signal is not None and not signal.is_entry:
            scalp_signal = micro_scalp.generate(row, context)
            if scalp_signal.is_entry:
                signal = scalp_signal
    elif mode == "range":
        signal = mean_reversion.generate(row, context)
        if signal is not None and not signal.is_entry:
            scalp_signal = micro_scalp.generate(row, context)
            if scalp_signal.is_entry:
                signal = scalp_signal
    elif mode == "scalp":
        signal = micro_scalp.generate(row, context)
    return mode, signal


def _paper_order_key(order) -> tuple[str, str, str, str, float]:
    return (
        order.timestamp.isoformat(),
        order.symbol,
        order.strategy,
        order.side.value,
        round(float(order.entry_price), 5),
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        data = load_market_csv(config.data.bars_csv, config, args.from_date, args.to_date)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print("Run export-mt5 first or update data.bars_csv in the config.", file=sys.stderr)
        return 2
    news = _news_calendar(config)
    if not news.available:
        print(f"Warning: news CSV not found: {config.data.news_csv}. Backtest skips news filtering.")
    summary = coverage_summary(data)
    ok, coverage_message = requested_coverage_ok(
        data, args.from_date, args.to_date, config.data.timezone
    )
    print(f"Data rows: {summary['rows']}")
    print(f"Data range: {summary['start']} -> {summary['end']}")
    if not ok:
        print(f"Warning: requested range is not fully covered: {coverage_message}", file=sys.stderr)
    result = BacktestEngine(config, news).run(data, features_ready=True)
    report_dir = write_backtest_report(result, config.report.output_dir)
    print(f"Run ID: {result.run_id}")
    print(f"Trades: {int(result.metrics['trade_count'])}")
    print(f"Ending equity: {result.metrics['ending_equity']:.2f}")
    print(f"Report: {report_dir / 'index.html'}")
    return 0


def cmd_walk_forward(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        data = load_market_csv(config.data.bars_csv, config, args.from_date, args.to_date)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    news = _news_calendar(config)
    if not news.available:
        print(f"Warning: news CSV not found: {config.data.news_csv}. Walk-forward skips news filtering.")
    run_dir = run_yearly_walk_forward(
        config,
        data,
        news,
        args.from_date,
        args.to_date,
        config.report.output_dir,
    )
    summary = pd.read_csv(run_dir / "summary.csv")
    print(f"Walk-forward report: {run_dir}")
    print(summary.to_string(index=False))
    return 0


def cmd_export_mt5(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    tz = config.data.timezone
    start = _parse_config_date(args.from_date, tz) or (
        pd.Timestamp.now(tz=tz) - pd.Timedelta(days=30)
    ).to_pydatetime()
    end = _parse_config_date(args.to_date, tz)
    end = (pd.Timestamp(end) + pd.Timedelta(days=1)).to_pydatetime() if end else pd.Timestamp.now(tz=tz).to_pydatetime()
    output = args.output or config.data.bars_csv
    path = export_rates_to_csv(
        config, args.symbol, args.timeframe, start, end, output, chunk_days=args.chunk_days
    )
    print(f"Exported MT5 rates to {path}")
    return 0


def cmd_sync_calendar(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    start = _parse_config_date(args.from_date, config.calendar.target_timezone)
    end = _parse_config_date(args.to_date, config.calendar.target_timezone)
    try:
        result = sync_calendar(
            config,
            start=start,
            end=end,
            provider=args.provider,
            force=args.force,
        )
    except CalendarSyncError as exc:
        print(f"Calendar sync failed: {exc}", file=sys.stderr)
        return 2
    cache_note = " using cache" if result.used_cache else ""
    print(f"Synced {len(result.events)} events from {result.provider}{cache_note}.")
    print(f"Output: {result.output_csv}")
    for event in result.events[:20]:
        print(f"{event.time.isoformat()} {event.currency} {event.impact} {event.event}")
    if len(result.events) > 20:
        print(f"... {len(result.events) - 20} more")
    return 0


def cmd_mt5_smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    source = MT5MarketDataSource(config)
    try:
        source.initialize()
        mt5 = source.mt5
        assert mt5 is not None
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        print("MT5 initialized: yes")
        print(f"Terminal: {getattr(terminal, 'name', None)}")
        print(f"Path: {getattr(terminal, 'path', None)}")
        print(f"Connected: {getattr(terminal, 'connected', None)}")
        if account is None:
            print(f"Account: not logged in ({mt5.last_error()})")
        else:
            print(f"Account: {account.login}")
            print(f"Server: {account.server}")
            print(f"Trade allowed: {account.trade_allowed}")

        candidates = source.matching_symbols()
        if candidates:
            print("Gold symbols: " + ", ".join(candidates[:20]))
        snapshot = source.symbol_snapshot(args.symbol)
        print(f"Symbol: {snapshot['name']}")
        print(f"Bid/Ask: {snapshot['bid']} / {snapshot['ask']}")
        print(f"Spread points: {snapshot['spread_points']}")
        bars = source.fetch_latest_rates(args.symbol, args.timeframe, args.bars)
        print(f"Latest {args.timeframe} bars: {len(bars)}")
        if not bars.empty:
            last = bars.iloc[-1]
            print(f"Last bar: {last['time']} close={last['close']} spread={last['spread']}")
    finally:
        source.shutdown()
    return 0


def cmd_cross_assets_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        cross_assets = load_cross_asset_csv(config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if cross_assets.empty:
        print(f"No cross-asset CSV found at {config.cross_assets.csv_path}.")
        print("Filter will be neutral because neutral_if_missing is enabled.")
        return 0
    market = pd.DataFrame(
        {
            "time": cross_assets.index,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "volume": 0.0,
            "spread": 0.0,
        }
    )
    merged = add_cross_asset_features(market.set_index("time"), config, cross_assets)
    latest = merged.iloc[-1]
    print(f"Rows: {len(cross_assets)}")
    print(f"Latest time: {merged.index[-1]}")
    print(f"DXY: {latest.get('dxy')}")
    print(f"US10Y: {latest.get('us10y')}")
    print(f"DXY momentum: {latest.get('dxy_momentum')}")
    print(f"Yield momentum: {latest.get('yield_momentum')}")
    print(f"Bias: {latest.get('cross_asset_bias')}")
    print(f"Reason: {latest.get('cross_asset_reason')}")
    return 0


def cmd_sync_fred_cross_assets(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = sync_fred_cross_assets(config, args.from_date, args.to_date, args.output)
    cross_assets = load_cross_asset_csv(config)
    print(f"Synced FRED cross-asset data to {output}")
    print(f"Rows: {len(cross_assets)}")
    if not cross_assets.empty:
        print(f"Range: {cross_assets.index[0]} -> {cross_assets.index[-1]}")
    return 0


def cmd_ibkr_smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"Gateway path configured: {config.ibkr.gateway_path}")
    print(f"Gateway path exists: {ib_gateway_path_exists(config)}")
    print(f"API target: {config.ibkr.host}:{config.ibkr.port} client_id={config.ibkr.client_id}")
    print(f"API port open: {ibkr_port_open(config)}")
    try:
        result = ibkr_smoke(config)
    except IBKRUnavailable as exc:
        print(f"IBKR smoke failed: {exc}", file=sys.stderr)
        print(
            "Start TWS/IB Gateway, log in, then enable API connections. "
            "Paper TWS usually uses port 7497; live TWS often uses 7496; "
            "IB Gateway paper often uses 4002.",
            file=sys.stderr,
        )
        return 2
    print("IBKR connected: yes")
    print(f"Next valid id: {result['next_valid_id']}")
    print(f"Probe symbol: {result['dxy_symbol']}")
    print(f"Historical bars: {result['bar_count']}")
    if result["last_error"]:
        print(f"Last API message: {result['last_error']}")
    return 0


def cmd_export_ibkr_cross_assets(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    exporter = IBKRCrossAssetExporter(config)
    try:
        output = exporter.export_csv(args.output)
    except IBKRUnavailable as exc:
        print(f"IBKR export failed: {exc}", file=sys.stderr)
        return 2
    print(f"Exported IBKR cross-asset data to {output}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report_path = Path(config.report.output_dir) / args.run_id / "index.html"
    if not report_path.exists():
        print(f"Report not found: {report_path}", file=sys.stderr)
        return 2
    print(report_path)
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    try:
        run_dir = run_portfolio_backtest(
            args.configs,
            args.from_date,
            args.to_date,
            args.output_dir,
            total_equity=args.total_equity,
            weights=args.weights,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Portfolio backtest failed: {exc}", file=sys.stderr)
        print(
            "Export missing symbols first with export-mt5 or update each config data.bars_csv.",
            file=sys.stderr,
        )
        return 2
    summary = pd.read_csv(run_dir / "summary.csv")
    payload = json.loads((run_dir / "portfolio_result.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    print(f"Portfolio report: {run_dir}")
    print(summary.to_string(index=False))
    print("Portfolio metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.calendar.auto_sync:
        try:
            result = sync_calendar(config)
            cache_note = " using cache" if result.used_cache else ""
            print(f"Calendar synced: {len(result.events)} events from {result.provider}{cache_note}.")
        except CalendarSyncError as exc:
            print(f"Calendar sync failed; refusing paper mode: {exc}", file=sys.stderr)
            return 2
    news = _news_calendar(config)
    if not news.available:
        print(
            f"Paper mode requires a news CSV at {config.data.news_csv}; refusing to run without news filtering.",
            file=sys.stderr,
        )
        return 2
    if not news.events:
        print(
            f"Warning: news CSV has no matching high-impact events: {config.data.news_csv}.",
            file=sys.stderr,
        )
    source = make_live_market_data_source(config)
    selector = RuleBasedStrategySelector(config)
    breakout = BreakoutStrategy(config)
    mean_reversion = MeanReversionStrategy(config)
    micro_scalp = MicroScalpStrategy(config)
    risk = RiskManager(config)
    report_dir = Path(config.report.output_dir)
    signal_log = PaperSignalLog(report_dir / "paper_signals.csv")
    trade_log = PaperTradeLog(report_dir / "paper_trades.csv")
    position_store = PaperPositionStore(report_dir / "paper_position.json")
    notifier = TelegramNotifier(config)
    telegram_status = notifier.status()
    if telegram_status.enabled and not telegram_status.ready:
        print(
            "Telegram is enabled but not configured. "
            f"Set {config.telegram.bot_token_env} and {config.telegram.chat_id_env}.",
            file=sys.stderr,
        )
        return 2
    if telegram_status.enabled:
        print("Telegram notifications: enabled")
    context = BacktestContext()
    equity = config.risk.initial_equity
    day_start_equity = equity
    cycles = 0
    sent_order_keys: set[tuple[str, str, str, str, float]] = set()
    active_position = position_store.load()
    if active_position is None:
        active_position = recover_latest_open_position(signal_log.path, trade_log, config)
        if active_position is not None:
            position_store.save(active_position)
            print(
                "Recovered open paper position: "
                f"{active_position.position_id} {active_position.side.value} "
                f"entry={active_position.entry_price:.2f} "
                f"sl={active_position.stop_loss:.2f} tp={active_position.take_profit:.2f}"
            )
    source.initialize()
    try:
        while True:
            try:
                raw = source.fetch_latest_rates(config.data.symbol, "M1", config.mt5.history_bars)
            except (IBKRUnavailable, MT5Unavailable) as exc:
                print(f"Live data unavailable; will retry next cycle: {exc}", file=sys.stderr)
                cycles += 1
                if args.max_cycles and cycles >= args.max_cycles:
                    break
                time.sleep(config.mt5.poll_seconds)
                continue
            age_minutes = latest_bar_age_minutes(raw, config.data.timezone)
            stale_live_bar = False
            if (
                age_minutes is not None
                and config.data.live_max_bar_age_minutes > 0
                and age_minutes > config.data.live_max_bar_age_minutes
            ):
                stale_live_bar = True
                print(
                    f"Latest bar is stale: {age_minutes:.1f} minutes old "
                    f"(limit {config.data.live_max_bar_age_minutes})."
                )
            data = compute_features(raw, config)
            if data.empty:
                print("No live bars returned.")
            else:
                row = data.iloc[-1]
                if active_position is not None:
                    trade = evaluate_position_exit(active_position, data, config)
                    if trade is not None:
                        trade_log.append(trade)
                        position_store.clear()
                        equity += trade.pnl
                        active_position = None
                        if telegram_status.enabled:
                            try:
                                notifier.send_trade_exit(trade)
                            except NotificationError as exc:
                                print(str(exc), file=sys.stderr)
                        print(
                            "Paper exit logged: "
                            f"{trade.trade_id} reason={trade.exit_reason} "
                            f"exit={trade.exit_price:.2f} pnl={trade.pnl:.2f}"
                        )
                    else:
                        position_store.save(active_position)
                        print(
                            "Open paper position: "
                            f"{active_position.position_id} {active_position.side.value} "
                            f"entry={active_position.entry_price:.2f} "
                            f"sl={active_position.stop_loss:.2f} "
                            f"tp={active_position.take_profit:.2f} "
                            f"last={row.name} close={row['close']:.2f}"
                        )

                if active_position is None:
                    if stale_live_bar:
                        print(
                            "New paper entries skipped because latest bar is stale: "
                            f"last={row.name} close={row['close']:.2f}"
                        )
                        cycles += 1
                        if args.max_cycles and cycles >= args.max_cycles:
                            break
                        time.sleep(config.mt5.poll_seconds)
                        continue
                    mode, signal = _paper_signal_for_row(
                        config,
                        row,
                        context,
                        selector,
                        breakout,
                        mean_reversion,
                        micro_scalp,
                    )
                    if signal is not None:
                        decision = risk.approve(
                            signal,
                            row,
                            equity,
                            day_start_equity,
                            None,
                            news,
                            paper_mode=True,
                        )
                        if decision.allowed and decision.order is not None:
                            order = decision.order
                            order_key = _paper_order_key(order)
                            if order_key in sent_order_keys:
                                print(f"Duplicate paper signal skipped: {order_key}")
                            else:
                                sent_order_keys.add(order_key)
                                signal_log.append(order)
                                active_position = position_from_order(
                                    order,
                                    signal.max_holding_minutes,
                                )
                                position_store.save(active_position)
                                context.increment(order.strategy, order.timestamp.date())
                                if telegram_status.enabled:
                                    try:
                                        notifier.send_order(order, mode=mode)
                                    except NotificationError as exc:
                                        print(str(exc), file=sys.stderr)
                                print(f"Paper signal logged: {order}")
                        else:
                            print(f"No paper order: {decision.reason}")
                    else:
                        print(f"No signal. Mode={mode}. Last bar={row.name} close={row['close']}")
            cycles += 1
            if args.max_cycles and cycles >= args.max_cycles:
                break
            time.sleep(config.mt5.poll_seconds)
    finally:
        source.shutdown()
    return 0


def cmd_paper_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ok = True
    print(f"Config: {args.config}")
    print(f"Symbol: {config.data.symbol}")
    print(f"Paper only: live order submission remains disabled")

    news = _news_calendar(config)
    print(f"News CSV: {config.data.news_csv} available={news.available} events={len(news.events)}")
    if not news.available:
        ok = False

    notifier = TelegramNotifier(config)
    telegram_status = notifier.status()
    print(
        "Telegram: "
        f"enabled={telegram_status.enabled} "
        f"token={telegram_status.bot_token_configured} "
        f"chat_id={telegram_status.chat_id_configured}"
    )
    if telegram_status.enabled and not telegram_status.ready:
        ok = False
    if args.send_telegram_test:
        try:
            result = notifier.send_healthcheck(args.config)
        except NotificationError as exc:
            print(str(exc), file=sys.stderr)
            ok = False
        else:
            print(f"Telegram test sent: {bool(result)}")

    if not args.skip_mt5:
        source = make_live_market_data_source(config)
        try:
            source.initialize()
            if hasattr(source, "symbol_snapshot"):
                snapshot = source.symbol_snapshot(config.data.symbol)
                print(f"MT5 symbol: {snapshot['name']}")
                print(f"MT5 bid/ask: {snapshot['bid']} / {snapshot['ask']}")
            else:
                print(f"Live source: {config.data.live_source}")
                print(f"Live CSV: {config.data.live_csv_path}")
            raw = source.fetch_latest_rates(config.data.symbol, "M1", config.mt5.history_bars)
            print(f"Latest bars: {len(raw)}")
            age_minutes = latest_bar_age_minutes(raw, config.data.timezone)
            if age_minutes is not None:
                print(f"Latest bar age: {age_minutes:.1f} minutes")
                if (
                    config.data.live_max_bar_age_minutes > 0
                    and age_minutes > config.data.live_max_bar_age_minutes
                ):
                    print(
                        f"Latest bar is stale: limit={config.data.live_max_bar_age_minutes} minutes",
                        file=sys.stderr,
                    )
                    ok = False
            data = compute_features(raw, config)
            if data.empty:
                print("Feature check failed: no computed rows", file=sys.stderr)
                ok = False
            else:
                selector = RuleBasedStrategySelector(config)
                breakout = BreakoutStrategy(config)
                mean_reversion = MeanReversionStrategy(config)
                micro_scalp = MicroScalpStrategy(config)
                context = BacktestContext()
                row = data.iloc[-1]
                mode, signal = _paper_signal_for_row(
                    config,
                    row,
                    context,
                    selector,
                    breakout,
                    mean_reversion,
                    micro_scalp,
                )
                print(f"Latest feature row: {row.name} close={float(row['close']):.2f}")
                print(f"Selector mode: {mode}")
                if signal is None:
                    print("Signal check: no signal for latest bar")
                else:
                    print(f"Signal check: {signal.strategy} entry={signal.is_entry} reason={signal.reason}")
                    decision = RiskManager(config).approve(
                        signal,
                        row,
                        config.risk.initial_equity,
                        config.risk.initial_equity,
                        None,
                        news,
                        paper_mode=True,
                    )
                    print(f"Risk check: allowed={decision.allowed} reason={decision.reason}")
                    if decision.order is not None:
                        print(
                            "Order intent: "
                            f"{decision.order.side.value} "
                            f"vol={decision.order.volume:.2f} "
                            f"entry={decision.order.entry_price:.2f} "
                            f"sl={decision.order.stop_loss:.2f} "
                            f"tp={decision.order.take_profit:.2f}"
                        )
        except (MT5Unavailable, RuntimeError, ValueError, FileNotFoundError) as exc:
            print(f"Live data check failed: {exc}", file=sys.stderr)
            ok = False
        finally:
            source.shutdown()

    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gold-scalper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--config", default="configs/default.yaml")
    backtest.add_argument("--from", dest="from_date", required=True)
    backtest.add_argument("--to", dest="to_date", required=True)
    backtest.set_defaults(func=cmd_backtest)

    walk_forward = subparsers.add_parser("walk-forward")
    walk_forward.add_argument("--config", default="configs/default.yaml")
    walk_forward.add_argument("--from", dest="from_date", required=True)
    walk_forward.add_argument("--to", dest="to_date", required=True)
    walk_forward.set_defaults(func=cmd_walk_forward)

    export = subparsers.add_parser("export-mt5")
    export.add_argument("--config", default="configs/default.yaml")
    export.add_argument("--symbol", default="XAUUSD")
    export.add_argument("--timeframe", default="M1")
    export.add_argument("--from", dest="from_date")
    export.add_argument("--to", dest="to_date")
    export.add_argument("--out", dest="output")
    export.add_argument("--chunk-days", type=int, default=30)
    export.set_defaults(func=cmd_export_mt5)

    calendar = subparsers.add_parser("sync-calendar")
    calendar.add_argument("--config", default="configs/default.yaml")
    calendar.add_argument(
        "--provider",
        choices=["fred_us_macro", "fxmacrodata", "forex_factory", "trading_economics"],
    )
    calendar.add_argument("--from", dest="from_date")
    calendar.add_argument("--to", dest="to_date")
    calendar.add_argument("--force", action="store_true")
    calendar.set_defaults(func=cmd_sync_calendar)

    smoke = subparsers.add_parser("mt5-smoke")
    smoke.add_argument("--config", default="configs/default.yaml")
    smoke.add_argument("--symbol", default="XAUUSD")
    smoke.add_argument("--timeframe", default="M1")
    smoke.add_argument("--bars", type=int, default=3)
    smoke.set_defaults(func=cmd_mt5_smoke)

    cross_assets = subparsers.add_parser("cross-assets-check")
    cross_assets.add_argument("--config", default="configs/default.yaml")
    cross_assets.set_defaults(func=cmd_cross_assets_check)

    fred_cross_assets = subparsers.add_parser("sync-fred-cross-assets")
    fred_cross_assets.add_argument("--config", default="configs/default.yaml")
    fred_cross_assets.add_argument("--from", dest="from_date")
    fred_cross_assets.add_argument("--to", dest="to_date")
    fred_cross_assets.add_argument("--out", dest="output")
    fred_cross_assets.set_defaults(func=cmd_sync_fred_cross_assets)

    ibkr_smoke_parser = subparsers.add_parser("ibkr-smoke")
    ibkr_smoke_parser.add_argument("--config", default="configs/default.yaml")
    ibkr_smoke_parser.set_defaults(func=cmd_ibkr_smoke)

    ibkr_export = subparsers.add_parser("export-ibkr-cross-assets")
    ibkr_export.add_argument("--config", default="configs/default.yaml")
    ibkr_export.add_argument("--out", dest="output")
    ibkr_export.set_defaults(func=cmd_export_ibkr_cross_assets)

    paper = subparsers.add_parser("paper")
    paper.add_argument("--config", default="configs/default.yaml")
    paper.add_argument("--max-cycles", type=int, default=1)
    paper.set_defaults(func=cmd_paper)

    paper_check = subparsers.add_parser("paper-check")
    paper_check.add_argument("--config", default="configs/default.yaml")
    paper_check.add_argument("--skip-mt5", action="store_true")
    paper_check.add_argument("--send-telegram-test", action="store_true")
    paper_check.set_defaults(func=cmd_paper_check)

    report = subparsers.add_parser("report")
    report.add_argument("--config", default="configs/default.yaml")
    report.add_argument("--run-id", required=True)
    report.set_defaults(func=cmd_report)

    portfolio = subparsers.add_parser("portfolio")
    portfolio.add_argument("--configs", nargs="+", required=True)
    portfolio.add_argument("--from", dest="from_date", required=True)
    portfolio.add_argument("--to", dest="to_date", required=True)
    portfolio.add_argument("--total-equity", type=float)
    portfolio.add_argument("--weights")
    portfolio.add_argument("--output-dir", default="reports")
    portfolio.set_defaults(func=cmd_portfolio)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
