# Gold Scalper

Python strategy system for XAUUSD based on the supplied plan. The first
version is intentionally limited to backtesting and paper/signal simulation.
It never sends live orders.

## What is included

- London/New York overlap breakout strategy.
- Low-liquidity mean-reversion strategy.
- Beijing-time session rotation.
- Tick/M1 approximations for OFI and VPIN when Level-2 data is unavailable.
- Spread, ATR, VPIN, news-window and daily drawdown filters.
- Optional breakeven stop migration after a favorable price move.
- Optional cross-asset filter using DXY and US 10Y momentum.
- Rule-based strategy selector: `trend`, `range`, `toxic`, `idle`.
- MT5 market-data adapter with live order submission disabled.
- CLI backtest/export/report/paper commands.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For MT5 export or paper mode:

```powershell
python -m pip install -e ".[mt5]"
```

If `MetaTrader5` cannot be installed on Python 3.13, install Python 3.12 x64
and create the virtual environment with that interpreter.

## Commands

```powershell
gold-scalper backtest --config configs/default.yaml --from 2026-01-01 --to 2026-01-31
gold-scalper walk-forward --config configs/default.yaml --from 2021-05-19 --to 2026-05-19
gold-scalper backtest --config configs/aggressive.yaml --from 2021-05-19 --to 2026-05-19
gold-scalper backtest --config configs/scalping.yaml --from 2021-05-19 --to 2026-05-19
gold-scalper backtest --config configs/hybrid_scalping.yaml --from 2021-05-19 --to 2026-05-19
gold-scalper backtest --config configs/high_frequency_scalping.yaml --from 2021-05-19 --to 2026-05-19
gold-scalper backtest --config configs/high_frequency_020_price_stable.yaml --from 2021-05-19 --to 2026-05-19
gold-scalper backtest --config configs/high_frequency_025_price_aggressive.yaml --from 2021-05-19 --to 2026-05-19
gold-scalper sync-calendar --config configs/default.yaml
gold-scalper sync-calendar --config configs/default.yaml --provider fred_us_macro --from 2021-05-19 --to 2026-05-19
gold-scalper cross-assets-check --config configs/default.yaml
gold-scalper sync-fred-cross-assets --config configs/default.yaml --from 2021-01-01 --to 2026-05-19
gold-scalper ibkr-smoke --config configs/default.yaml
gold-scalper export-ibkr-cross-assets --config configs/default.yaml
gold-scalper mt5-smoke --config configs/default.yaml --symbol XAUUSD
gold-scalper paper-check --config configs/high_frequency_025_price_aggressive.yaml --send-telegram-test
gold-scalper export-mt5 --symbol XAUUSD --timeframe M1 --from 2026-01-01 --to 2026-01-31
gold-scalper export-mt5 --config configs/xagusd_hybrid_scalping.yaml --symbol XAGUSD --timeframe M1 --from 2021-05-19 --to 2026-05-19 --out data/xagusd_m1.csv
gold-scalper export-mt5 --config configs/usoil_hybrid_scalping.yaml --symbol XTIUSD --timeframe M1 --from 2021-05-19 --to 2026-05-19 --out data/usoil_m1.csv
gold-scalper portfolio --configs configs/hybrid_scalping.yaml configs/xagusd_hybrid_scalping.yaml configs/usoil_hybrid_scalping.yaml --from 2021-05-19 --to 2026-05-19 --total-equity 30000
gold-scalper paper --config configs/default.yaml
gold-scalper report --run-id <run_id>
```

The default backtest CSV path is `data/xauusd_m1.csv`. It must contain a
`time` column plus OHLC columns and either `volume` or `tick_volume`.

Set `risk.profit_target_equity_pct` to a positive value to override strategy
TP with a fixed account-equity target per trade. For example, `0.003` exits at
about 0.3% net account profit after estimated commission and slippage. It is
disabled by default because the current five-year test favors the ATR target.
Set `risk.profit_target_price_pct` to a positive value to override strategy TP
with a fixed favorable price move from estimated entry fill. For example,
`0.0025` exits after about a 0.25% favorable XAUUSD price move. Use
`risk.strategy_profit_target_price_pct` to apply this only to selected
strategies, such as `micro_scalp`. Price targets take precedence over
account-equity targets when both are configured.
Set `filters.max_atr_quantile` below `1.0` to block entries when current ATR is
in the extreme tail of its rolling history, and set `filters.max_atr_abs` to a
positive XAUUSD price distance for a hard ATR cap. The shipped configs use
`0.95` plus a wide absolute cap as a volatility shock guard.
Set `risk.breakeven_trigger_price_pct` to a positive value to move the stop to
estimated breakeven after price moves in favor of the trade. The shipped
configs use `0.0015`, equal to a 0.15% favorable price move, with
`risk.breakeven_buffer_points` adding a small point buffer beyond estimated
commission and exit slippage.
The default config is the balanced profile. `configs/aggressive.yaml` uses the
same entry logic with 2% risk per trade for higher return and higher drawdown.
`configs/scalping.yaml` is a separate research profile for strict M1
micro-scalping: short rolling-range breakouts, OFI confirmation, volume spike,
VPIN/spread/news filters, and a three-minute maximum holding time.
`configs/hybrid_scalping.yaml` keeps the main breakout profile and lets the
low-risk micro-scalper run only when the main breakout has no entry. It uses a
more active New York window to create a larger sample while keeping the
volume-spike requirement. `configs/high_frequency_scalping.yaml` is a research
profile for higher trade count; expect weaker profit factor and larger noise
sensitivity than the balanced hybrid profile.
`configs/high_frequency_020_price_stable.yaml` is the screened stability
profile: micro-scalp exits after about a 0.20% favorable price move, allows an
eight-minute micro-scalp hold, keeps baseline risk, and passed the current
retrospective yearly/monthly stability filters.
`configs/high_frequency_025_price_aggressive.yaml` is a higher-risk research
profile: micro-scalp exits after about a 0.25% favorable price move, allows an
eight-minute micro-scalp hold, and raises position risk. Treat it as a
stress-test profile, not a live preset.
`configs/xagusd_hybrid_scalping.yaml` and `configs/usoil_hybrid_scalping.yaml`
are research templates for multi-asset replication. They scale absolute
range/spread/ATR thresholds to silver and oil price units, but broker symbol
names, contract sizes, point values, spreads and commissions must be checked
with `mt5-smoke` before trusting results.

The `portfolio` command runs several configs independently, combines their
equity curves, writes component-level metrics, and computes daily equity
correlations:

```powershell
gold-scalper portfolio --configs configs/hybrid_scalping.yaml configs/xagusd_hybrid_scalping.yaml configs/usoil_hybrid_scalping.yaml --from 2021-05-19 --to 2026-05-19 --total-equity 30000
```

Use `--weights 0.4,0.3,0.3` with `--total-equity` to test non-equal capital
allocation. The command refuses to run if any configured `data.bars_csv` is
missing, because portfolio results should not be inferred from placeholder
returns.

Cross-asset filter CSV format:

```csv
time,dxy,us10y
2026-05-19T20:00:00+08:00,100.20,4.48
```

If `data/cross_assets.csv` is missing, the filter stays neutral by default.
When present, rising DXY and rising 10Y yield are treated as bearish for gold;
falling DXY and falling 10Y yield are treated as bullish for gold. The default
configuration keeps this as an audit signal; set `cross_assets.block_on_conflict`
to `true` if you want conflicting macro bias to reject entries.
The `sync-fred-cross-assets` command fills this file from FRED using
`DTWEXBGS` as a broad-dollar proxy and `DGS10` as the 10Y Treasury yield.

IBKR export uses TWS/IB Gateway on `127.0.0.1:7497` by default. Enable API
connections in TWS/Gateway, keep it logged in, then run `ibkr-smoke`. The
default contracts use `UUP` as a USD proxy and `IEF` inverted as a 10Y yield
proxy, so the generated CSV still matches the `dxy,us10y` filter schema.

News CSV format:

```csv
time,currency,impact,event
2026-01-10 21:30:00,USD,high,CPI
```

Times are interpreted in `Asia/Shanghai` unless the timestamp contains an
explicit timezone offset.

Paper mode refuses to run unless `data/news.csv` exists. By default, paper
mode attempts to refresh the real calendar first with the `fxmacrodata`
provider and writes selected high-impact USD events into `data/news.csv`.
Set `telegram.enabled` to `true` and provide `TELEGRAM_BOT_TOKEN` plus
`TELEGRAM_CHAT_ID` to send approved paper signals to Telegram. Use
`paper-check --send-telegram-test` to verify MT5 market data, news filtering,
signal/risk evaluation and Telegram delivery before running continuously. See
`docs/vps_paper_telegram_deploy.md` for VPS setup.

For historical backtests, use the `fred_us_macro` provider. It pulls CPI and
Employment Situation release times from BLS schedule pages, FOMC statement and
minutes times from the Federal Reserve calendar, and falls back to FRED release
calendar pages if BLS is unavailable.

The `forex_factory` provider is available as a backup, and the `trading_economics`
provider is available if you have a paid API client. Set `TRADING_ECONOMICS_CLIENT`
or `calendar.trading_economics_client` in the config, then run:

```powershell
gold-scalper sync-calendar --config configs/default.yaml --provider trading_economics
```

IBKR has economic event views in TWS/mobile, but the public API path is not a
good first source for automated macro calendar filtering. Use the calendar
sync above for macro blackout windows; use IBKR later for cross-asset market
data such as ZN/IEF/DXY-style filters.
