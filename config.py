from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    symbol: str = "XAUUSD"
    bars_csv: str = "data/xauusd_m1.csv"
    news_csv: str = "data/news.csv"
    timezone: str = "Asia/Shanghai"
    input_timezone: str = "Asia/Shanghai"
    default_spread: float = 0.30
    point: float = 0.01
    contract_size: float = 100.0
    live_source: str = "mt5"
    live_csv_path: str = ""
    live_csv_time_column: str = "time"
    live_max_bar_age_minutes: int = 15


@dataclass
class CalendarConfig:
    provider: str = "fxmacrodata"
    auto_sync: bool = True
    output_csv: str = "data/news.csv"
    cache_file: str = "data/calendar_cache.xml"
    refresh_hours: int = 6
    lookback_days: int = 1
    lookahead_days: int = 7
    source_timezone: str = "UTC"
    target_timezone: str = "Asia/Shanghai"
    fxmacrodata_url: str = "https://fxmacrodata.com/api/v1/calendar/USD"
    fxmacrodata_high_impact_releases: list[str] = field(
        default_factory=lambda: [
            "policy_rate",
            "non_farm_payrolls",
            "unemployment",
            "employment",
            "inflation",
            "core_inflation",
            "inflation_mom",
            "core_inflation_mom",
            "pce",
            "pce_mom",
            "gdp",
            "ppi",
            "ppi_mom",
            "retail_sales",
            "durable_goods_orders",
            "consumer_sentiment",
            "job_openings",
            "pmi",
            "nmi",
        ]
    )
    forex_factory_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    trading_economics_client: str | None = None
    trading_economics_client_env: str = "TRADING_ECONOMICS_CLIENT"
    trading_economics_country: str = "united states"
    trading_economics_importance: int = 3
    bls_schedule_url: str = "https://www.bls.gov/schedule/{year}/home.htm"
    fred_calendar_url: str = "https://fred.stlouisfed.org/releases/calendar"
    fomc_calendar_url: str = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"


@dataclass
class CrossAssetConfig:
    enabled: bool = True
    csv_path: str = "data/cross_assets.csv"
    neutral_if_missing: bool = True
    block_on_conflict: bool = False
    require_confirmation: bool = False
    dxy_column: str = "dxy"
    yield_column: str = "us10y"
    momentum_lookback: int = 20
    asof_tolerance_minutes: int = 4320
    dxy_min_change: float = 0.05
    yield_min_change: float = 0.01
    required_components: int = 2


@dataclass
class IndicatorConfig:
    atr_period: int = 14
    atr_quantile_lookback: int = 500
    rsi_period: int = 14
    ofi_window: int = 20
    vpin_window: int = 50
    vpin_quantile_lookback: int = 500
    volume_window: int = 20
    volume_spike_factor: float = 1.5
    fair_value_window: int = 20
    spread_window: int = 50


@dataclass
class BreakoutConfig:
    enabled: bool = True
    asia_range_start: str = "06:00"
    asia_range_end: str = "16:00"
    trade_start: str = "20:00"
    trade_end: str = "21:30"
    min_asia_width: float = 25.0
    min_rsi: float = 40.0
    max_rsi: float = 60.0
    stop_atr_multiple: float = 0.8
    take_profit_atr_multiple: float = 2.4
    max_holding_minutes: int = 10
    max_daily_trades: int = 8


@dataclass
class MeanReversionConfig:
    enabled: bool = False
    windows: list[list[str]] = field(
        default_factory=lambda: [["01:00", "03:00"], ["06:00", "07:00"]]
    )
    ofi_ratio_threshold: float = 3.0
    fair_value_atr_deviation: float = 0.25
    max_vpin_quantile: float = 0.70
    stop_atr_multiple: float = 0.5
    max_stop_distance: float = 5.0
    max_holding_minutes: int = 5


@dataclass
class MicroScalpConfig:
    enabled: bool = False
    windows: list[list[str]] = field(default_factory=lambda: [["20:00", "21:30"]])
    lookback: int = 8
    min_asia_width: float = 25.0
    min_rsi_long: float = 55.0
    max_rsi_short: float = 45.0
    min_ofi_abs: float = 50.0
    max_vpin_quantile: float = 0.75
    require_volume_spike: bool = True
    stop_atr_multiple: float = 0.45
    take_profit_atr_multiple: float = 0.65
    max_holding_minutes: int = 3
    max_daily_trades: int = 20
    risk_per_trade: float = 0.002


@dataclass
class StrategyConfig:
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    micro_scalp: MicroScalpConfig = field(default_factory=MicroScalpConfig)


@dataclass
class FilterConfig:
    toxic_vpin_quantile: float = 0.90
    max_atr_abs: float = 0.0
    max_atr_quantile: float = 1.0
    max_spread_multiple: float = 1.5
    max_spread_abs: float = 2.5
    max_slippage_abs: float = 2.0
    news_before_minutes: int = 15
    news_after_minutes: int = 30
    news_currencies: list[str] = field(default_factory=lambda: ["USD"])
    news_impacts: list[str] = field(default_factory=lambda: ["high"])


@dataclass
class RiskConfig:
    initial_equity: float = 10000.0
    risk_per_trade: float = 0.015
    profit_target_equity_pct: float = 0.0
    strategy_profit_target_equity_pct: dict[str, float] = field(default_factory=dict)
    profit_target_price_pct: float = 0.0
    strategy_profit_target_price_pct: dict[str, float] = field(default_factory=dict)
    breakeven_trigger_price_pct: float = 0.0
    breakeven_buffer_points: float = 0.0
    max_daily_drawdown: float = 0.03
    min_volume: float = 0.01
    max_volume: float = 10.0
    volume_step: float = 0.01
    commission_per_lot: float = 3.5
    slippage_abs: float = 0.05
    max_quote_gap_minutes: int = 10


@dataclass
class ReportConfig:
    output_dir: str = "reports"


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str | None = None
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id: str | None = None
    chat_id_env: str = "TELEGRAM_CHAT_ID"
    api_base: str = "https://api.telegram.org"
    timeout_seconds: int = 10
    disable_notification: bool = False


@dataclass
class MT5Config:
    terminal_path: str | None = None
    live_trading_enabled: bool = False
    poll_seconds: int = 5
    history_bars: int = 600


@dataclass
class IBKRContractConfig:
    symbol: str = ""
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    lastTradeDateOrContractMonth: str = ""
    multiplier: str = ""
    primaryExchange: str = ""


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 77
    timeout_seconds: int = 30
    gateway_path: str | None = "C:\\Jts\\ibgateway\\1046\\ibgateway.exe"
    duration: str = "2 D"
    bar_size: str = "5 mins"
    what_to_show: str = "TRADES"
    use_rth: bool = False
    market_contract: IBKRContractConfig = field(
        default_factory=lambda: IBKRContractConfig(
            symbol="XAUUSD",
            secType="CFD",
            exchange="SMART",
            currency="USD",
        )
    )
    dxy_contract: IBKRContractConfig = field(
        default_factory=lambda: IBKRContractConfig(
            symbol="UUP",
            secType="STK",
            exchange="SMART",
            currency="USD",
            primaryExchange="ARCA",
        )
    )
    yield_contract: IBKRContractConfig = field(
        default_factory=lambda: IBKRContractConfig(
            symbol="IEF",
            secType="STK",
            exchange="SMART",
            currency="USD",
            primaryExchange="NASDAQ",
        )
    )
    yield_transform: str = "inverse_price"


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    cross_assets: CrossAssetConfig = field(default_factory=CrossAssetConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name in values:
            current = values[item.name]
        elif item.default is not MISSING:
            current = item.default
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            current = item.default_factory()  # type: ignore[misc]
        else:
            current = None
        kwargs[item.name] = current
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any] | None) -> SystemConfig:
    raw = raw or {}
    strategies_raw = raw.get("strategies", {}) or {}
    strategy_config = StrategyConfig(
        breakout=_dataclass_from_dict(BreakoutConfig, strategies_raw.get("breakout")),
        mean_reversion=_dataclass_from_dict(
            MeanReversionConfig, strategies_raw.get("mean_reversion")
        ),
        micro_scalp=_dataclass_from_dict(
            MicroScalpConfig, strategies_raw.get("micro_scalp")
        ),
    )
    ibkr_raw = raw.get("ibkr", {}) or {}
    ibkr_config = _dataclass_from_dict(IBKRConfig, ibkr_raw)
    if isinstance(ibkr_raw.get("dxy_contract"), dict):
        ibkr_config.dxy_contract = _dataclass_from_dict(
            IBKRContractConfig, ibkr_raw.get("dxy_contract")
        )
    if isinstance(ibkr_raw.get("yield_contract"), dict):
        ibkr_config.yield_contract = _dataclass_from_dict(
            IBKRContractConfig, ibkr_raw.get("yield_contract")
        )
    if isinstance(ibkr_raw.get("market_contract"), dict):
        ibkr_config.market_contract = _dataclass_from_dict(
            IBKRContractConfig, ibkr_raw.get("market_contract")
        )
    return SystemConfig(
        data=_dataclass_from_dict(DataConfig, raw.get("data")),
        calendar=_dataclass_from_dict(CalendarConfig, raw.get("calendar")),
        cross_assets=_dataclass_from_dict(CrossAssetConfig, raw.get("cross_assets")),
        indicators=_dataclass_from_dict(IndicatorConfig, raw.get("indicators")),
        strategies=strategy_config,
        filters=_dataclass_from_dict(FilterConfig, raw.get("filters")),
        risk=_dataclass_from_dict(RiskConfig, raw.get("risk")),
        report=_dataclass_from_dict(ReportConfig, raw.get("report")),
        telegram=_dataclass_from_dict(TelegramConfig, raw.get("telegram")),
        mt5=_dataclass_from_dict(MT5Config, raw.get("mt5")),
        ibkr=ibkr_config,
    )


def load_config(path: str | Path) -> SystemConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)
