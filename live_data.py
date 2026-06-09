from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SystemConfig
from .ibkr_adapter import IBKRUnavailable, _IBKRHistoricalApp, _normalize_bars, _to_contract
from .mt5_adapter import MT5MarketDataSource


class CSVLiveMarketDataSource:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def initialize(self) -> None:
        if not self.config.data.live_csv_path:
            raise ValueError("data.live_csv_path is required when data.live_source=csv")
        path = Path(self.config.data.live_csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Live CSV not found: {path}")

    def shutdown(self) -> None:
        return None

    def fetch_latest_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        del symbol, timeframe
        path = Path(self.config.data.live_csv_path)
        frame = pd.read_csv(path)
        time_column = self.config.data.live_csv_time_column
        if time_column not in frame.columns and "datetime" in frame.columns:
            time_column = "datetime"
        if time_column not in frame.columns:
            raise ValueError(f"Live CSV missing time column: {time_column}")

        rename = {time_column: "time"}
        frame = frame.rename(columns=rename)
        required = ["time", "open", "high", "low", "close"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Live CSV missing columns: {', '.join(missing)}")
        if "volume" not in frame.columns:
            frame["volume"] = 1.0
        if "spread" not in frame.columns:
            frame["spread"] = self.config.data.default_spread

        keep = ["time", "open", "high", "low", "close", "volume", "spread"]
        data = frame[keep].copy()
        data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_convert(
            self.config.data.timezone
        )
        for column in ["open", "high", "low", "close", "volume", "spread"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data["volume"] = data["volume"].where(data["volume"] > 0, 1.0)
        data["spread"] = data["spread"].fillna(self.config.data.default_spread)
        data = data.dropna(subset=["time", "open", "high", "low", "close"])
        data = data.drop_duplicates(subset=["time"]).sort_values("time")
        return data.tail(max(1, int(count))).reset_index(drop=True)


class IBKRLiveMarketDataSource:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def initialize(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def fetch_latest_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        del symbol, timeframe
        cfg = self.config.ibkr
        app = _IBKRHistoricalApp()
        try:
            app.connect(cfg.host, cfg.port, cfg.client_id, cfg.timeout_seconds)
            bars = app.historical_bars(
                7101,
                _to_contract(cfg.market_contract),
                cfg.duration,
                cfg.bar_size,
                cfg.what_to_show,
                cfg.use_rth,
                cfg.timeout_seconds,
            )
        finally:
            app.disconnect()
        if bars.empty:
            raise IBKRUnavailable("IBKR returned empty market history.")
        data = _normalize_bars(bars, self.config.data.timezone)
        data["spread"] = self.config.data.default_spread
        data["volume"] = pd.to_numeric(data.get("volume", 1.0), errors="coerce").fillna(1.0)
        data["volume"] = data["volume"].where(data["volume"] > 0, 1.0)
        keep = ["time", "open", "high", "low", "close", "volume", "spread"]
        data = data[keep].dropna(subset=["time", "open", "high", "low", "close"])
        data = data.drop_duplicates(subset=["time"]).sort_values("time")
        return data.tail(max(1, int(count))).reset_index(drop=True)


def make_live_market_data_source(config: SystemConfig):
    source = config.data.live_source.strip().lower()
    if source == "mt5":
        return MT5MarketDataSource(config)
    if source in {"ib", "ibkr", "ib_gateway"}:
        return IBKRLiveMarketDataSource(config)
    if source in {"csv", "csv_live", "file"}:
        return CSVLiveMarketDataSource(config)
    raise ValueError(f"Unsupported live data source: {config.data.live_source}")


def latest_bar_age_minutes(data: pd.DataFrame, timezone: str) -> float | None:
    if data.empty or "time" not in data.columns:
        return None
    latest = pd.Timestamp(data["time"].iloc[-1])
    if latest.tzinfo is None:
        latest = latest.tz_localize(timezone)
    now = pd.Timestamp.now(tz=timezone)
    return max(0.0, float((now - latest).total_seconds() / 60))
