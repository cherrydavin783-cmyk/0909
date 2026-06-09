from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import SystemConfig
from .models import OrderIntent


class MT5Unavailable(RuntimeError):
    pass


class MT5MarketDataSource:
    TIMEFRAMES = {
        "M1": "TIMEFRAME_M1",
        "M5": "TIMEFRAME_M5",
        "M15": "TIMEFRAME_M15",
        "H1": "TIMEFRAME_H1",
    }

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.mt5 = None

    def _load_mt5(self):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise MT5Unavailable(
                "MetaTrader5 package is not installed. Install with: python -m pip install -e .[mt5]"
            ) from exc
        return mt5

    def initialize(self) -> None:
        self.mt5 = self._load_mt5()
        kwargs = {}
        if self.config.mt5.terminal_path:
            kwargs["path"] = self.config.mt5.terminal_path
        if not self.mt5.initialize(**kwargs):
            raise MT5Unavailable(f"MT5 initialize failed: {self.mt5.last_error()}")

    def shutdown(self) -> None:
        if self.mt5 is not None:
            self.mt5.shutdown()

    def _timeframe(self, timeframe: str):
        if self.mt5 is None:
            raise MT5Unavailable("MT5 is not initialized")
        key = timeframe.upper()
        if key not in self.TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return getattr(self.mt5, self.TIMEFRAMES[key])

    def ensure_symbol(self, symbol: str) -> None:
        if self.mt5 is None:
            raise MT5Unavailable("MT5 is not initialized")
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise MT5Unavailable(f"MT5 symbol not found: {symbol}")
        if not info.visible and not self.mt5.symbol_select(symbol, True):
            raise MT5Unavailable(f"MT5 symbol_select failed for {symbol}: {self.mt5.last_error()}")

    def symbol_snapshot(self, symbol: str) -> dict[str, object]:
        if self.mt5 is None:
            raise MT5Unavailable("MT5 is not initialized")
        self.ensure_symbol(symbol)
        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        assert info is not None
        return {
            "name": info.name,
            "visible": info.visible,
            "trade_mode": info.trade_mode,
            "digits": info.digits,
            "point": info.point,
            "spread_points": info.spread,
            "tick_time": None if tick is None else tick.time,
            "bid": None if tick is None else tick.bid,
            "ask": None if tick is None else tick.ask,
        }

    def matching_symbols(self, patterns: tuple[str, ...] = ("*XAU*", "*GOLD*")) -> list[str]:
        if self.mt5 is None:
            raise MT5Unavailable("MT5 is not initialized")
        names: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for info in self.mt5.symbols_get(pattern) or []:
                if info.name not in seen:
                    seen.add(info.name)
                    names.append(info.name)
        return names

    def fetch_rates_range(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        if self.mt5 is None:
            raise MT5Unavailable("MT5 is not initialized")
        self.ensure_symbol(symbol)
        rates = self.mt5.copy_rates_range(symbol, self._timeframe(timeframe), start, end)
        if rates is None:
            raise MT5Unavailable(f"copy_rates_range failed: {self.mt5.last_error()}")
        return self._rates_to_frame(rates)

    def fetch_latest_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        if self.mt5 is None:
            raise MT5Unavailable("MT5 is not initialized")
        self.ensure_symbol(symbol)
        rates = self.mt5.copy_rates_from_pos(symbol, self._timeframe(timeframe), 0, count)
        if rates is None:
            raise MT5Unavailable(f"copy_rates_from_pos failed: {self.mt5.last_error()}")
        return self._rates_to_frame(rates)

    def _rates_to_frame(self, rates) -> pd.DataFrame:
        frame = pd.DataFrame(rates)
        if frame.empty:
            return frame
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_convert(
            self.config.data.timezone
        )
        if "tick_volume" in frame.columns:
            frame["volume"] = frame["tick_volume"]
        if "spread" in frame.columns:
            frame["spread"] = frame["spread"].astype(float) * self.config.data.point
        return frame[["time", "open", "high", "low", "close", "volume", "spread"]]


class MT5BrokerGateway:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def send_order(self, order: OrderIntent) -> None:
        del order
        if not self.config.mt5.live_trading_enabled:
            raise RuntimeError("Live trading is disabled by config.")
        raise RuntimeError("Live trading is intentionally not implemented in v0.1.")


def export_rates_to_csv(
    config: SystemConfig,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    output: str | Path,
    chunk_days: int = 30,
) -> Path:
    source = MT5MarketDataSource(config)
    frames: list[pd.DataFrame] = []
    try:
        source.initialize()
        cursor = start.astimezone(timezone.utc)
        final = end.astimezone(timezone.utc)
        step = timedelta(days=max(1, chunk_days))
        while cursor < final:
            chunk_end = min(cursor + step, final)
            frame = source.fetch_rates_range(symbol, timeframe, cursor, chunk_end)
            if not frame.empty:
                frames.append(frame)
            cursor = chunk_end
    finally:
        source.shutdown()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.drop_duplicates(subset=["time"]).sort_values("time")
    else:
        frame = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "spread"])
    frame.to_csv(output_path, index=False)
    return output_path
