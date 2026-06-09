from __future__ import annotations

import threading
import time
import socket
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .config import IBKRContractConfig, SystemConfig


class IBKRUnavailable(RuntimeError):
    pass


def _import_ibapi():
    try:
        from ibapi.client import EClient  # type: ignore
        from ibapi.contract import Contract  # type: ignore
        from ibapi.wrapper import EWrapper  # type: ignore
    except ImportError as exc:
        raise IBKRUnavailable(
            "ibapi is not installed. Install with: python -m pip install -e .[ibkr]"
        ) from exc
    return EClient, Contract, EWrapper


def _to_contract(config: IBKRContractConfig):
    _, Contract, _ = _import_ibapi()
    contract = Contract()
    for key, value in asdict(config).items():
        if value not in {"", None}:
            setattr(contract, key, value)
    return contract


class _IBKRHistoricalApp:
    def __init__(self) -> None:
        EClient, _, EWrapper = _import_ibapi()

        class App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                EClient.__init__(self, self)
                self.next_valid_id: int | None = None
                self.connected_event = threading.Event()
                self.done_events: dict[int, threading.Event] = {}
                self.bars: dict[int, list[dict[str, Any]]] = {}
                self.errors: list[tuple[int, int, str]] = []

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                self.next_valid_id = orderId
                self.connected_event.set()

            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:  # noqa: N802
                del advancedOrderRejectJson
                self.errors.append((int(reqId), int(errorCode), str(errorString)))
                if reqId in self.done_events and errorCode not in {2104, 2106, 2158}:
                    self.done_events[reqId].set()

            def historicalData(self, reqId, bar) -> None:  # noqa: N802
                self.bars.setdefault(int(reqId), []).append(
                    {
                        "time": bar.date,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    }
                )

            def historicalDataEnd(self, reqId, start, end) -> None:  # noqa: N802
                del start, end
                self.done_events.setdefault(int(reqId), threading.Event()).set()

        self.app = App()
        self.thread: threading.Thread | None = None

    def connect(self, host: str, port: int, client_id: int, timeout_seconds: int) -> None:
        try:
            self.app.connect(host, port, client_id)
        except OSError as exc:
            raise IBKRUnavailable(f"IBKR connect failed to {host}:{port}: {exc}") from exc
        self.thread = threading.Thread(target=self.app.run, daemon=True)
        self.thread.start()
        if not self.app.connected_event.wait(timeout_seconds):
            self.disconnect()
            raise IBKRUnavailable(
                f"IBKR API did not return nextValidId within {timeout_seconds}s. "
                "Open TWS/IB Gateway and enable API connections."
            )

    def disconnect(self) -> None:
        if self.app.isConnected():
            self.app.disconnect()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def historical_bars(
        self,
        req_id: int,
        contract,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
        timeout_seconds: int,
    ) -> pd.DataFrame:
        done = threading.Event()
        self.app.done_events[req_id] = done
        self.app.bars[req_id] = []
        self.app.reqHistoricalData(
            req_id,
            contract,
            "",
            duration,
            bar_size,
            what_to_show,
            1 if use_rth else 0,
            1,
            False,
            [],
        )
        if not done.wait(timeout_seconds):
            self.app.cancelHistoricalData(req_id)
            raise IBKRUnavailable(f"IBKR historical data request timed out: req_id={req_id}")
        errors = [
            error for error in self.app.errors if error[0] in {req_id, -1} and error[1] >= 300
        ]
        if errors and not self.app.bars.get(req_id):
            code, message = errors[-1][1], errors[-1][2]
            raise IBKRUnavailable(f"IBKR historical data failed ({code}): {message}")
        return pd.DataFrame(self.app.bars.get(req_id, []))


def parse_ibkr_bar_time(value: str, timezone: str) -> pd.Timestamp:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return pd.Timestamp(datetime.strptime(text, "%Y%m%d"), tz=timezone)
    if " " in text:
        left, right, *_ = text.split()
        if len(left) == 8 and left.isdigit():
            return pd.Timestamp(datetime.strptime(f"{left} {right}", "%Y%m%d %H:%M:%S"), tz=timezone)
    return pd.Timestamp(text, tz=timezone)


def _normalize_bars(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    data["time"] = [parse_ibkr_bar_time(value, timezone) for value in data["time"]]
    return data.sort_values("time")


def transform_yield_proxy(series: pd.Series, transform: str) -> pd.Series:
    if transform == "inverse_price":
        return -series
    if transform in {"none", "price"}:
        return series
    raise ValueError(f"Unsupported IBKR yield transform: {transform}")


class IBKRCrossAssetExporter:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def fetch(self) -> pd.DataFrame:
        app = _IBKRHistoricalApp()
        cfg = self.config.ibkr
        try:
            app.connect(cfg.host, cfg.port, cfg.client_id, cfg.timeout_seconds)
            dxy = app.historical_bars(
                1001,
                _to_contract(cfg.dxy_contract),
                cfg.duration,
                cfg.bar_size,
                cfg.what_to_show,
                cfg.use_rth,
                cfg.timeout_seconds,
            )
            yld = app.historical_bars(
                1002,
                _to_contract(cfg.yield_contract),
                cfg.duration,
                cfg.bar_size,
                cfg.what_to_show,
                cfg.use_rth,
                cfg.timeout_seconds,
            )
        finally:
            app.disconnect()

        dxy = _normalize_bars(dxy, self.config.data.timezone)
        yld = _normalize_bars(yld, self.config.data.timezone)
        if dxy.empty or yld.empty:
            raise IBKRUnavailable("IBKR returned empty cross-asset history.")

        dxy_series = dxy[["time", "close"]].rename(columns={"close": "dxy"})
        yld_series = yld[["time", "close"]].rename(columns={"close": "us10y"})
        yld_series["us10y"] = transform_yield_proxy(
            yld_series["us10y"], self.config.ibkr.yield_transform
        )
        merged = pd.merge_asof(
            dxy_series.sort_values("time"),
            yld_series.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=self.config.cross_assets.asof_tolerance_minutes),
        ).dropna()
        return merged

    def export_csv(self, output_path: str | Path | None = None) -> Path:
        output = Path(output_path or self.config.cross_assets.csv_path)
        frame = self.fetch()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return output


def ibkr_smoke(config: SystemConfig) -> dict[str, Any]:
    app = _IBKRHistoricalApp()
    cfg = config.ibkr
    try:
        app.connect(cfg.host, cfg.port, cfg.client_id, cfg.timeout_seconds)
        bars = app.historical_bars(
            9001,
            _to_contract(cfg.dxy_contract),
            "1 D",
            cfg.bar_size,
            cfg.what_to_show,
            cfg.use_rth,
            cfg.timeout_seconds,
        )
        return {
            "connected": True,
            "next_valid_id": app.app.next_valid_id,
            "dxy_symbol": cfg.dxy_contract.symbol,
            "bar_count": len(bars),
            "last_error": app.app.errors[-1] if app.app.errors else None,
        }
    finally:
        app.disconnect()


def ib_gateway_path_exists(config: SystemConfig) -> bool:
    return bool(config.ibkr.gateway_path and Path(config.ibkr.gateway_path).exists())


def ibkr_port_open(config: SystemConfig, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection(
            (config.ibkr.host, config.ibkr.port), timeout=timeout_seconds
        ):
            return True
    except OSError:
        return False
