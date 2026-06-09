from __future__ import annotations

import pandas as pd

from gold_scalper.config import config_from_dict
from gold_scalper.ibkr_adapter import ibkr_port_open, parse_ibkr_bar_time, transform_yield_proxy


def test_parse_ibkr_intraday_bar_time() -> None:
    timestamp = parse_ibkr_bar_time("20260519  20:05:00", "Asia/Shanghai")
    assert timestamp.isoformat() == "2026-05-19T20:05:00+08:00"


def test_parse_ibkr_daily_bar_time() -> None:
    timestamp = parse_ibkr_bar_time("20260519", "Asia/Shanghai")
    assert timestamp.isoformat() == "2026-05-19T00:00:00+08:00"


def test_transform_yield_proxy_inverse_price() -> None:
    series = pd.Series([95.0, 96.0])
    transformed = transform_yield_proxy(series, "inverse_price")
    assert transformed.tolist() == [-95.0, -96.0]


def test_ibkr_nested_config_contract_override() -> None:
    config = config_from_dict(
        {
            "ibkr": {
                "port": 4002,
                "dxy_contract": {"symbol": "DX", "secType": "FUT", "exchange": "ICEUS"},
            }
        }
    )
    assert config.ibkr.port == 4002
    assert config.ibkr.dxy_contract.symbol == "DX"
    assert config.ibkr.dxy_contract.secType == "FUT"
    assert config.ibkr.yield_contract.symbol == "IEF"


def test_ibkr_port_open_false_for_unused_port() -> None:
    config = config_from_dict({"ibkr": {"port": 1}})
    assert not ibkr_port_open(config, timeout_seconds=0.1)
