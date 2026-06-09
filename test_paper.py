from __future__ import annotations

import pandas as pd

from gold_scalper.config import config_from_dict
from gold_scalper.models import OrderIntent, Side
from gold_scalper.paper import (
    PaperTradeLog,
    evaluate_position_exit,
    position_from_order,
    recover_latest_open_position,
)


def _bars() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai"),
            pd.Timestamp("2026-01-05 20:56", tz="Asia/Shanghai"),
            pd.Timestamp("2026-01-05 20:57", tz="Asia/Shanghai"),
        ]
    )
    return pd.DataFrame(
        {
            "open": [2500.0, 2500.2, 2498.0],
            "high": [2500.4, 2500.5, 2498.4],
            "low": [2499.7, 2498.8, 2489.9],
            "close": [2500.1, 2499.1, 2490.5],
            "spread": [0.3, 0.3, 0.3],
            "volume": [1.0, 1.0, 1.0],
        },
        index=index,
    )


def _order() -> OrderIntent:
    return OrderIntent(
        timestamp=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        symbol="XAUUSD",
        side=Side.LONG,
        volume=0.4,
        entry_price=2500.0,
        stop_loss=2490.0,
        take_profit=2510.0,
        strategy="micro_scalp",
        risk_amount=400.0,
    )


def test_evaluate_position_exit_detects_stop_loss_after_entry_bar() -> None:
    config = config_from_dict({"risk": {"slippage_abs": 0.05}})
    position = position_from_order(_order(), max_holding_minutes=8)

    trade = evaluate_position_exit(position, _bars(), config)

    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_time == pd.Timestamp("2026-01-05 20:57", tz="Asia/Shanghai")
    assert trade.exit_price == 2489.95


def test_recover_latest_open_position_skips_closed_trades(tmp_path) -> None:
    config = config_from_dict({"strategies": {"micro_scalp": {"max_holding_minutes": 8}}})
    signal_path = tmp_path / "paper_signals.csv"
    signal_path.write_text(
        "time,symbol,strategy,side,volume,entry,stop_loss,take_profit,reason\n"
        "2026-01-05T20:55:00+08:00,XAUUSD,micro_scalp,long,0.4,2500.0,2490.0,2510.0,test\n",
        encoding="utf-8",
    )
    trade_log = PaperTradeLog(tmp_path / "paper_trades.csv")

    position = recover_latest_open_position(signal_path, trade_log, config)

    assert position is not None
    assert position.strategy == "micro_scalp"
    assert position.max_holding_minutes == 8

    trade = evaluate_position_exit(position, _bars(), config)
    assert trade is not None
    trade_log.append(trade)

    assert recover_latest_open_position(signal_path, trade_log, config) is None
