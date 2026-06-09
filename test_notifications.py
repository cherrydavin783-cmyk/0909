from __future__ import annotations

import pandas as pd
import pytest

from gold_scalper.config import config_from_dict
from gold_scalper.models import OrderIntent, Side, TradeRecord
from gold_scalper.notifications import (
    NotificationError,
    TelegramNotifier,
    format_order_message,
    format_trade_exit_message,
)


def _order() -> OrderIntent:
    return OrderIntent(
        timestamp=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        symbol="XAUUSD",
        side=Side.LONG,
        volume=0.42,
        entry_price=2500.0,
        stop_loss=2490.0,
        take_profit=2506.25,
        strategy="micro_scalp",
        risk_amount=40.0,
        reason="test setup",
    )


def test_format_order_message_contains_trade_fields() -> None:
    text = format_order_message(_order(), mode="scalp")

    assert "PAPER SIGNAL" in text
    assert "XAUUSD" in text
    assert "micro_scalp / mode=scalp" in text
    assert "Side: LONG" in text
    assert "Volume: 0.42" in text
    assert "Target: 2506.25" in text
    assert "no live order sent" in text


def test_format_trade_exit_message_contains_exit_fields() -> None:
    trade = TradeRecord(
        trade_id="abc123",
        symbol="XAUUSD",
        side=Side.LONG,
        strategy="micro_scalp",
        volume=0.42,
        entry_time=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        entry_price=2500.0,
        exit_time=pd.Timestamp("2026-01-05 20:58", tz="Asia/Shanghai").to_pydatetime(),
        exit_price=2490.0,
        pnl=-421.47,
        exit_reason="stop_loss",
    )

    text = format_trade_exit_message(trade)

    assert "PAPER EXIT" in text
    assert "XAUUSD" in text
    assert "Side: LONG" in text
    assert "Exit: 2490.00" in text
    assert "PnL: -421.47" in text
    assert "Exit reason: stop_loss" in text


def test_telegram_notifier_disabled_returns_none() -> None:
    notifier = TelegramNotifier(config_from_dict({"telegram": {"enabled": False}}))

    assert notifier.status().enabled is False
    assert notifier.send_order(_order()) is None


def test_telegram_notifier_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    notifier = TelegramNotifier(config_from_dict({"telegram": {"enabled": True}}))

    assert notifier.status().ready is False
    with pytest.raises(NotificationError):
        notifier.send_order(_order())
