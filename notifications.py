from __future__ import annotations

import json
import os
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import SystemConfig
from .models import OrderIntent, TradeRecord


class NotificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelegramStatus:
    enabled: bool
    bot_token_configured: bool
    chat_id_configured: bool

    @property
    def ready(self) -> bool:
        return self.enabled and self.bot_token_configured and self.chat_id_configured


def _configured_value(value: str | None, env_name: str) -> str:
    if value:
        return value.strip()
    return os.environ.get(env_name, "").strip()


def format_order_message(order: OrderIntent, mode: str | None = None) -> str:
    side = order.side.value.upper()
    risk_distance = abs(order.entry_price - order.stop_loss)
    target_distance = abs(order.take_profit - order.entry_price)
    rr = target_distance / risk_distance if risk_distance > 0 else 0.0
    lines = [
        "PAPER SIGNAL - no live order sent",
        f"Time: {order.timestamp.isoformat()}",
        f"Symbol: {order.symbol}",
        f"Strategy: {order.strategy}" + (f" / mode={mode}" if mode else ""),
        f"Side: {side}",
        f"Volume: {order.volume:.2f}",
        f"Entry: {order.entry_price:.2f}",
        f"Stop: {order.stop_loss:.2f}",
        f"Target: {order.take_profit:.2f}",
        f"Risk amount: {order.risk_amount:.2f}",
        f"Approx R:R: {rr:.2f}",
    ]
    if order.reason:
        lines.append(f"Reason: {order.reason}")
    return "\n".join(lines)


def format_trade_exit_message(trade: TradeRecord) -> str:
    pnl_prefix = "+" if trade.pnl > 0 else ""
    lines = [
        "PAPER EXIT - no live order sent",
        f"Time: {trade.exit_time.isoformat()}",
        f"Symbol: {trade.symbol}",
        f"Strategy: {trade.strategy}",
        f"Side: {trade.side.value.upper()}",
        f"Volume: {trade.volume:.2f}",
        f"Entry: {trade.entry_price:.2f}",
        f"Exit: {trade.exit_price:.2f}",
        f"PnL: {pnl_prefix}{trade.pnl:.2f}",
        f"Exit reason: {trade.exit_reason}",
    ]
    return "\n".join(lines)


class TelegramNotifier:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config.telegram

    @property
    def bot_token(self) -> str:
        return _configured_value(self.config.bot_token, self.config.bot_token_env)

    @property
    def chat_id(self) -> str:
        return _configured_value(self.config.chat_id, self.config.chat_id_env)

    def status(self) -> TelegramStatus:
        return TelegramStatus(
            enabled=bool(self.config.enabled),
            bot_token_configured=bool(self.bot_token),
            chat_id_configured=bool(self.chat_id),
        )

    def send_text(self, text: str) -> dict[str, Any] | None:
        status = self.status()
        if not status.enabled:
            return None
        if not status.bot_token_configured or not status.chat_id_configured:
            raise NotificationError(
                "Telegram is enabled but bot token or chat id is missing. "
                f"Set {self.config.bot_token_env} and {self.config.chat_id_env}."
            )

        endpoint = (
            f"{self.config.api_base.rstrip('/')}/bot{self.bot_token}/sendMessage"
        )
        payload = urllib.parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": text,
                "disable_notification": str(bool(self.config.disable_notification)).lower(),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - configured Telegram API endpoint.
                request,
                timeout=max(1, int(self.config.timeout_seconds)),
            ) as response:
                body = response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - network failure shape varies.
            raise NotificationError(f"Telegram send failed: {exc}") from exc

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise NotificationError(f"Telegram returned non-JSON response: {body}") from exc
        if not decoded.get("ok"):
            raise NotificationError(f"Telegram rejected message: {decoded}")
        return decoded

    def send_order(self, order: OrderIntent, mode: str | None = None) -> dict[str, Any] | None:
        return self.send_text(format_order_message(order, mode=mode))

    def send_trade_exit(self, trade: TradeRecord) -> dict[str, Any] | None:
        return self.send_text(format_trade_exit_message(trade))

    def send_healthcheck(self, config_path: str) -> dict[str, Any] | None:
        host = socket.gethostname()
        return self.send_text(
            "Gold Scalper paper-check OK\n"
            f"Host: {host}\n"
            f"Config: {config_path}\n"
            "Mode: paper only, no live order submission"
        )
