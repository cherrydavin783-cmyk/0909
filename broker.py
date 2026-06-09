from __future__ import annotations

import csv
from pathlib import Path

from .models import OrderIntent


class LiveTradingDisabled(RuntimeError):
    pass


class BrokerGateway:
    def send_order(self, order: OrderIntent) -> None:
        del order
        raise LiveTradingDisabled("Live order submission is disabled in this version.")


class PaperSignalLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "time,symbol,strategy,side,volume,entry,stop_loss,take_profit,reason\n",
                encoding="utf-8",
            )

    def append(self, order: OrderIntent) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    order.timestamp.isoformat(),
                    order.symbol,
                    order.strategy,
                    order.side.value,
                    order.volume,
                    order.entry_price,
                    order.stop_loss,
                    order.take_profit,
                    order.reason,
                ]
            )
