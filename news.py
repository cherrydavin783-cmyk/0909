from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


@dataclass
class NewsEvent:
    time: datetime
    currency: str
    impact: str
    event: str


class NewsCalendar:
    def __init__(
        self,
        events: list[NewsEvent] | None = None,
        available: bool = True,
        source: str | None = None,
    ) -> None:
        self.events = events or []
        self.available = available
        self.source = source

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        timezone: str,
        currencies: list[str],
        impacts: list[str],
    ) -> "NewsCalendar":
        source = str(path)
        csv_path = Path(path)
        if not csv_path.exists():
            return cls(events=[], available=False, source=source)
        frame = pd.read_csv(csv_path)
        required = {"time", "currency", "impact", "event"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"News CSV missing columns: {sorted(missing)}")
        parsed = pd.to_datetime(frame["time"])
        if getattr(parsed.dt, "tz", None) is None:
            parsed = parsed.dt.tz_localize(timezone)
        else:
            parsed = parsed.dt.tz_convert(timezone)

        currency_set = {item.upper() for item in currencies}
        impact_set = {item.lower() for item in impacts}
        events: list[NewsEvent] = []
        for idx, row in frame.iterrows():
            currency = str(row["currency"]).upper()
            impact = str(row["impact"]).lower()
            if currency not in currency_set or impact not in impact_set:
                continue
            events.append(
                NewsEvent(
                    time=parsed.iloc[idx].to_pydatetime(),
                    currency=currency,
                    impact=impact,
                    event=str(row["event"]),
                )
            )
        return cls(events=events, available=True, source=source)

    def in_blackout(self, timestamp: datetime, before_minutes: int, after_minutes: int) -> bool:
        before = timedelta(minutes=before_minutes)
        after = timedelta(minutes=after_minutes)
        for event in self.events:
            if event.time - before <= timestamp <= event.time + after:
                return True
        return False
