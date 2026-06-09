from __future__ import annotations

from datetime import datetime, time

import pandas as pd


def parse_hhmm(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


def time_in_range(moment: datetime | pd.Timestamp, start: str, end: str) -> bool:
    current = moment.timetz().replace(tzinfo=None) if hasattr(moment, "timetz") else moment.time()
    start_time = parse_hhmm(start)
    end_time = parse_hhmm(end)
    if start_time <= end_time:
        return start_time <= current < end_time
    return current >= start_time or current < end_time


def time_in_any_range(moment: datetime | pd.Timestamp, windows: list[list[str]]) -> bool:
    return any(time_in_range(moment, start, end) for start, end in windows)


def session_label(moment: datetime | pd.Timestamp) -> str:
    if time_in_range(moment, "21:00", "01:00"):
        return "overlap"
    if time_in_range(moment, "01:00", "06:00"):
        return "ny_tail"
    if time_in_range(moment, "06:00", "16:00"):
        return "asia"
    if time_in_range(moment, "16:00", "21:00"):
        return "london"
    return "off"
