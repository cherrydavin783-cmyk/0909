from __future__ import annotations

import csv
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .config import SystemConfig
from .news import NewsEvent


class CalendarSyncError(RuntimeError):
    pass


@dataclass
class CalendarSyncResult:
    provider: str
    output_csv: Path
    events: list[NewsEvent]
    source: str
    used_cache: bool = False


def _request_text(url: str, timeout: int = 30) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "gold-scalper/0.1 (+https://local)",
            "Accept": "application/xml,text/xml,application/json,*/*",
        },
    )
    last_error: Exception | None = None
    for _ in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            break
        except Exception as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    for encoding in ("utf-8", "windows-1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _html_lines(raw: str) -> list[str]:
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|li|tr|td|th|h[1-6])>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def _is_cache_fresh(path: Path, refresh_hours: int) -> bool:
    if not path.exists() or refresh_hours <= 0:
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age <= timedelta(hours=refresh_hours)


def _parse_forex_factory_datetime(
    date_value: str,
    time_value: str,
    source_timezone: str,
    target_timezone: str,
) -> datetime | None:
    clean_time = (time_value or "").strip().lower()
    if not clean_time or clean_time in {"all day", "tentative"} or clean_time.startswith("day "):
        return None
    try:
        naive = datetime.strptime(f"{date_value.strip()} {clean_time}", "%m-%d-%Y %I:%M%p")
    except ValueError:
        return None
    return naive.replace(tzinfo=ZoneInfo(source_timezone)).astimezone(ZoneInfo(target_timezone))


def _filter_events(
    events: list[NewsEvent],
    config: SystemConfig,
    start: datetime | None,
    end: datetime | None,
) -> list[NewsEvent]:
    currencies = {item.upper() for item in config.filters.news_currencies}
    impacts = {item.lower() for item in config.filters.news_impacts}
    filtered: list[NewsEvent] = []
    seen: set[tuple[datetime, str, str]] = set()
    for event in events:
        if event.currency.upper() not in currencies:
            continue
        if event.impact.lower() not in impacts:
            continue
        if start and event.time < start:
            continue
        if end and event.time > end:
            continue
        key = (event.time, event.currency.upper(), event.event)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(event)
    return sorted(filtered, key=lambda item: item.time)


def _write_news_csv(events: list[NewsEvent], output_csv: str | Path) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "currency", "impact", "event"])
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "time": event.time.isoformat(),
                    "currency": event.currency.upper(),
                    "impact": event.impact.lower(),
                    "event": event.event,
                }
            )
    return output_path


def fetch_forex_factory_events(
    config: SystemConfig,
    force: bool = False,
) -> tuple[list[NewsEvent], bool]:
    cache_path = Path(config.calendar.cache_file)
    used_cache = False
    if not force and _is_cache_fresh(cache_path, config.calendar.refresh_hours):
        raw = cache_path.read_text(encoding="utf-8")
        used_cache = True
    else:
        try:
            raw = _request_text(config.calendar.forex_factory_url)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(raw, encoding="utf-8")
        except Exception as exc:
            if cache_path.exists():
                raw = cache_path.read_text(encoding="utf-8")
                used_cache = True
            else:
                raise CalendarSyncError(f"ForexFactory calendar fetch failed: {exc}") from exc

    root = ET.fromstring(raw)
    events: list[NewsEvent] = []
    for item in root.findall("event"):
        currency = (item.findtext("country") or "").strip().upper()
        impact = (item.findtext("impact") or "").strip().lower()
        title = (item.findtext("title") or "").strip()
        date_value = item.findtext("date") or ""
        time_value = item.findtext("time") or ""
        event_time = _parse_forex_factory_datetime(
            date_value,
            time_value,
            config.calendar.source_timezone,
            config.calendar.target_timezone,
        )
        if event_time is None:
            continue
        events.append(NewsEvent(event_time, currency, impact, title))
    return events, used_cache


def fetch_trading_economics_events(
    config: SystemConfig,
    start: datetime,
    end: datetime,
) -> list[NewsEvent]:
    client = config.calendar.trading_economics_client or os.getenv(
        config.calendar.trading_economics_client_env, ""
    )
    if not client:
        raise CalendarSyncError(
            "Trading Economics requires an API client. Set TRADING_ECONOMICS_CLIENT."
        )
    country = quote(config.calendar.trading_economics_country)
    start_day = start.date().isoformat()
    end_day = end.date().isoformat()
    url = (
        "https://api.tradingeconomics.com/calendar/country/"
        f"{country}/{start_day}/{end_day}?c={quote(client)}"
        f"&importance={config.calendar.trading_economics_importance}&f=json"
    )
    raw = _request_text(url)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CalendarSyncError(f"Trading Economics returned non-JSON data: {raw[:200]}") from exc
    if isinstance(payload, dict) and "Message" in payload:
        raise CalendarSyncError(str(payload["Message"]))
    if not isinstance(payload, list):
        raise CalendarSyncError("Trading Economics returned an unexpected payload.")

    events: list[NewsEvent] = []
    source_tz = ZoneInfo("UTC")
    target_tz = ZoneInfo(config.calendar.target_timezone)
    for item in payload:
        date_value = str(item.get("Date", "")).strip()
        if not date_value:
            continue
        try:
            event_time = datetime.fromisoformat(date_value).replace(tzinfo=source_tz)
        except ValueError:
            continue
        events.append(
            NewsEvent(
                time=event_time.astimezone(target_tz),
                currency="USD",
                impact="high" if int(item.get("Importance", 0) or 0) >= 3 else "medium",
                event=str(item.get("Event") or item.get("Category") or "Economic event"),
            )
        )
    return events


def fetch_fxmacrodata_events(config: SystemConfig) -> list[NewsEvent]:
    raw = _request_text(config.calendar.fxmacrodata_url)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CalendarSyncError(f"FXMacroData returned non-JSON data: {raw[:200]}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise CalendarSyncError("FXMacroData returned an unexpected payload.")

    selected = {item.lower() for item in config.calendar.fxmacrodata_high_impact_releases}
    target_tz = ZoneInfo(config.calendar.target_timezone)
    currency = str(payload.get("currency", "USD")).upper()
    events: list[NewsEvent] = []
    for item in data:
        release = str(item.get("release", "")).lower()
        if release not in selected:
            continue
        timestamp = item.get("announcement_datetime")
        if timestamp is None:
            continue
        try:
            event_time = datetime.fromtimestamp(float(timestamp), tz=ZoneInfo("UTC"))
        except (TypeError, ValueError, OSError):
            continue
        name = str(item.get("name") or release.replace("_", " ").title())
        events.append(
            NewsEvent(
                time=event_time.astimezone(target_tz),
                currency=currency,
                impact="high",
                event=name,
            )
        )
    return events


def fetch_bls_schedule_events(
    config: SystemConfig,
    start: datetime,
    end: datetime,
) -> list[NewsEvent]:
    eastern = ZoneInfo("America/New_York")
    target_tz = ZoneInfo(config.calendar.target_timezone)
    date_pattern = re.compile(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), "
        r"[A-Za-z]+ \d{1,2}, \d{4}$"
    )
    time_pattern = re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)$", re.IGNORECASE)
    selected = {
        "consumer price index": "Consumer Price Index",
        "employment situation": "Employment Situation",
    }
    events: list[NewsEvent] = []
    seen: set[tuple[datetime, str]] = set()

    for year in range(start.year, end.year + 1):
        url = config.calendar.bls_schedule_url.format(year=year)
        raw = _request_text(url, timeout=60)
        lines = _html_lines(raw)
        current_date: str | None = None
        for idx, line in enumerate(lines):
            if date_pattern.match(line):
                current_date = line
                continue
            if current_date is None or not time_pattern.match(line):
                continue
            if idx + 1 >= len(lines):
                continue
            title = lines[idx + 1]
            event_name = next(
                (
                    canonical
                    for prefix, canonical in selected.items()
                    if title.lower().startswith(f"{prefix} for")
                ),
                None,
            )
            if event_name is None:
                continue
            try:
                naive = datetime.strptime(
                    f"{current_date} {line.upper()}",
                    "%A, %B %d, %Y %I:%M %p",
                )
            except ValueError:
                continue
            event_time = naive.replace(tzinfo=eastern).astimezone(target_tz)
            if not (start <= event_time <= end):
                continue
            key = (event_time, event_name)
            if key in seen:
                continue
            seen.add(key)
            events.append(NewsEvent(event_time, "USD", "high", event_name))
    return sorted(events, key=lambda item: item.time)


def _fred_release_events(
    config: SystemConfig,
    release_id: int,
    event_name: str,
    start: datetime,
    end: datetime,
) -> list[NewsEvent]:
    central = ZoneInfo("America/Chicago")
    target_tz = ZoneInfo(config.calendar.target_timezone)
    events: list[NewsEvent] = []
    date_pattern = re.compile(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) "
        r"([A-Za-z]+ \d{1,2}, \d{4}) Updated$"
    )
    time_pattern = re.compile(r"^(\d{1,2}:\d{2})\s*(am|pm)(?:\s+(.+))?$", re.IGNORECASE)
    for year in range(start.year, end.year + 1):
        url = (
            f"{config.calendar.fred_calendar_url}?ob=n&od=asc&rid={release_id}"
            f"&ve={year}-12-31&view=year&vs={year}-01-01"
        )
        raw = _request_text(url, timeout=60)
        lines = _html_lines(raw)
        for idx, line in enumerate(lines):
            date_match = date_pattern.match(line)
            if not date_match:
                continue
            release_date = date_match.group(2)
            for offset, lookahead in enumerate(lines[idx + 1 : idx + 5], start=1):
                time_match = time_pattern.match(lookahead)
                if not time_match:
                    continue
                title = time_match.group(3) or (
                    lines[idx + offset + 1] if idx + offset + 1 < len(lines) else ""
                )
                if event_name.lower() not in title.lower():
                    continue
                naive = datetime.strptime(
                    f"{release_date} {time_match.group(1)} {time_match.group(2).upper()}",
                    "%B %d, %Y %I:%M %p",
                )
                event_time = naive.replace(tzinfo=central).astimezone(target_tz)
                if start <= event_time <= end:
                    events.append(NewsEvent(event_time, "USD", "high", event_name))
                break
    return events


def _fomc_events(config: SystemConfig, start: datetime, end: datetime) -> list[NewsEvent]:
    raw = _request_text(config.calendar.fomc_calendar_url, timeout=60)
    lines = _html_lines(raw)
    eastern = ZoneInfo("America/New_York")
    target_tz = ZoneInfo(config.calendar.target_timezone)
    month_names = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    events: list[NewsEvent] = []
    year: int | None = None
    month_label: str | None = None
    for idx, line in enumerate(lines):
        year_match = re.match(r"^(\d{4}) FOMC Meetings$", line)
        if year_match:
            year = int(year_match.group(1))
            month_label = None
            continue
        if year is None:
            continue
        if line in month_names or line == "Apr/May":
            month_label = line
            continue
        if month_label is None:
            continue
        if "notation vote" in line.lower():
            continue
        day_match = re.match(r"^(\d{1,2})(?:-(\d{1,2}))?\*?$", line)
        if not day_match:
            continue
        if not any("Statement:" in item for item in lines[idx + 1 : idx + 8]):
            continue
        start_day = int(day_match.group(1))
        end_day = int(day_match.group(2) or day_match.group(1))
        if month_label == "Apr/May" and end_day < start_day:
            month = month_names["May"]
        elif month_label == "Apr/May":
            month = month_names["April"]
        else:
            month = month_names[month_label]
        statement_time = datetime(year, month, end_day, 14, 0, tzinfo=eastern).astimezone(
            target_tz
        )
        if start <= statement_time <= end:
            events.append(NewsEvent(statement_time, "USD", "high", "FOMC Statement"))
        for lookahead in lines[idx + 1 : idx + 20]:
            minutes_match = re.search(r"\(Released ([A-Za-z]+ \d{2}, \d{4})\)", lookahead)
            if not minutes_match:
                continue
            minutes_date = datetime.strptime(minutes_match.group(1), "%B %d, %Y")
            minutes_time = minutes_date.replace(hour=14, minute=0, tzinfo=eastern).astimezone(
                target_tz
            )
            if start <= minutes_time <= end:
                events.append(NewsEvent(minutes_time, "USD", "high", "FOMC Minutes"))
            break
    return events


def fetch_fred_us_macro_events(
    config: SystemConfig,
    start: datetime,
    end: datetime,
) -> list[NewsEvent]:
    events = []
    try:
        bls_events = fetch_bls_schedule_events(config, start, end)
    except Exception:
        bls_events = []
    if bls_events:
        events.extend(bls_events)
    else:
        events.extend(_fred_release_events(config, 10, "Consumer Price Index", start, end))
        events.extend(_fred_release_events(config, 50, "Employment Situation", start, end))
    events.extend(_fomc_events(config, start, end))
    return sorted(events, key=lambda item: item.time)


def sync_calendar(
    config: SystemConfig,
    start: datetime | None = None,
    end: datetime | None = None,
    provider: str | None = None,
    force: bool = False,
) -> CalendarSyncResult:
    target_tz = ZoneInfo(config.calendar.target_timezone)
    now = datetime.now(target_tz)
    start = start or (now - timedelta(days=config.calendar.lookback_days))
    end = end or (now + timedelta(days=config.calendar.lookahead_days))
    provider_name = (provider or config.calendar.provider).lower().replace("-", "_")

    if provider_name == "fred_us_macro":
        events = fetch_fred_us_macro_events(config, start, end)
        used_cache = False
        source = "BLS release schedule + Federal Reserve FOMC calendar"
    elif provider_name == "fxmacrodata":
        events = fetch_fxmacrodata_events(config)
        used_cache = False
        source = config.calendar.fxmacrodata_url
    elif provider_name == "forex_factory":
        events, used_cache = fetch_forex_factory_events(config, force=force)
        source = config.calendar.forex_factory_url
    elif provider_name == "trading_economics":
        events = fetch_trading_economics_events(config, start, end)
        used_cache = False
        source = "https://api.tradingeconomics.com/calendar"
    else:
        raise CalendarSyncError(f"Unsupported calendar provider: {provider_name}")

    filtered = _filter_events(events, config, start, end)
    output = _write_news_csv(filtered, config.calendar.output_csv)
    return CalendarSyncResult(provider_name, output, filtered, source, used_cache)
