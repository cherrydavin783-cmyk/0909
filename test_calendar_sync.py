from __future__ import annotations

from pathlib import Path

import pandas as pd

from gold_scalper.calendar_sync import (
    fetch_bls_schedule_events,
    fetch_forex_factory_events,
    fetch_fred_us_macro_events,
    fetch_fxmacrodata_events,
    sync_calendar,
)
from gold_scalper.config import config_from_dict


SAMPLE_XML = """<?xml version="1.0" encoding="windows-1252"?>
<weeklyevents>
  <event>
    <title>FOMC Meeting Minutes</title>
    <country>USD</country>
    <date><![CDATA[05-20-2026]]></date>
    <time><![CDATA[6:00pm]]></time>
    <impact><![CDATA[High]]></impact>
  </event>
  <event>
    <title>Low Impact Event</title>
    <country>USD</country>
    <date><![CDATA[05-20-2026]]></date>
    <time><![CDATA[1:00pm]]></time>
    <impact><![CDATA[Low]]></impact>
  </event>
</weeklyevents>
"""


def test_forex_factory_cache_parse_and_filter(tmp_path) -> None:
    cache = tmp_path / "calendar_cache.xml"
    output = tmp_path / "news.csv"
    cache.write_text(SAMPLE_XML, encoding="utf-8")
    config = config_from_dict(
        {
            "calendar": {
                "cache_file": str(cache),
                "output_csv": str(output),
                "refresh_hours": 24,
            }
        }
    )
    events, used_cache = fetch_forex_factory_events(config)
    assert used_cache
    assert len(events) == 2
    assert events[0].time.isoformat() == "2026-05-21T02:00:00+08:00"

    result = sync_calendar(
        config,
        start=pd.Timestamp("2026-05-20", tz="Asia/Shanghai").to_pydatetime(),
        end=pd.Timestamp("2026-05-22", tz="Asia/Shanghai").to_pydatetime(),
        provider="forex_factory",
    )
    assert result.used_cache
    assert len(result.events) == 1
    assert output.exists()
    rows = output.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "time,currency,impact,event"
    assert "FOMC Meeting Minutes" in rows[1]


def test_fxmacrodata_parse(monkeypatch) -> None:
    payload = """{
      "currency": "USD",
      "data": [
        {"release": "policy_rate", "announcement_datetime": 1772272800, "name": "Fed Funds Rate"},
        {"release": "building_permits", "announcement_datetime": 1772276400, "name": "Building Permits"}
      ]
    }"""

    monkeypatch.setattr("gold_scalper.calendar_sync._request_text", lambda url: payload)
    config = config_from_dict({})
    events = fetch_fxmacrodata_events(config)
    assert len(events) == 1
    assert events[0].currency == "USD"
    assert events[0].impact == "high"
    assert events[0].event == "Fed Funds Rate"


def test_bls_schedule_parse(monkeypatch) -> None:
    payload = """
    <html><body>
    <p>Friday, January 05, 2024</p><p>08:30 AM</p>
    <p>Employment Situation for December 2023</p>
    <p>Thursday, January 11, 2024</p><p>08:30 AM</p>
    <p>Consumer Price Index for December 2023</p>
    <p>Thursday, January 11, 2024</p><p>08:30 AM</p>
    <p>Real Earnings for December 2023</p>
    </body></html>
    """

    monkeypatch.setattr("gold_scalper.calendar_sync._request_text", lambda url, timeout=30: payload)
    config = config_from_dict({})
    events = fetch_bls_schedule_events(
        config,
        start=pd.Timestamp("2024-01-01", tz="Asia/Shanghai").to_pydatetime(),
        end=pd.Timestamp("2024-01-31", tz="Asia/Shanghai").to_pydatetime(),
    )
    assert [event.event for event in events] == ["Employment Situation", "Consumer Price Index"]
    assert events[0].time.isoformat() == "2024-01-05T21:30:00+08:00"


def test_fred_us_macro_parse(monkeypatch) -> None:
    fred_cpi = """
    <html><body>
    <p>Wednesday January 15, 2025 Updated</p>
    <p>7:30 am <a>Consumer Price Index</a></p>
    </body></html>
    """
    fred_jobs = """
    <html><body>
    <p>Friday February 07, 2025 Updated</p>
    <p>7:30 am <a>Employment Situation</a></p>
    </body></html>
    """
    fed_fomc = """
    <html><body>
    <h4>2025 FOMC Meetings</h4>
    <p>January</p><p>28-29</p><p>Statement:</p>
    <p>Minutes:</p><p>(Released February 19, 2025)</p>
    </body></html>
    """

    def fake_request(url, timeout=30):
        if "rid=10" in url:
            return fred_cpi
        if "rid=50" in url:
            return fred_jobs
        return fed_fomc

    monkeypatch.setattr("gold_scalper.calendar_sync._request_text", fake_request)
    config = config_from_dict({})
    events = fetch_fred_us_macro_events(
        config,
        start=pd.Timestamp("2025-01-01", tz="Asia/Shanghai").to_pydatetime(),
        end=pd.Timestamp("2025-03-01", tz="Asia/Shanghai").to_pydatetime(),
    )
    names = [event.event for event in events]
    assert names == [
        "Consumer Price Index",
        "FOMC Statement",
        "Employment Situation",
        "FOMC Minutes",
    ]
    assert events[0].time.isoformat() == "2025-01-15T21:30:00+08:00"
