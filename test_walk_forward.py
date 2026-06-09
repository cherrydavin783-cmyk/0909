from __future__ import annotations

from gold_scalper.walk_forward import yearly_periods


def test_yearly_periods_split_partial_years() -> None:
    periods = yearly_periods("2021-05-19", "2023-02-10", "Asia/Shanghai")
    assert [item[0] for item in periods] == [2021, 2022, 2023]
    assert periods[0][1].isoformat() == "2021-05-19T00:00:00+08:00"
    assert periods[-1][2].isoformat() == "2023-02-10T00:00:00+08:00"
