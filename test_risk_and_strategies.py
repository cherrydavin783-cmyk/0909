from __future__ import annotations

import pandas as pd

from gold_scalper.config import config_from_dict
from gold_scalper.models import Signal, SignalType, Side
from gold_scalper.news import NewsCalendar, NewsEvent
from gold_scalper.risk import RiskManager
from gold_scalper.strategies import (
    BacktestContext,
    BreakoutStrategy,
    MeanReversionStrategy,
    MicroScalpStrategy,
)


def _row(timestamp: str, **overrides):
    data = {
        "open": 2000.0,
        "high": 2022.0,
        "low": 1998.0,
        "close": 2021.0,
        "volume": 100.0,
        "spread": 0.3,
        "spread_median": 0.3,
        "atr": 2.0,
        "rsi": 50.0,
        "ofi": 50.0,
        "ofi_ratio": 4.0,
        "vpin_quantile": 0.1,
        "volume_spike": True,
        "asia_high": 2020.0,
        "asia_low": 2000.0,
        "asia_width": 20.0,
        "fair_value": 2019.0,
    }
    data.update(overrides)
    return pd.Series(data, name=pd.Timestamp(timestamp, tz="Asia/Shanghai"))


def test_breakout_rejects_narrow_asia_range() -> None:
    config = config_from_dict({})
    signal = BreakoutStrategy(config).generate(
        _row("2026-01-05 20:55", asia_width=10.0), BacktestContext()
    )
    assert not signal.is_entry
    assert "narrow" in signal.reason


def test_breakout_requires_same_direction_ofi() -> None:
    config = config_from_dict({})
    signal = BreakoutStrategy(config).generate(
        _row("2026-01-05 20:55", ofi=-10.0), BacktestContext()
    )
    assert not signal.is_entry


def test_breakout_emits_long_signal() -> None:
    config = config_from_dict({"strategies": {"breakout": {"min_asia_width": 15.0}}})
    signal = BreakoutStrategy(config).generate(_row("2026-01-05 20:55"), BacktestContext())
    assert signal.is_entry
    assert signal.side is Side.LONG
    assert signal.stop_loss == 2019.4
    assert signal.take_profit == 2025.8


def test_mean_reversion_only_in_configured_window() -> None:
    config = config_from_dict({"strategies": {"mean_reversion": {"enabled": True}}})
    strategy = MeanReversionStrategy(config)
    assert not strategy.generate(_row("2026-01-05 12:00")).is_entry
    signal = strategy.generate(_row("2026-01-05 06:30", close=2021.0, fair_value=2019.0))
    assert signal.is_entry
    assert signal.side is Side.SHORT


def test_micro_scalp_emits_short_signal() -> None:
    config = config_from_dict({"strategies": {"micro_scalp": {"enabled": True}}})
    signal = MicroScalpStrategy(config).generate(
        _row(
            "2026-01-05 20:30",
            close=1998.0,
            micro_high=2002.0,
            micro_low=1999.0,
            rsi=42.0,
            ofi=-80.0,
            asia_width=30.0,
        ),
        BacktestContext(),
    )
    assert signal.is_entry
    assert signal.side is Side.SHORT


def test_risk_filters_spread_and_news_and_sizes_position() -> None:
    config = config_from_dict({})
    risk = RiskManager(config)
    signal = Signal(
        timestamp=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=2021.0,
        stop_loss=2019.0,
        take_profit=2024.0,
    )
    approved = risk.approve(
        signal,
        _row("2026-01-05 20:55"),
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert approved.allowed
    assert approved.order is not None
    assert approved.order.volume > 0

    wide = risk.approve(
        signal,
        _row("2026-01-05 20:55", spread=3.0),
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert not wide.allowed

    event_time = pd.Timestamp("2026-01-05 21:00", tz="Asia/Shanghai").to_pydatetime()
    news = NewsCalendar([NewsEvent(event_time, "USD", "high", "CPI")])
    blocked = risk.approve(
        signal,
        _row("2026-01-05 20:55"),
        10000.0,
        10000.0,
        None,
        news,
    )
    assert not blocked.allowed
    assert "news" in blocked.reason


def test_risk_filters_extreme_atr() -> None:
    config = config_from_dict(
        {"filters": {"max_atr_abs": 3.0, "max_atr_quantile": 0.95}}
    )
    risk = RiskManager(config)
    signal = Signal(
        timestamp=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=2021.0,
        stop_loss=2019.0,
        take_profit=2024.0,
    )
    absolute = risk.approve(
        signal,
        _row("2026-01-05 20:55", atr=3.0),
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert not absolute.allowed
    assert "atr" in absolute.reason

    quantile = risk.approve(
        signal,
        _row("2026-01-05 20:55", atr=2.0, atr_quantile=0.99),
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert not quantile.allowed
    assert "atr" in quantile.reason


def test_risk_overrides_take_profit_for_equity_target() -> None:
    config = config_from_dict(
        {
            "risk": {
                "risk_per_trade": 0.01,
                "profit_target_equity_pct": 0.003,
                "commission_per_lot": 0.0,
                "slippage_abs": 0.0,
            }
        }
    )
    risk = RiskManager(config)
    signal = Signal(
        timestamp=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=110.0,
    )
    decision = risk.approve(
        signal,
        _row("2026-01-05 20:55", spread=0.0, spread_median=0.0),
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert decision.allowed
    assert decision.order is not None
    assert decision.order.volume == 1.0
    assert round(decision.order.take_profit, 8) == 100.3


def test_strategy_profit_target_only_applies_to_named_strategy() -> None:
    config = config_from_dict(
        {
            "risk": {
                "profit_target_equity_pct": 0.0,
                "strategy_profit_target_equity_pct": {"micro_scalp": 0.0025},
                "commission_per_lot": 0.0,
                "slippage_abs": 0.0,
            }
        }
    )
    risk = RiskManager(config)
    timestamp = pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime()
    row = _row("2026-01-05 20:55", spread=0.0, spread_median=0.0)

    scalp_signal = Signal(
        timestamp=timestamp,
        strategy="micro_scalp",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=110.0,
    )
    scalp_decision = risk.approve(
        scalp_signal,
        row,
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert scalp_decision.allowed
    assert scalp_decision.order is not None
    assert round(scalp_decision.order.take_profit, 8) == 101.25

    breakout_signal = Signal(
        timestamp=timestamp,
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=110.0,
    )
    breakout_decision = risk.approve(
        breakout_signal,
        row,
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert breakout_decision.allowed
    assert breakout_decision.order is not None
    assert breakout_decision.order.take_profit == 110.0


def test_risk_overrides_take_profit_for_price_target() -> None:
    config = config_from_dict(
        {
            "risk": {
                "profit_target_price_pct": 0.0025,
                "commission_per_lot": 0.0,
                "slippage_abs": 0.0,
            }
        }
    )
    risk = RiskManager(config)
    timestamp = pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime()
    row = _row("2026-01-05 20:55", spread=0.0, spread_median=0.0)

    long_signal = Signal(
        timestamp=timestamp,
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=110.0,
    )
    long_decision = risk.approve(
        long_signal,
        row,
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert long_decision.allowed
    assert long_decision.order is not None
    assert round(long_decision.order.take_profit, 8) == 100.25

    short_signal = Signal(
        timestamp=timestamp,
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.SHORT,
        entry_price=100.0,
        stop_loss=101.0,
        take_profit=90.0,
    )
    short_decision = risk.approve(
        short_signal,
        row,
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert short_decision.allowed
    assert short_decision.order is not None
    assert round(short_decision.order.take_profit, 8) == 99.75


def test_price_target_takes_precedence_over_equity_target() -> None:
    config = config_from_dict(
        {
            "risk": {
                "risk_per_trade": 0.015,
                "profit_target_equity_pct": 0.003,
                "profit_target_price_pct": 0.0025,
                "commission_per_lot": 0.0,
                "slippage_abs": 0.0,
            }
        }
    )
    signal = Signal(
        timestamp=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=110.0,
    )
    decision = RiskManager(config).approve(
        signal,
        _row("2026-01-05 20:55", spread=0.0, spread_median=0.0),
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert decision.allowed
    assert decision.order is not None
    assert round(decision.order.take_profit, 8) == 100.25


def test_strategy_price_target_only_applies_to_named_strategy() -> None:
    config = config_from_dict(
        {
            "risk": {
                "strategy_profit_target_price_pct": {"micro_scalp": 0.0025},
                "commission_per_lot": 0.0,
                "slippage_abs": 0.0,
            }
        }
    )
    timestamp = pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime()
    row = _row("2026-01-05 20:55", spread=0.0, spread_median=0.0)
    scalp_signal = Signal(
        timestamp=timestamp,
        strategy="micro_scalp",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=110.0,
    )
    breakout_signal = Signal(
        timestamp=timestamp,
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=110.0,
    )
    risk = RiskManager(config)

    scalp_decision = risk.approve(
        scalp_signal,
        row,
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    breakout_decision = risk.approve(
        breakout_signal,
        row,
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert scalp_decision.allowed
    assert scalp_decision.order is not None
    assert round(scalp_decision.order.take_profit, 8) == 100.25
    assert breakout_decision.allowed
    assert breakout_decision.order is not None
    assert breakout_decision.order.take_profit == 110.0


def test_micro_scalp_uses_strategy_risk_size() -> None:
    config = config_from_dict(
        {
            "strategies": {"micro_scalp": {"risk_per_trade": 0.002}},
            "risk": {"risk_per_trade": 0.02},
        }
    )
    signal = Signal(
        timestamp=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        strategy="micro_scalp",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=101.0,
    )
    approved = RiskManager(config).approve(
        signal,
        _row("2026-01-05 20:55", spread=0.0, spread_median=0.0),
        10000.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert approved.allowed
    assert approved.order is not None
    assert approved.order.volume == 0.2


def test_daily_drawdown_circuit_breaker() -> None:
    config = config_from_dict({})
    risk = RiskManager(config)
    signal = Signal(
        timestamp=pd.Timestamp("2026-01-05 20:55", tz="Asia/Shanghai").to_pydatetime(),
        strategy="breakout",
        signal_type=SignalType.ENTRY,
        side=Side.LONG,
        entry_price=2021.0,
        stop_loss=2019.0,
        take_profit=2024.0,
    )
    decision = risk.approve(
        signal,
        _row("2026-01-05 20:55"),
        9600.0,
        10000.0,
        None,
        NewsCalendar([]),
    )
    assert not decision.allowed
    assert "drawdown" in decision.reason
