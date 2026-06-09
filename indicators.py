from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SystemConfig
from .timeutils import parse_hhmm


def ensure_datetime_index(
    frame: pd.DataFrame, timezone: str, input_timezone: str
) -> pd.DataFrame:
    data = frame.copy()
    if "time" in data.columns:
        parsed = pd.to_datetime(data.pop("time"))
    else:
        parsed = pd.to_datetime(data.index)

    if isinstance(parsed, pd.Series):
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize(input_timezone)
        parsed = parsed.dt.tz_convert(timezone)
        idx = pd.DatetimeIndex(parsed)
    else:
        idx = pd.DatetimeIndex(parsed)
        if idx.tz is None:
            idx = idx.tz_localize(input_timezone)
        idx = idx.tz_convert(timezone)
    data.index = idx
    data = data.sort_index()
    return data


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    value = 100 - (100 / (1 + rs))
    neutral = (avg_gain.fillna(0.0) == 0.0) & (avg_loss.fillna(0.0) == 0.0)
    value = value.mask(neutral, 50.0)
    value = value.mask((avg_gain > 0.0) & (avg_loss == 0.0), 100.0)
    value = value.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return value.fillna(50.0)


def _minutes(value) -> int:
    return int(value.hour) * 60 + int(value.minute)


def _time_window_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    start_minute = _minutes(parse_hhmm(start))
    end_minute = _minutes(parse_hhmm(end))
    minutes = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    if start_minute <= end_minute:
        return (minutes >= start_minute) & (minutes < end_minute)
    return (minutes >= start_minute) | (minutes < end_minute)


def session_labels(index: pd.DatetimeIndex) -> np.ndarray:
    labels = np.full(len(index), "off", dtype=object)
    labels[_time_window_mask(index, "21:00", "01:00")] = "overlap"
    labels[_time_window_mask(index, "01:00", "06:00")] = "ny_tail"
    labels[_time_window_mask(index, "06:00", "16:00")] = "asia"
    labels[_time_window_mask(index, "16:00", "21:00")] = "london"
    return labels


def add_asia_range(frame: pd.DataFrame, start: str = "06:00", end: str = "16:00") -> pd.DataFrame:
    data = frame.copy()
    day_key = pd.Series(data.index.normalize(), index=data.index)
    asia_mask = _time_window_mask(data.index, start, end)
    if not asia_mask.any():
        data["asia_high"] = np.nan
        data["asia_low"] = np.nan
        data["asia_width"] = np.nan
        return data

    asia = data.loc[asia_mask]
    asia_day_key = day_key.loc[asia_mask]
    asia_high = asia["high"].groupby(asia_day_key).max()
    asia_low = asia["low"].groupby(asia_day_key).min()
    data["asia_high"] = day_key.map(asia_high).to_numpy()
    data["asia_low"] = day_key.map(asia_low).to_numpy()
    data["asia_width"] = data["asia_high"] - data["asia_low"]
    return data


def _rolling_last_percentile(values: np.ndarray) -> float:
    if len(values) == 0 or np.isnan(values[-1]):
        return np.nan
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    return float((valid <= values[-1]).mean())


def add_tick_microstructure_features(frame: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    data = frame.copy()
    cfg = config.indicators
    volume = data["volume"].astype(float)
    delta = data["close"].diff().fillna(0.0)
    buy_volume = volume.where(delta > 0.0, 0.0)
    sell_volume = volume.where(delta < 0.0, 0.0)
    signed_volume = buy_volume - sell_volume

    buy_roll = buy_volume.rolling(cfg.ofi_window, min_periods=1).sum()
    sell_roll = sell_volume.rolling(cfg.ofi_window, min_periods=1).sum()
    data["ofi"] = signed_volume.rolling(cfg.ofi_window, min_periods=1).sum()
    ratio = np.where(
        buy_roll >= sell_roll,
        buy_roll / sell_roll.replace(0.0, np.nan),
        -sell_roll / buy_roll.replace(0.0, np.nan),
    )
    ratio = pd.Series(ratio, index=data.index).replace([np.inf, -np.inf], np.nan)
    data["ofi_ratio"] = ratio.fillna(np.sign(data["ofi"]) * 99.0).clip(-99.0, 99.0)

    total_volume = volume.rolling(cfg.vpin_window, min_periods=1).sum()
    imbalance = signed_volume.rolling(cfg.vpin_window, min_periods=1).sum().abs()
    data["vpin"] = (imbalance / total_volume.replace(0.0, np.nan)).fillna(0.0)
    data["vpin_quantile"] = data["vpin"].rolling(
        cfg.vpin_quantile_lookback, min_periods=10
    ).apply(_rolling_last_percentile, raw=True)
    data["vpin_quantile"] = data["vpin_quantile"].fillna(0.0)
    return data


def compute_features(frame: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    data = ensure_datetime_index(frame, config.data.timezone, config.data.input_timezone)
    if "tick_volume" in data.columns and "volume" not in data.columns:
        data["volume"] = data["tick_volume"]
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in data.columns:
            raise ValueError(f"Missing required market data column: {column}")
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if {"bid", "ask"}.issubset(data.columns):
        data["spread"] = pd.to_numeric(data["ask"], errors="coerce") - pd.to_numeric(
            data["bid"], errors="coerce"
        )
    elif "spread" not in data.columns:
        data["spread"] = config.data.default_spread
    else:
        data["spread"] = pd.to_numeric(data["spread"], errors="coerce").fillna(
            config.data.default_spread
        )

    cfg = config.indicators
    data["atr"] = atr(data, cfg.atr_period)
    data["atr_quantile"] = data["atr"].rolling(
        cfg.atr_quantile_lookback, min_periods=10
    ).apply(_rolling_last_percentile, raw=True)
    data["atr_quantile"] = data["atr_quantile"].fillna(0.0)
    data["rsi"] = rsi(data["close"], cfg.rsi_period)
    data["session"] = session_labels(data.index)
    data["volume_median"] = (
        data["volume"].rolling(cfg.volume_window, min_periods=1).median().shift(1)
    )
    data["volume_median"] = data["volume_median"].fillna(data["volume"])
    data["volume_spike"] = data["volume"] > (
        data["volume_median"] * cfg.volume_spike_factor
    )
    data["fair_value"] = data["close"].rolling(cfg.fair_value_window, min_periods=1).mean()
    scalp_lookback = max(2, int(config.strategies.micro_scalp.lookback))
    data["micro_high"] = (
        data["high"].rolling(scalp_lookback, min_periods=scalp_lookback).max().shift(1)
    )
    data["micro_low"] = (
        data["low"].rolling(scalp_lookback, min_periods=scalp_lookback).min().shift(1)
    )
    data["spread_median"] = (
        data["spread"].rolling(cfg.spread_window, min_periods=1).median().shift(1)
    )
    data["spread_median"] = data["spread_median"].fillna(data["spread"])
    data = add_tick_microstructure_features(data, config)
    data = add_asia_range(
        data,
        config.strategies.breakout.asia_range_start,
        config.strategies.breakout.asia_range_end,
    )
    from .cross_assets import add_cross_asset_features

    data = add_cross_asset_features(data, config)
    return data
