from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SystemConfig
from .models import Side


def _ensure_time_index(frame: pd.DataFrame, timezone: str, input_timezone: str) -> pd.DataFrame:
    data = frame.copy()
    if "time" in data.columns:
        parsed = pd.to_datetime(data.pop("time"))
    else:
        parsed = pd.to_datetime(data.index)
    if isinstance(parsed, pd.Series):
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize(input_timezone)
        parsed = parsed.dt.tz_convert(timezone)
        data.index = pd.DatetimeIndex(parsed)
    else:
        index = pd.DatetimeIndex(parsed)
        if index.tz is None:
            index = index.tz_localize(input_timezone)
        data.index = index.tz_convert(timezone)
    return data.sort_index()


def _empty_cross_asset_columns(frame: pd.DataFrame, reason: str = "missing") -> pd.DataFrame:
    data = frame.copy()
    data["dxy"] = pd.NA
    data["us10y"] = pd.NA
    data["dxy_momentum"] = 0.0
    data["yield_momentum"] = 0.0
    data["cross_asset_score"] = 0
    data["cross_asset_bias"] = "missing" if reason == "missing" else "neutral"
    data["cross_asset_reason"] = reason
    return data


def load_cross_asset_csv(config: SystemConfig) -> pd.DataFrame:
    path = Path(config.cross_assets.csv_path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    data = _ensure_time_index(frame, config.data.timezone, config.data.input_timezone)
    required = [config.cross_assets.dxy_column, config.cross_assets.yield_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Cross-asset CSV missing columns: {missing}")
    data = data.rename(
        columns={
            config.cross_assets.dxy_column: "dxy",
            config.cross_assets.yield_column: "us10y",
        }
    )
    data["dxy"] = pd.to_numeric(data["dxy"], errors="coerce")
    data["us10y"] = pd.to_numeric(data["us10y"], errors="coerce")
    return data[["dxy", "us10y"]].dropna(how="all")


def sync_fred_cross_assets(
    config: SystemConfig,
    start: str | None = None,
    end: str | None = None,
    output: str | Path | None = None,
) -> Path:
    start_ts = pd.Timestamp(start or "2006-01-01")
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz=config.data.timezone).tz_localize(None)
    series = {
        "dxy": "DTWEXBGS",
        "us10y": "DGS10",
    }
    frames: list[pd.DataFrame] = []
    for column, series_id in series.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        frame = pd.read_csv(url)
        frame = frame.rename(columns={"observation_date": "date", series_id: column})
        frame["date"] = pd.to_datetime(frame["date"])
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame.set_index("date")[[column]])
    data = pd.concat(frames, axis=1).sort_index()
    data = data.loc[(data.index >= start_ts) & (data.index <= end_ts)]
    data = data.ffill().dropna(how="all")
    output_path = Path(output or config.cross_assets.csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FRED daily observations are not intraday timestamps. Expose them from the
    # following Beijing morning so the filter does not look into the future.
    out = data.reset_index()
    out["time"] = (
        out["date"]
        + pd.Timedelta(days=1)
        + pd.Timedelta(hours=6)
    ).dt.tz_localize(config.data.timezone)
    out[["time", "dxy", "us10y"]].to_csv(output_path, index=False)
    return output_path


def compute_cross_asset_bias(cross_assets: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    if cross_assets.empty:
        return cross_assets.copy()
    cfg = config.cross_assets
    data = cross_assets.copy().sort_index()
    data["dxy_momentum"] = data["dxy"] - data["dxy"].shift(cfg.momentum_lookback)
    data["yield_momentum"] = data["us10y"] - data["us10y"].shift(cfg.momentum_lookback)

    dxy_gold = pd.Series(0, index=data.index, dtype="int64")
    yield_gold = pd.Series(0, index=data.index, dtype="int64")
    dxy_gold = dxy_gold.mask(data["dxy_momentum"] <= -cfg.dxy_min_change, 1)
    dxy_gold = dxy_gold.mask(data["dxy_momentum"] >= cfg.dxy_min_change, -1)
    yield_gold = yield_gold.mask(data["yield_momentum"] <= -cfg.yield_min_change, 1)
    yield_gold = yield_gold.mask(data["yield_momentum"] >= cfg.yield_min_change, -1)

    data["cross_asset_score"] = dxy_gold + yield_gold
    data["cross_asset_bias"] = "neutral"
    data.loc[data["cross_asset_score"] >= cfg.required_components, "cross_asset_bias"] = "long"
    data.loc[data["cross_asset_score"] <= -cfg.required_components, "cross_asset_bias"] = "short"
    data["cross_asset_reason"] = (
        "dxy_momentum="
        + data["dxy_momentum"].round(6).astype(str)
        + ";yield_momentum="
        + data["yield_momentum"].round(6).astype(str)
    )
    return data


def add_cross_asset_features(
    market_data: pd.DataFrame,
    config: SystemConfig,
    cross_assets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not config.cross_assets.enabled:
        return _empty_cross_asset_columns(market_data, reason="disabled")
    if cross_assets is None:
        cross_assets = load_cross_asset_csv(config)
    if cross_assets.empty:
        reason = "missing" if config.cross_assets.neutral_if_missing else "unavailable"
        return _empty_cross_asset_columns(market_data, reason=reason)

    features = compute_cross_asset_bias(cross_assets, config)
    if features.empty:
        return _empty_cross_asset_columns(market_data, reason="missing")

    market = market_data.copy().sort_index()
    left = market.reset_index(names="time")
    right = features.reset_index(names="time")
    merged = pd.merge_asof(
        left,
        right,
        on="time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=config.cross_assets.asof_tolerance_minutes),
    )
    merged = merged.set_index("time")
    for column in ["dxy_momentum", "yield_momentum", "cross_asset_score"]:
        merged[column] = merged[column].fillna(0.0)
    merged["cross_asset_bias"] = merged["cross_asset_bias"].fillna("missing")
    merged["cross_asset_reason"] = merged["cross_asset_reason"].fillna("missing")
    return merged


def cross_asset_allows(side: Side, row: pd.Series, config: SystemConfig) -> tuple[bool, str]:
    if not config.cross_assets.enabled:
        return True, "cross-asset filter disabled"
    bias = str(row.get("cross_asset_bias", "missing"))
    if bias == "missing" and config.cross_assets.neutral_if_missing:
        return True, "cross-asset data missing"
    if bias not in {"long", "short", "neutral", "missing"}:
        return True, f"unknown cross-asset bias: {bias}"
    if config.cross_assets.require_confirmation and bias != side.value:
        return False, f"cross-asset confirmation required: {bias}"
    if config.cross_assets.block_on_conflict and bias in {"long", "short"} and bias != side.value:
        return False, f"cross-asset conflict: {bias}"
    return True, f"cross-asset bias: {bias}"
