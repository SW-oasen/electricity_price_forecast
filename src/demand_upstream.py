"""Recursive demand forecasts used as point-in-time price-model inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.demand_forecast_weather import (
    inject_archived_demand_weather,
    load_archived_demand_weather,
)
from src.etl_demand import load_combined_data
from src.fetch_demand_data import create_time_based_features
from src.forecast_protocol import PRICE_WALK_FORWARD_PROTOCOL, PriceWalkForwardProtocol


TARGET = "energy_demand_mwh"
NON_FEATURES = {"time", TARGET, "smard_forecast_mwh", "data_source"}


def _ensure_demand_horizon_rows(
    combined_data: pd.DataFrame,
    target_date: str,
    protocol: PriceWalkForwardProtocol,
) -> pd.DataFrame:
    """Add calendar-only D-1/D rows when forecasting beyond stored actuals."""
    data = combined_data.copy()
    data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_convert(protocol.timezone)
    target_start = protocol.target_start(target_date)
    expected = pd.date_range(
        target_start - pd.DateOffset(days=1),
        target_start + pd.DateOffset(days=1),
        freq="h", inclusive="left",
    )
    missing = expected.difference(pd.DatetimeIndex(data["time"]))
    if missing.empty:
        return data
    generated = create_time_based_features(
        pd.DataFrame({"time": missing}), in_year=target_start.year
    )
    for column in data.columns:
        if column not in generated:
            generated[column] = pd.NA
    for column in generated.columns:
        if column not in data:
            data[column] = pd.NA
    return pd.concat([data, generated[data.columns]], ignore_index=True)


def _load_history_features(history: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(history, errors="coerce").dropna()
    if len(values) < 168:
        raise ValueError("At least 168 preceding demand values are required")
    return {
        "energy_demand_lag_24h": float(values.iloc[-24]),
        "energy_demand_lag_168h": float(values.iloc[-168]),
        "energy_demand_rolling_mean_24h": float(values.iloc[-24:].mean()),
        "energy_demand_rolling_mean_168h": float(values.iloc[-168:].mean()),
    }


def predict_demand_for_price_target(
    model,
    combined_data: pd.DataFrame,
    target_date: str,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> pd.DataFrame:
    """Recursively forecast D-1 and D with only actual demand through D-2.

    ``combined_data`` must provide calendar and *forecast-weather* feature
    columns for the two-day horizon.  It deliberately does not create weather
    inputs, so their historical availability remains explicit in the caller.
    """
    data = combined_data.sort_values("time").drop_duplicates("time").copy()
    data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_convert(protocol.timezone)
    target_start = protocol.target_start(target_date)
    origin = target_start - pd.DateOffset(days=1)
    target_end = target_start + pd.DateOffset(days=1)
    history = data.loc[data["time"] < origin].set_index("time")[TARGET].dropna().copy()
    horizon = data.loc[(data["time"] >= origin) & (data["time"] < target_end)].copy()
    if horizon.empty:
        raise ValueError("Demand source lacks D-1/D forecast horizon")

    rows = []
    names = list(getattr(model, "feature_name_", getattr(model, "feature_names_in_", [])))
    for _, row in horizon.iterrows():
        features = row.drop(labels=list(NON_FEATURES), errors="ignore").copy()
        for name, value in _load_history_features(history).items():
            features.loc[name] = value
        frame = features.to_frame().T.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
        if frame.isna().any().any():
            missing = frame.columns[frame.isna().any()].tolist()
            raise ValueError(f"Demand forecast features unavailable: {missing}")
        prediction = float(np.asarray(model.predict(frame)).reshape(-1)[0])
        history.loc[row["time"]] = prediction
        rows.append((row["time"], prediction))
    return pd.DataFrame(rows, columns=["time", "demand_forecast_ml_mwh"])


def predict_demand_for_price_target_from_db(
    model,
    conn,
    target_date: str,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> pd.DataFrame:
    """Forecast D-1/D from database inputs available at the price decision.

    Demand observations after D-2 are deliberately ignored by the recursive
    forecaster.  D-1/D weather is sourced from a persisted archived model run.
    """
    target_start = protocol.target_start(target_date)
    start = (target_start - pd.DateOffset(days=9)).strftime("%Y-%m-%d")
    end = target_start.strftime("%Y-%m-%d")
    combined = load_combined_data(conn, start_date=start, end_date=end)
    combined = _ensure_demand_horizon_rows(combined, target_date, protocol)
    forecast_weather = load_archived_demand_weather(conn, target_date, protocol)
    prepared = inject_archived_demand_weather(combined, forecast_weather, target_date, protocol)
    return predict_demand_for_price_target(model, prepared, target_date, protocol)
