"""Archived forecast-weather inputs for the frozen upstream demand model."""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.config import (
    BASE_TEMPERATURE_COOLING,
    BASE_TEMPERATURE_HEATING,
    DEMAND_FORECAST_WEATHER_SERIES_IDS,
    TABLE_WEATHER_FORECAST_RUNS,
    TABLE_WEATHER_FORECAST_VALUES,
    WEATHER_VARIABLES,
)
from src.forecast_protocol import PRICE_WALK_FORWARD_PROTOCOL, PriceWalkForwardProtocol


ENGINEERED_WEATHER_COLUMNS = [
    *WEATHER_VARIABLES,
    "apparent_temperature_lag_24h",
    "apparent_temperature_rolling_mean_24h",
    "shortwave_radiation_0m_lag_24h",
    "shortwave_radiation_0m_rolling_mean_24h",
    "heating_degree",
    "cooling_degree",
]


def _engineer_weather(weather: pd.DataFrame) -> pd.DataFrame:
    data = weather.sort_values("time").drop_duplicates("time", keep="last").copy()
    data["apparent_temperature_rolling_mean_24h"] = (
        data["apparent_temperature"].shift(1).rolling(24).mean()
    )
    data["apparent_temperature_lag_24h"] = data["apparent_temperature"].shift(24)
    data["shortwave_radiation_0m_rolling_mean_24h"] = (
        data["shortwave_radiation"].shift(1).rolling(24).mean()
    )
    data["shortwave_radiation_0m_lag_24h"] = data["shortwave_radiation"].shift(24)
    data["heating_degree"] = (BASE_TEMPERATURE_HEATING - data["apparent_temperature"]).clip(lower=0)
    data["cooling_degree"] = (data["apparent_temperature"] - BASE_TEMPERATURE_COOLING).clip(lower=0)
    return data


def inject_archived_demand_weather(
    combined_data: pd.DataFrame,
    forecast_weather: pd.DataFrame,
    target_date: str,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> pd.DataFrame:
    """Replace D-1/D weather by one as-of archived forecast run.

    The preceding 24 observed hours are retained solely to calculate weather
    lag/rolling features.  No observed D-1/D weather survives this operation.
    """
    data = combined_data.sort_values("time").drop_duplicates("time").copy()
    data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_convert(protocol.timezone)
    forecast = forecast_weather.copy()
    forecast["time"] = pd.to_datetime(forecast["time"], utc=True).dt.tz_convert(protocol.timezone)
    missing = set(WEATHER_VARIABLES) - set(forecast.columns)
    if missing:
        raise ValueError(f"forecast weather lacks columns: {sorted(missing)}")

    target_start = protocol.target_start(target_date)
    origin = target_start - pd.DateOffset(days=1)
    target_end = target_start + pd.DateOffset(days=1)
    expected = pd.date_range(origin, target_end, freq="h", inclusive="left")
    horizon = forecast.loc[(forecast["time"] >= origin) & (forecast["time"] < target_end)]
    horizon = horizon.set_index("time").reindex(expected).reset_index(names="time")
    if horizon[WEATHER_VARIABLES].isna().any().any():
        raise ValueError("Archived demand weather does not fully cover D-1/D")
    context = data.loc[data["time"] < origin, ["time", *WEATHER_VARIABLES]].tail(24)
    if len(context) != 24 or context[WEATHER_VARIABLES].isna().any().any():
        raise ValueError("At least 24 complete observed weather rows are required")
    engineered = _engineer_weather(pd.concat([context, horizon], ignore_index=True))
    engineered = engineered.loc[engineered["time"] >= origin, ["time", *ENGINEERED_WEATHER_COLUMNS]]
    if engineered[ENGINEERED_WEATHER_COLUMNS].isna().any().any():
        raise ValueError("Archived weather feature engineering produced missing values")
    indexed = engineered.set_index("time")
    mask = (data["time"] >= origin) & (data["time"] < target_end)
    data.loc[mask, ENGINEERED_WEATHER_COLUMNS] = indexed.loc[data.loc[mask, "time"], ENGINEERED_WEATHER_COLUMNS].to_numpy()
    return data


def load_archived_demand_weather(
    conn: sqlite3.Connection,
    target_date: str,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> pd.DataFrame:
    """Load the newest demand-weather run that was available at the decision time."""
    as_of = protocol.as_of(target_date).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = pd.read_sql_query(
        f"""SELECT forecast_run_id FROM {TABLE_WEATHER_FORECAST_RUNS}
        WHERE aggregation_key = ? AND available_at_utc <= ?
        ORDER BY available_at_utc DESC, forecast_run_id DESC""",
        conn, params=("demand_cities_de", as_of),
    )
    if runs.empty:
        raise ValueError(f"No archived demand weather is stored for {target_date}")
    inverse_ids = {series_id: name for name, series_id in DEMAND_FORECAST_WEATHER_SERIES_IDS.items()}
    target_start = protocol.target_start(target_date)
    origin = target_start - pd.DateOffset(days=1)
    expected = pd.date_range(origin, target_start + pd.DateOffset(days=1), freq="h", inclusive="left")
    for run_id in runs["forecast_run_id"]:
        rows = pd.read_sql_query(
            f"""SELECT valid_time_utc, series_id, value FROM {TABLE_WEATHER_FORECAST_VALUES}
            WHERE forecast_run_id = ?""",
            conn, params=(int(run_id),),
        )
        rows["variable"] = rows["series_id"].map(inverse_ids)
        forecast = rows.pivot(index="valid_time_utc", columns="variable", values="value").reset_index()
        forecast = forecast.rename(columns={"valid_time_utc": "time"})
        times = pd.to_datetime(forecast["time"], utc=True).dt.tz_convert(protocol.timezone)
        complete = (
            set(expected).issubset(set(times))
            and set(WEATHER_VARIABLES).issubset(forecast.columns)
            and forecast[WEATHER_VARIABLES].notna().all().all()
        )
        if complete:
            return forecast
    raise ValueError(f"No complete archived demand weather run is stored for {target_date}")
