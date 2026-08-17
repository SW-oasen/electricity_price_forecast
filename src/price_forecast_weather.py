"""Load archived PV/wind weather as point-in-time price-model inputs."""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.config import (
    PV_WEATHER_SERIES_IDS,
    TABLE_WEATHER_FORECAST_RUNS,
    TABLE_WEATHER_FORECAST_VALUES,
    WIND_WEATHER_SERIES_IDS,
)
from src.forecast_protocol import PRICE_WALK_FORWARD_PROTOCOL, PriceWalkForwardProtocol


def _load_complete_run(conn, key, series_ids, target_date, protocol):
    as_of = protocol.as_of(target_date).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = pd.read_sql_query(
        f"""SELECT forecast_run_id FROM {TABLE_WEATHER_FORECAST_RUNS}
        WHERE aggregation_key = ? AND available_at_utc <= ?
        ORDER BY available_at_utc DESC, forecast_run_id DESC""",
        conn, params=(key, as_of),
    )
    origin = protocol.target_start(target_date) - pd.DateOffset(days=1)
    end = protocol.target_start(target_date) + pd.DateOffset(days=1)
    expected = set(pd.date_range(origin, end, freq="h", inclusive="left"))
    inverse = {value: name for name, value in series_ids.items()}
    for run_id in runs["forecast_run_id"]:
        rows = pd.read_sql_query(
            f"SELECT valid_time_utc, series_id, value FROM {TABLE_WEATHER_FORECAST_VALUES} WHERE forecast_run_id = ?",
            conn, params=(int(run_id),),
        )
        rows["variable"] = rows.series_id.map(inverse)
        values = rows.pivot(index="valid_time_utc", columns="variable", values="value").reset_index()
        values = values.rename(columns={"valid_time_utc": "time"})
        times = pd.to_datetime(values.time, utc=True).dt.tz_convert(protocol.timezone)
        if expected.issubset(set(times)) and set(series_ids).issubset(values.columns):
            return values
    raise ValueError(f"No complete archived {key} weather run is stored for {target_date}")


def inject_archived_price_weather(
    price_data: pd.DataFrame,
    conn: sqlite3.Connection,
    target_date: str,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> pd.DataFrame:
    """Replace PV/wind weather in D-1/D, including later price-model lags."""
    data = price_data.copy()
    data["time"] = pd.to_datetime(data.time, utc=True).dt.tz_convert(protocol.timezone)
    origin = protocol.target_start(target_date) - pd.DateOffset(days=1)
    end = protocol.target_start(target_date) + pd.DateOffset(days=1)
    mask = (data.time >= origin) & (data.time < end)
    for key, ids, prefix in (
        ("pv_clusters_de", PV_WEATHER_SERIES_IDS, "pv_weather_"),
        ("wind_clusters_de", WIND_WEATHER_SERIES_IDS, "wind_weather_"),
    ):
        forecast = _load_complete_run(conn, key, ids, target_date, protocol)
        forecast["time"] = pd.to_datetime(forecast.time, utc=True).dt.tz_convert(protocol.timezone)
        indexed = forecast.set_index("time")
        columns = {name: f"{prefix}{name}" for name in ids}
        missing = set(columns.values()) - set(data.columns)
        if missing:
            raise ValueError(f"Price inputs lack weather columns: {sorted(missing)}")
        data.loc[mask, list(columns.values())] = indexed.loc[data.loc[mask, "time"], list(columns)].rename(columns=columns).to_numpy()
    return data
