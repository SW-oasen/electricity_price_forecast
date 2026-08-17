"""Archived ECMWF weather runs for the leakage-safe price backtest."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import (
    PV_CLUSTER_LOCATIONS,
    PV_CLUSTER_YEARLY_CAPACITY_PATH,
    PV_WEATHER_SERIES_IDS,
    PV_WEATHER_VARIABLES,
    WIND_CLUSTER_LOCATIONS,
    WIND_CLUSTER_YEARLY_CAPACITY_PATH,
    WIND_WEATHER_SERIES_IDS,
    WIND_WEATHER_VARIABLES,
    SELECTED_CITIES,
    CITY_POPULATION,
    WEATHER_VARIABLES,
    DEMAND_FORECAST_WEATHER_SERIES_IDS,
)
from src.etl_price import store_weather_forecast_run
from src.forecast_protocol import PRICE_WALK_FORWARD_PROTOCOL, PriceWalkForwardProtocol
from util.openmeteo_client import OpenMeteoClient
from util.weather_weighted import build_yearly_weights


SINGLE_RUN_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "openmeteo_single_runs"


def _technology_client(
    locations: dict, weights: dict[str, float], variables: list[str]
) -> OpenMeteoClient:
    # OpenMeteoClient normalizes these weights internally; floats are intentional.
    return OpenMeteoClient(locations, weights, variables, city_sleep=0.1)


def _latest_complete_run(as_of: pd.Timestamp, first_valid_time: pd.Timestamp) -> pd.Timestamp:
    """Choose a published six-hour run whose forecast starts before the horizon."""
    latest = OpenMeteoClient.latest_available_run(as_of)
    horizon_start_utc = pd.Timestamp(first_valid_time).tz_convert("UTC")
    latest_covering = horizon_start_utc.normalize() + pd.Timedelta(
        hours=((horizon_start_utc.hour // 6) * 6)
    )
    return min(latest, latest_covering)


def fetch_and_store_weather_for_target(
    conn: sqlite3.Connection,
    target_date: str,
    *,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
    cache_dir: Path = SINGLE_RUN_CACHE_DIR,
    client_factory: Callable[[dict, dict[str, float], list[str]], OpenMeteoClient] = _technology_client,
) -> dict[str, int]:
    """Persist the latest ECMWF run available at the price decision time.

    Values cover the target day and are stored separately for PV and wind
    because their capacity-weighted location sets differ.
    """
    target_start = protocol.target_start(target_date)
    as_of = protocol.as_of(target_date)
    forecast_origin = target_start - pd.DateOffset(days=1)
    run = _latest_complete_run(as_of, forecast_origin)
    run_available = run + pd.Timedelta(hours=6)
    if run_available > as_of:
        raise RuntimeError("selected weather run is not yet available at as_of")

    specs = (
        ("pv_clusters_de", PV_CLUSTER_LOCATIONS, PV_CLUSTER_YEARLY_CAPACITY_PATH,
         "pv", PV_WEATHER_VARIABLES, PV_WEATHER_SERIES_IDS),
        ("wind_clusters_de", WIND_CLUSTER_LOCATIONS, WIND_CLUSTER_YEARLY_CAPACITY_PATH,
         "wind", WIND_WEATHER_VARIABLES, WIND_WEATHER_SERIES_IDS),
    )
    stored: dict[str, int] = {}
    for key, locations, capacity_path, prefix, variables, series_ids in specs:
        weights_by_year = build_yearly_weights(capacity_path, prefix)
        weights = weights_by_year[target_start.year]
        common = set(locations) & set(weights)
        client = client_factory(
            {name: locations[name] for name in common},
            {name: weights[name] for name in common},
            variables,
        )
        forecast = client.fetch_single_run(
            run, forecast_days=3, model="ecmwf_ifs", cache_dir=cache_dir / key
        )
        target_end = target_start + pd.DateOffset(days=1)
        forecast = forecast[
            (forecast["time"] >= forecast_origin)
            & (forecast["time"] < target_end)
        ].copy()
        expected_hours = len(pd.date_range(
            forecast_origin,
            target_end,
            freq="h",
            inclusive="left",
        ))
        if len(forecast) != expected_hours:
            raise ValueError(f"{key} run does not cover the complete target day")
        stored[key] = store_weather_forecast_run(
            conn, forecast,
            provider="open-meteo", model="ecmwf_ifs",
            initialized_at_utc=run.strftime("%Y-%m-%dT%H:%M:%SZ"),
            available_at_utc=run_available.strftime("%Y-%m-%dT%H:%M:%SZ"),
            aggregation_key=key, aggregation_version=f"capacity-weights-{target_start.year}",
            series_ids=series_ids, source_url="https://single-runs-api.open-meteo.com/v1/forecast",
        )
    return stored


def fetch_and_store_demand_weather_for_target(
    conn: sqlite3.Connection,
    target_date: str,
    *,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
    cache_dir: Path = SINGLE_RUN_CACHE_DIR,
) -> int:
    """Store the archived ECMWF run used by the upstream demand model."""
    target_start = protocol.target_start(target_date)
    as_of = protocol.as_of(target_date)
    forecast_origin = target_start - pd.DateOffset(days=1)
    run = _latest_complete_run(as_of, forecast_origin)
    client = OpenMeteoClient(SELECTED_CITIES, CITY_POPULATION, WEATHER_VARIABLES, city_sleep=0.1)
    forecast = client.fetch_single_run(run, forecast_days=3, model="ecmwf_ifs", cache_dir=cache_dir / "demand_cities_de")
    target_end = target_start + pd.DateOffset(days=1)
    forecast = forecast[(forecast["time"] >= forecast_origin) & (forecast["time"] < target_end)]
    return store_weather_forecast_run(
        conn, forecast, provider="open-meteo", model="ecmwf_ifs",
        initialized_at_utc=run.strftime("%Y-%m-%dT%H:%M:%SZ"),
        available_at_utc=(run + pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        aggregation_key="demand_cities_de", aggregation_version="population-v1",
        series_ids=DEMAND_FORECAST_WEATHER_SERIES_IDS,
        source_url="https://single-runs-api.open-meteo.com/v1/forecast",
    )
