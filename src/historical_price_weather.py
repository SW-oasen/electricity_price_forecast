"""Archived ECMWF weather runs for the leakage-safe price backtest."""

from __future__ import annotations

import sqlite3
import subprocess
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
from src.price_input_lineage import record_input_lineage, record_weather_rejection
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
    evaluation_id: str | None = None,
) -> dict[str, int]:
    """Persist the latest ECMWF run available at the price decision time.

    Values cover the target day and are stored separately for PV and wind
    because their capacity-weighted location sets differ.
    """
    target_start = protocol.target_start(target_date)
    as_of = protocol.as_of(target_date)
    forecast_origin = target_start - pd.DateOffset(days=1)
    selected_run = _latest_complete_run(as_of, forecast_origin)
    target_end = target_start + pd.DateOffset(days=1)
    expected_times = pd.date_range(forecast_origin, target_end, freq="h", inclusive="left")

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
        forecast = None
        run = None
        # Single Runs can contain null values or omit an archived cycle.  An
        # older, already available run is valid as long as it fully covers D-1/D.
        for offset_hours in range(0, 43, 6):
            candidate = selected_run - pd.Timedelta(hours=offset_hours)
            hours_to_target_end = (
                target_end.tz_convert("UTC") - candidate.tz_convert("UTC")
            ).total_seconds() / 3600
            forecast_days = max(3, int((hours_to_target_end + 23) // 24))
            try:
                candidate_forecast = client.fetch_single_run(
                    candidate, forecast_days=forecast_days, model="ecmwf_ifs",
                    cache_dir=cache_dir / key,
                )
            except subprocess.CalledProcessError:
                candidate_forecast = None
            if candidate_forecast is None:
                if evaluation_id is not None:
                    record_weather_rejection(
                        conn, evaluation_id=evaluation_id, target_date=target_date,
                        aggregation_key=key,
                        candidate_initialized_at_utc=candidate.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        rejection_reason="api_unavailable",
                    )
                continue
            candidate_forecast = candidate_forecast[
                (candidate_forecast["time"] >= forecast_origin)
                & (candidate_forecast["time"] < target_end)
            ].drop_duplicates("time").sort_values("time")
            complete = (
                set(expected_times).issubset(set(candidate_forecast["time"]))
                and candidate_forecast[variables].notna().all().all()
            )
            if complete:
                forecast = candidate_forecast
                run = candidate
                break
            if evaluation_id is not None:
                reason = (
                    "missing_values"
                    if candidate_forecast[variables].isna().any().any()
                    else "incomplete_horizon"
                )
                record_weather_rejection(
                    conn, evaluation_id=evaluation_id, target_date=target_date,
                    aggregation_key=key,
                    candidate_initialized_at_utc=candidate.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    rejection_reason=reason,
                )
        if forecast is None or run is None:
            raise ValueError(f"{key} has no complete archived weather run for {target_date}")
        run_available = run + pd.Timedelta(hours=6)
        if run_available > as_of:
            raise RuntimeError("selected weather run is not yet available at as_of")
        stored[key] = store_weather_forecast_run(
            conn, forecast,
            provider="open-meteo", model="ecmwf_ifs",
            initialized_at_utc=run.strftime("%Y-%m-%dT%H:%M:%SZ"),
            available_at_utc=run_available.strftime("%Y-%m-%dT%H:%M:%SZ"),
            aggregation_key=key, aggregation_version=f"capacity-weights-{target_start.year}",
            series_ids=series_ids, source_url="https://single-runs-api.open-meteo.com/v1/forecast",
        )
        if evaluation_id is not None:
            offset_hours = int((selected_run - run).total_seconds() / 3600)
            record_input_lineage(
                conn, evaluation_id=evaluation_id, target_date=target_date,
                input_group=key, source="open-meteo",
                selected_forecast_run_id=stored[key],
                preferred_initialized_at_utc=selected_run.strftime("%Y-%m-%dT%H:%M:%SZ"),
                fallback_offset_hours=offset_hours,
                fallback_type="older_weather_run" if offset_hours else None,
                selection_reason="complete_archived_run",
            )
    return stored


def fetch_and_store_demand_weather_for_target(
    conn: sqlite3.Connection,
    target_date: str,
    *,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
    cache_dir: Path = SINGLE_RUN_CACHE_DIR,
    evaluation_id: str | None = None,
) -> int:
    """Store the archived ECMWF run used by the upstream demand model."""
    target_start = protocol.target_start(target_date)
    as_of = protocol.as_of(target_date)
    forecast_origin = target_start - pd.DateOffset(days=1)
    selected_run = _latest_complete_run(as_of, forecast_origin)
    client = OpenMeteoClient(SELECTED_CITIES, CITY_POPULATION, WEATHER_VARIABLES, city_sleep=0.1)
    target_end = target_start + pd.DateOffset(days=1)
    expected_times = pd.date_range(forecast_origin, target_end, freq="h", inclusive="left")
    demand_cache = cache_dir / "demand_cities_de"

    def fetch_forecast(run, cache):
        # Older fallback cycles need a longer requested horizon.  Keeping this
        # at three days makes the final hours of D unavailable once we fall
        # back by even one six-hour cycle.
        hours_to_target_end = (
            target_end.tz_convert("UTC") - run.tz_convert("UTC")
        ).total_seconds() / 3600
        forecast_days = max(3, int((hours_to_target_end + 23) // 24))
        try:
            result = client.fetch_single_run(
                run, forecast_days=forecast_days, model="ecmwf_ifs", cache_dir=cache
            )
        except subprocess.CalledProcessError:
            # The Single Runs endpoint does not retain every historical cycle.
            # A missing candidate must not abort the walk-forward evaluation.
            return None
        result = result[(result["time"] >= forecast_origin) & (result["time"] < target_end)]
        result = result.drop_duplicates("time").sort_values("time")
        complete = (
            set(expected_times).issubset(set(result["time"]))
            and result[list(DEMAND_FORECAST_WEATHER_SERIES_IDS)].notna().all().all()
        )
        return result if complete else None

    forecast = None
    run = None
    # A model cycle can be incomplete in the historical single-run endpoint.
    # Try older cycles, all of which were available by the decision timestamp.
    for offset_hours in range(0, 43, 6):
        candidate = selected_run - pd.Timedelta(hours=offset_hours)
        forecast = fetch_forecast(candidate, demand_cache)
        if forecast is None:
            # Ignore a stale/incomplete cache entry and retry the API.
            forecast = fetch_forecast(candidate, None)
        if forecast is None and evaluation_id is not None:
            record_weather_rejection(
                conn, evaluation_id=evaluation_id, target_date=target_date,
                aggregation_key="demand_cities_de",
                candidate_initialized_at_utc=candidate.strftime("%Y-%m-%dT%H:%M:%SZ"),
                rejection_reason="missing_or_incomplete_weather",
            )
        if forecast is not None:
            run = candidate
            break
    if forecast is None or run is None:
        raise ValueError(
            f"Demand weather forecast does not completely cover D-1/D for {target_date}"
        )
    stored = store_weather_forecast_run(
        conn, forecast, provider="open-meteo", model="ecmwf_ifs",
        initialized_at_utc=run.strftime("%Y-%m-%dT%H:%M:%SZ"),
        available_at_utc=(run + pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        aggregation_key="demand_cities_de", aggregation_version="population-v1",
        series_ids=DEMAND_FORECAST_WEATHER_SERIES_IDS,
        source_url="https://single-runs-api.open-meteo.com/v1/forecast",
    )
    if evaluation_id is not None:
        offset_hours = int((selected_run - run).total_seconds() / 3600)
        record_input_lineage(
            conn, evaluation_id=evaluation_id, target_date=target_date,
            input_group="demand_cities_de", source="open-meteo",
            selected_forecast_run_id=stored,
            preferred_initialized_at_utc=selected_run.strftime("%Y-%m-%dT%H:%M:%SZ"),
            fallback_offset_hours=offset_hours,
            fallback_type="older_weather_run" if offset_hours else None,
            selection_reason="complete_archived_run",
        )
    return stored
