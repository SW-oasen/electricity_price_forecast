"""Reconstruct auditable weather-fallback lineage from persisted forecast runs."""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.config import (
    DATABASE_PATH,
    DEMAND_FORECAST_WEATHER_SERIES_IDS,
    PV_WEATHER_SERIES_IDS,
    TABLE_WEATHER_FORECAST_RUNS,
    TABLE_WEATHER_FORECAST_VALUES,
    WIND_WEATHER_SERIES_IDS,
)
from src.forecast_protocol import PRICE_WALK_FORWARD_PROTOCOL, PriceWalkForwardProtocol
from src.historical_price_weather import _latest_complete_run
from src.price_input_lineage import record_input_lineage, record_weather_rejection


WEATHER_INPUTS = {
    "demand_cities_de": DEMAND_FORECAST_WEATHER_SERIES_IDS,
    "pv_clusters_de": PV_WEATHER_SERIES_IDS,
    "wind_clusters_de": WIND_WEATHER_SERIES_IDS,
}


def _complete_run_for_target(
    conn: sqlite3.Connection, key: str, series_ids: dict[str, str],
    target_date: str, protocol: PriceWalkForwardProtocol,
) -> tuple[int, pd.Timestamp] | None:
    as_of = protocol.as_of(target_date).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    origin = protocol.target_start(target_date) - pd.DateOffset(days=1)
    end = protocol.target_start(target_date) + pd.DateOffset(days=1)
    expected = set(pd.date_range(origin, end, freq="h", inclusive="left"))
    inverse = {value: name for name, value in series_ids.items()}
    runs = pd.read_sql_query(
        f"""SELECT forecast_run_id, initialized_at_utc FROM {TABLE_WEATHER_FORECAST_RUNS}
        WHERE aggregation_key=? AND available_at_utc <= ?
        ORDER BY available_at_utc DESC, forecast_run_id DESC""",
        conn, params=(key, as_of),
    )
    for row in runs.itertuples(index=False):
        values = pd.read_sql_query(
            f"""SELECT valid_time_utc, series_id, value
            FROM {TABLE_WEATHER_FORECAST_VALUES} WHERE forecast_run_id=?""",
            conn, params=(int(row.forecast_run_id),),
        )
        values["variable"] = values["series_id"].map(inverse)
        values = values.pivot(index="valid_time_utc", columns="variable", values="value").reset_index()
        times = pd.to_datetime(values["valid_time_utc"], utc=True).dt.tz_convert(protocol.timezone)
        if (
            expected.issubset(set(times))
            and set(series_ids).issubset(values.columns)
            and values[list(series_ids)].notna().all().all()
        ):
            return int(row.forecast_run_id), pd.Timestamp(row.initialized_at_utc)
    return None


def backfill_weather_fallbacks(
    conn: sqlite3.Connection,
    start: str,
    end: str,
    *,
    evaluation_id: str = "reconstructed-weather-fallbacks-2026-08-20",
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> int:
    """Write reconstructed fallback decisions for delivery days in ``[start, end)``.

    The function records only cases where a complete stored run is older than
    the preferred as-of run. Rejected API calls cannot be reconstructed and
    are intentionally not fabricated.
    """
    conn.execute(
        "DELETE FROM weather_run_rejections WHERE evaluation_id=? AND record_origin='reconstructed'",
        (evaluation_id,),
    )
    conn.execute(
        "DELETE FROM price_walk_forward_input_lineage WHERE evaluation_id=? AND record_origin='reconstructed'",
        (evaluation_id,),
    )
    conn.commit()
    count = 0
    for target in pd.date_range(start, end, freq="D", inclusive="left"):
        target_date = target.date().isoformat()
        preferred = _latest_complete_run(
            protocol.as_of(target_date),
            protocol.target_start(target_date) - pd.DateOffset(days=1),
        )
        for key, series_ids in WEATHER_INPUTS.items():
            selected = _complete_run_for_target(conn, key, series_ids, target_date, protocol)
            if selected is None:
                continue
            run_id, initialized = selected
            offset = int((preferred - initialized).total_seconds() / 3600)
            if offset <= 0:
                continue
            record_input_lineage(
                conn, evaluation_id=evaluation_id, target_date=target_date,
                input_group=key, source="open-meteo", selected_forecast_run_id=run_id,
                preferred_initialized_at_utc=preferred.strftime("%Y-%m-%dT%H:%M:%SZ"),
                fallback_offset_hours=offset, fallback_type="older_weather_run",
                selection_reason="reconstructed_from_persisted_runs",
                record_origin="reconstructed",
            )
            record_weather_rejection(
                conn, evaluation_id=evaluation_id, target_date=target_date,
                aggregation_key=key,
                candidate_initialized_at_utc=preferred.strftime("%Y-%m-%dT%H:%M:%SZ"),
                rejection_reason="preferred_run_not_selected",
                details="Reconstructed from the selected complete persisted run; original API failure details are unavailable.",
                record_origin="reconstructed",
            )
            count += 1
    return count


if __name__ == "__main__":
    with sqlite3.connect(DATABASE_PATH) as connection:
        print(backfill_weather_fallbacks(connection, "2025-10-01", "2026-08-01"))
