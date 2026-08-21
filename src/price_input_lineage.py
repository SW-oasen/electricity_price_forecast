"""Audit helpers for point-in-time inputs of price walk-forward runs."""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.config import TABLE_PRICE_INPUT_LINEAGE, TABLE_WEATHER_RUN_REJECTIONS


def record_input_lineage(
    conn: sqlite3.Connection,
    *,
    evaluation_id: str,
    target_date: str,
    input_group: str,
    source: str,
    selected_forecast_run_id: int | None = None,
    preferred_initialized_at_utc: str | None = None,
    fallback_offset_hours: int = 0,
    availability_status: str = "available",
    fallback_type: str | None = None,
    selection_reason: str | None = None,
    record_origin: str = "live",
) -> None:
    """Upsert the selected input for one target day and input group."""
    conn.execute(
        f"""INSERT INTO {TABLE_PRICE_INPUT_LINEAGE}
        (evaluation_id, target_date, input_group, source, selected_forecast_run_id,
         preferred_initialized_at_utc, fallback_offset_hours, availability_status,
         fallback_type, selection_reason, record_origin)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(evaluation_id, target_date, input_group) DO UPDATE SET
            source=excluded.source,
            selected_forecast_run_id=excluded.selected_forecast_run_id,
            preferred_initialized_at_utc=excluded.preferred_initialized_at_utc,
            fallback_offset_hours=excluded.fallback_offset_hours,
            availability_status=excluded.availability_status,
            fallback_type=excluded.fallback_type,
            selection_reason=excluded.selection_reason,
            record_origin=excluded.record_origin,
            recorded_at_utc=datetime('now')""",
        (
            evaluation_id, target_date, input_group, source,
            selected_forecast_run_id, preferred_initialized_at_utc,
            fallback_offset_hours, availability_status, fallback_type,
            selection_reason, record_origin,
        ),
    )
    conn.commit()


def record_weather_rejection(
    conn: sqlite3.Connection,
    *,
    evaluation_id: str | None,
    target_date: str,
    aggregation_key: str,
    candidate_initialized_at_utc: str,
    rejection_reason: str,
    details: str | None = None,
    record_origin: str = "live",
) -> None:
    """Append one rejected historical weather-run candidate."""
    conn.execute(
        f"""INSERT INTO {TABLE_WEATHER_RUN_REJECTIONS}
        (evaluation_id, target_date, aggregation_key, candidate_initialized_at_utc,
         rejection_reason, details, record_origin)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            evaluation_id, target_date, aggregation_key,
            candidate_initialized_at_utc, rejection_reason, details,
            record_origin,
        ),
    )
    conn.commit()


def lineage_summary(conn: sqlite3.Connection, evaluation_id: str) -> dict[str, float]:
    """Return MLflow-ready aggregate metrics for one evaluation."""
    rows = pd.read_sql_query(
        f"""SELECT fallback_offset_hours, availability_status
        FROM {TABLE_PRICE_INPUT_LINEAGE} WHERE evaluation_id=?""",
        conn, params=(evaluation_id,),
    )
    if rows.empty:
        return {
            "input_lineage_records": 0.0,
            "weather_fallback_targets": 0.0,
            "weather_max_fallback_hours": 0.0,
            "input_unavailable_records": 0.0,
        }
    weather_fallback = rows[rows["fallback_offset_hours"] > 0]
    return {
        "input_lineage_records": float(len(rows)),
        "weather_fallback_targets": float(len(weather_fallback)),
        "weather_max_fallback_hours": float(
            weather_fallback["fallback_offset_hours"].max()
            if not weather_fallback.empty else 0
        ),
        "input_unavailable_records": float(
            (rows["availability_status"] != "available").sum()
        ),
    }


def lineages_for_export(conn: sqlite3.Connection, evaluation_id: str) -> pd.DataFrame:
    """Return a stable, human-readable lineage extract for optional CSV export."""
    return pd.read_sql_query(
        f"""SELECT * FROM {TABLE_PRICE_INPUT_LINEAGE}
        WHERE evaluation_id=? ORDER BY target_date, input_group""",
        conn, params=(evaluation_id,),
    )
