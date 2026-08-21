"""SQLite persistence for resumable price walk-forward predictions."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd


TABLE = "price_walk_forward_predictions"
RUN_TABLE = "price_walk_forward_runs"
VERSIONED_TABLE = "price_walk_forward_prediction_versions"
COLUMNS = [
    "target_time", "as_of_time", "model_name", "evaluation_mode",
    "prediction_eur_mwh", "actual_eur_mwh", "created_at",
]


def ensure_price_prediction_table(connection: sqlite3.Connection) -> None:
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            target_time TEXT NOT NULL,
            as_of_time TEXT NOT NULL,
            model_name TEXT NOT NULL,
            evaluation_mode TEXT NOT NULL,
            prediction_eur_mwh REAL NOT NULL,
            actual_eur_mwh REAL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (target_time, model_name, evaluation_mode)
        )
    """)
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            run_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_sha256 TEXT NOT NULL,
            demand_model_sha256 TEXT NOT NULL,
            feature_pipeline_version TEXT NOT NULL,
            protocol_version TEXT NOT NULL,
            evaluation_mode TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {VERSIONED_TABLE} (
            run_id TEXT NOT NULL,
            target_time TEXT NOT NULL,
            as_of_time TEXT NOT NULL,
            prediction_eur_mwh REAL NOT NULL,
            actual_eur_mwh REAL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, target_time),
            FOREIGN KEY (run_id) REFERENCES {RUN_TABLE}(run_id)
        )
    """)
    connection.commit()


def create_prediction_run(
    connection: sqlite3.Connection, *, model_name: str, model_sha256: str,
    demand_model_sha256: str, feature_pipeline_version: str,
    protocol_version: str, evaluation_mode: str,
) -> str:
    """Create an immutable lineage record for one reproducible model run."""
    ensure_price_prediction_table(connection)
    run_id = str(uuid4())
    connection.execute(
        f"""INSERT INTO {RUN_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, model_name, model_sha256, demand_model_sha256,
         feature_pipeline_version, protocol_version, evaluation_mode,
         datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    return run_id


def store_versioned_price_predictions(
    connection: sqlite3.Connection, run_id: str, predictions: pd.DataFrame,
) -> int:
    """Persist predictions without overwriting outcomes from another model run."""
    missing = [column for column in COLUMNS if column not in predictions]
    if missing:
        raise ValueError(f"Missing price walk-forward columns: {missing}")
    if predictions.empty:
        return 0
    if connection.execute(f"SELECT 1 FROM {RUN_TABLE} WHERE run_id=?", (run_id,)).fetchone() is None:
        raise ValueError(f"Unknown prediction run: {run_id}")
    rows = predictions.loc[:, ["target_time", "as_of_time", "prediction_eur_mwh", "actual_eur_mwh", "created_at"]].copy()
    for column in ("target_time", "as_of_time"):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    connection.executemany(
        f"""INSERT INTO {VERSIONED_TABLE}
        (run_id, target_time, as_of_time, prediction_eur_mwh, actual_eur_mwh, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, target_time) DO UPDATE SET
            as_of_time=excluded.as_of_time, prediction_eur_mwh=excluded.prediction_eur_mwh,
            actual_eur_mwh=excluded.actual_eur_mwh, created_at=excluded.created_at""",
        ((run_id, *row) for row in rows.itertuples(index=False, name=None)),
    )
    connection.commit()
    return len(rows)


def upsert_price_predictions(connection: sqlite3.Connection, predictions: pd.DataFrame) -> int:
    missing = [column for column in COLUMNS if column not in predictions]
    if missing:
        raise ValueError(f"Missing price walk-forward columns: {missing}")
    if predictions.empty:
        return 0
    rows = predictions.loc[:, COLUMNS].copy()
    for column in ("target_time", "as_of_time"):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="raise").dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    connection.executemany(
        f"""INSERT INTO {TABLE} ({', '.join(COLUMNS)})
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(target_time, model_name, evaluation_mode) DO UPDATE SET
            as_of_time=excluded.as_of_time,
            prediction_eur_mwh=excluded.prediction_eur_mwh,
            actual_eur_mwh=excluded.actual_eur_mwh,
            created_at=excluded.created_at
        """,
        rows.itertuples(index=False, name=None),
    )
    connection.commit()
    return len(rows)
