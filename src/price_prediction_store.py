"""SQLite persistence for resumable price walk-forward predictions."""

from __future__ import annotations

import sqlite3

import pandas as pd


TABLE = "price_walk_forward_predictions"
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
    connection.commit()


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
