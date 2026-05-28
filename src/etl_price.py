"""
Price ETL foundation: create normalized tables for multi-series time series data.

This module only handles DDL in step 2.
No fetching, no transformations yet.
"""

from pathlib import Path
import sqlite3
from typing import Optional

import pandas as pd

from config import (
    SMARD_BASE,
    SMARD_HEADERS,
    SMARD_REGION,
    SMARD_RESOLUTION,
    SMARD_PRICE_START_FILTERS,
    SMARD_REGION_PRICE_DE_LU,
    SMARD_FILTER_PRICE_DE_LU,
    SMARD_FILTER_WIND_ONSHORE,
    SMARD_FILTER_WIND_OFFSHORE,
    SMARD_FILTER_PV,
    SMARD_FILTER_OTHER_CONVENTIONAL,
    TABLE_SERIES_CATALOG,
    TABLE_TIMESERIES_VALUES,
    TABLE_INGESTION_RUNS,
    TABLE_DATA_QUALITY_LOG,
)
from util.smard_client import SmardClient


ROOT_DIR = Path(__file__).parent.parent
DB_DIR = ROOT_DIR / "db"
DEFAULT_DB_PATH = DB_DIR / "energy_demand.db"


SERIES_CATALOG_SEED = [
    {
        "series_id": "price_de_lu_eur_mwh",
        "source": "smard",
        "filter_id": SMARD_FILTER_PRICE_DE_LU,
        "region": SMARD_REGION_PRICE_DE_LU,
        "resolution": SMARD_RESOLUTION,
        "unit": "EUR/MWh",
        "active": 1,
        "description": "SMARD day-ahead market price DE/LU",
    },
    {
        "series_id": "gen_wind_onshore_mwh",
        "source": "smard",
        "filter_id": SMARD_FILTER_WIND_ONSHORE,
        "region": SMARD_REGION,
        "resolution": SMARD_RESOLUTION,
        "unit": "MWh",
        "active": 1,
        "description": "SMARD realized generation wind onshore",
    },
    {
        "series_id": "gen_wind_offshore_mwh",
        "source": "smard",
        "filter_id": SMARD_FILTER_WIND_OFFSHORE,
        "region": SMARD_REGION,
        "resolution": SMARD_RESOLUTION,
        "unit": "MWh",
        "active": 1,
        "description": "SMARD realized generation wind offshore",
    },
    {
        "series_id": "gen_pv_mwh",
        "source": "smard",
        "filter_id": SMARD_FILTER_PV,
        "region": SMARD_REGION,
        "resolution": SMARD_RESOLUTION,
        "unit": "MWh",
        "active": 1,
        "description": "SMARD realized generation photovoltaics",
    },
    {
        "series_id": "gen_other_conventional_mwh",
        "source": "smard",
        "filter_id": SMARD_FILTER_OTHER_CONVENTIONAL,
        "region": SMARD_REGION,
        "resolution": SMARD_RESOLUTION,
        "unit": "MWh",
        "active": 1,
        "description": "SMARD realized generation other conventional",
    },
]


def _ddl_series_catalog() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_SERIES_CATALOG} (
    series_id    TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    filter_id    INTEGER,
    region       TEXT,
    resolution   TEXT,
    unit         TEXT,
    active       INTEGER NOT NULL DEFAULT 1,
    description  TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT
)
"""


def _ddl_timeseries_values() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_TIMESERIES_VALUES} (
    time         TEXT NOT NULL,
    series_id    TEXT NOT NULL,
    value        REAL,
    data_source  TEXT,
    fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
    version      INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (time, series_id, version),
    FOREIGN KEY (series_id) REFERENCES {TABLE_SERIES_CATALOG}(series_id)
)
"""


def _ddl_ingestion_runs() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_INGESTION_RUNS} (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts      TEXT NOT NULL DEFAULT (datetime('now')),
    end_ts        TEXT,
    status        TEXT NOT NULL,
    source        TEXT,
    rows_loaded   INTEGER NOT NULL DEFAULT 0,
    error_text    TEXT
)
"""


def _ddl_data_quality_log() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_DATA_QUALITY_LOG} (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER,
    series_id    TEXT,
    check_name   TEXT NOT NULL,
    result       TEXT NOT NULL,
    details      TEXT,
    checked_at   TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES {TABLE_INGESTION_RUNS}(run_id),
    FOREIGN KEY (series_id) REFERENCES {TABLE_SERIES_CATALOG}(series_id)
)
"""


def create_price_tables(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create all price-pipeline tables and return an open DB connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(_ddl_series_catalog())
    cur.execute(_ddl_timeseries_values())
    cur.execute(_ddl_ingestion_runs())
    cur.execute(_ddl_data_quality_log())

    # Helpful indexes for future fetch/query steps.
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_TIMESERIES_VALUES}_series_time "
        f"ON {TABLE_TIMESERIES_VALUES}(series_id, time)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_TIMESERIES_VALUES}_time "
        f"ON {TABLE_TIMESERIES_VALUES}(time)"
    )

    conn.commit()
    return conn


def seed_series_catalog(conn: sqlite3.Connection) -> int:
    """Insert or update the initial series catalog entries. Returns affected row count."""
    cur = conn.cursor()
    upsert_sql = f"""
    INSERT INTO {TABLE_SERIES_CATALOG}
        (series_id, source, filter_id, region, resolution, unit, active, description, updated_at)
    VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    ON CONFLICT(series_id) DO UPDATE SET
        source = excluded.source,
        filter_id = excluded.filter_id,
        region = excluded.region,
        resolution = excluded.resolution,
        unit = excluded.unit,
        active = excluded.active,
        description = excluded.description,
        updated_at = datetime('now')
    """

    affected = 0
    for row in SERIES_CATALOG_SEED:
        cur.execute(
            upsert_sql,
            (
                row["series_id"],
                row["source"],
                row["filter_id"],
                row["region"],
                row["resolution"],
                row["unit"],
                row["active"],
                row["description"],
            ),
        )
        affected += cur.rowcount

    conn.commit()
    return affected


def fetch_and_store_smard_series(
    conn: sqlite3.Connection,
    series_id: str,
    start_date: str,
    end_date: str,
    sleep: float = 0.3,
    version: int = 1,
) -> int:
    """
    Fetch one SMARD series by series_id from catalog and store into timeseries_values.

    Returns inserted row count (idempotent: duplicates are ignored).
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT filter_id, region, resolution, source, active
        FROM {TABLE_SERIES_CATALOG}
        WHERE series_id = ?
        """,
        (series_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"series_id not found in {TABLE_SERIES_CATALOG}: {series_id}")

    filter_id, region, resolution, source, active = row
    if int(active) != 1:
        raise ValueError(f"series_id is inactive: {series_id}")
    if source != "smard":
        raise ValueError(f"series_id source is not 'smard': {series_id}")

    client = SmardClient(
        filter_id=int(filter_id),
        region=str(region),
        base_url=SMARD_BASE,
        headers=SMARD_HEADERS,
        resolution=str(resolution),
        sleep=sleep,
    )
    df = client.fetch(start_date, end_date)
    if df.empty:
        return 0

    df = df.copy()
    df["time"] = (
        pd.to_datetime(df["time"], utc=True)
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    df["value"] = pd.to_numeric(df["load_MWh"], errors="coerce")
    df = df.dropna(subset=["value"])

    rows = [
        (t, series_id, float(v), "smard_api", int(version))
        for t, v in zip(df["time"], df["value"])
    ]

    before = conn.total_changes
    cur.executemany(
        f"""
        INSERT OR IGNORE INTO {TABLE_TIMESERIES_VALUES}
            (time, series_id, value, data_source, version)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def _start_ingestion_run(conn: sqlite3.Connection, source: str = "smard") -> int:
    """Create a running ingestion run and return run_id."""
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT INTO {TABLE_INGESTION_RUNS} (status, source)
        VALUES ('running', ?)
        """,
        (source,),
    )
    run_id = cur.lastrowid
    conn.commit()
    if run_id is None:
        raise RuntimeError("Failed to create ingestion run row")
    return int(run_id)


def _finish_ingestion_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    rows_loaded: int,
    error_text: str | None = None,
) -> None:
    """Finalize ingestion run with status, row count and optional error."""
    cur = conn.cursor()
    cur.execute(
        f"""
        UPDATE {TABLE_INGESTION_RUNS}
        SET end_ts = datetime('now'),
            status = ?,
            rows_loaded = ?,
            error_text = ?
        WHERE run_id = ?
        """,
        (status, int(rows_loaded), error_text, int(run_id)),
    )
    conn.commit()


def fetch_and_store_smard_batch(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    series_ids: list[str] | None = None,
    sleep: float = 0.3,
    version: int = 1,
    continue_on_error: bool = False,
) -> dict:
    """
    Fetch and store multiple SMARD series with ingestion run logging.

    Returns dict with run_id, status, total_inserted, per_series, errors.
    """
    if series_ids is None:
        series_ids = list(SMARD_PRICE_START_FILTERS.keys())

    run_id = _start_ingestion_run(conn, source="smard")
    per_series: dict[str, int] = {}
    errors: dict[str, str] = {}
    total_inserted = 0

    try:
        for series_id in series_ids:
            try:
                inserted = fetch_and_store_smard_series(
                    conn=conn,
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                    sleep=sleep,
                    version=version,
                )
                per_series[series_id] = inserted
                total_inserted += inserted
            except Exception as exc:
                errors[series_id] = str(exc)
                if not continue_on_error:
                    raise

        status = "success" if not errors else "partial_success"
        _finish_ingestion_run(
            conn=conn,
            run_id=run_id,
            status=status,
            rows_loaded=total_inserted,
            error_text=(None if not errors else str(errors)),
        )
        return {
            "run_id": run_id,
            "status": status,
            "total_inserted": total_inserted,
            "per_series": per_series,
            "errors": errors,
        }
    except Exception as exc:
        _finish_ingestion_run(
            conn=conn,
            run_id=run_id,
            status="failed",
            rows_loaded=total_inserted,
            error_text=str(exc),
        )
        raise


def check_price_data_status(conn: sqlite3.Connection) -> dict:
    """
    Query timeseries_values for max timestamp and row count per series.
    Returns dict: {series_id: {max_time, rows}}
    """
    cur = conn.cursor()
    cur.execute(f"SELECT series_id FROM {TABLE_SERIES_CATALOG} WHERE active=1")
    series_ids = [row[0] for row in cur.fetchall()]
    status = {}
    for sid in series_ids:
        cur.execute(
            f"SELECT MAX(time), COUNT(*) FROM {TABLE_TIMESERIES_VALUES} WHERE series_id=?",
            (sid,)
        )
        max_time, rows = cur.fetchone()
        status[sid] = {"max_time": max_time, "rows": rows}
    return status

from datetime import datetime, timedelta

def update_database(db_path: Path = DEFAULT_DB_PATH, start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
    """
    Orchestrator for price ETL: ensures tables, seeds catalog, fetches and stores all active series.
    """
    conn = create_price_tables(db_path)
    try:
        changed = seed_series_catalog(conn)
        print(f"Series catalog seeded/updated rows: {changed}")
        status = check_price_data_status(conn)
        print("\nCurrent price data status:")
        for sid, s in status.items():
            print(f"  {sid:28s}: {s['rows']:>6} rows | max: {s['max_time']}")

        # Check if all active series are up to date (max_time >= today). If so, skip fetching.
        today = datetime.utcnow().date()
        all_up_to_date = True
        for s in status.values():
            if not s["max_time"]:
                all_up_to_date = False
                break
            max_time = datetime.strptime(s["max_time"], "%Y-%m-%dT%H:%M:%SZ").date()
            if max_time < today:
                all_up_to_date = False
                break

        if all_up_to_date:
            print("\nAll series up to date — nothing to do.")
            return

        # Default: fetch full range if empty, else fetch only missing tail
        #today = datetime.utcnow().date()
        if start_date is None:
            # If any series is empty, fetch from 2019-01-01
            if any(s["rows"] == 0 for s in status.values()):
                start_date = "2019-01-01"
            else:
                # Fetch from latest max_time + 1h (assume hourly)
                latest = max([s["max_time"] for s in status.values() if s["max_time"]], default=None)
                if latest:
                    from dateutil import parser
                    dt = parser.isoparse(latest)
                    start_date = (dt + timedelta(hours=1)).strftime("%Y-%m-%d")
                else:
                    start_date = "2019-01-01"
        if end_date is None:
            end_date = today.strftime("%Y-%m-%d")

        print(f"\nFetching and storing SMARD price/generation series: {start_date} → {end_date}")
        result = fetch_and_store_smard_batch(conn, start_date, end_date)
        print(f"Batch ingestion result: {result}")

        final = check_price_data_status(conn)
        print("\nDone.")
        for sid, s in final.items():
            print(f"  {sid:25s}: {s['rows']:>6} rows | max: {s['max_time']}")
    finally:
        conn.close()
