"""
Price ETL foundation: create normalized tables for multi-series time series data.

This module only handles DDL in step 2.
No fetching, no transformations yet.
"""

from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Sequence

import pandas as pd

try:
    # Preferred import path when used as a project package/module.
    from src.config import (
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
        SMARD_FILTER_WIND_ONSHORE_FC,
        SMARD_FILTER_WIND_OFFSHORE_FC,
        SMARD_FILTER_PV_FC,
        TABLE_SERIES_CATALOG,
        TABLE_TIMESERIES_VALUES,
        TABLE_INGESTION_RUNS,
        TABLE_DATA_QUALITY_LOG,
        TABLE_WEATHER_FORECAST_RUNS,
        TABLE_WEATHER_FORECAST_VALUES,
        TABLE_EXTERNAL_FORECAST_SNAPSHOTS,
        TABLE_PRICE_INPUT_LINEAGE,
        TABLE_WEATHER_RUN_REJECTIONS,
        PV_WEATHER_SERIES_IDS,
        WIND_WEATHER_SERIES_IDS,
        DEMAND_FORECAST_WEATHER_SERIES_IDS,
        PV_CLUSTER_LOCATIONS,
        WIND_CLUSTER_LOCATIONS,
        SELECTED_CITIES,
        CITY_POPULATION,
    )
except ImportError:
    # Backward-compatible fallback for direct script execution from src/.
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
        SMARD_FILTER_WIND_ONSHORE_FC,
        SMARD_FILTER_WIND_OFFSHORE_FC,
        SMARD_FILTER_PV_FC,
        TABLE_SERIES_CATALOG,
        TABLE_TIMESERIES_VALUES,
        TABLE_INGESTION_RUNS,
        TABLE_DATA_QUALITY_LOG,
        TABLE_WEATHER_FORECAST_RUNS,
        TABLE_WEATHER_FORECAST_VALUES,
        TABLE_EXTERNAL_FORECAST_SNAPSHOTS,
        TABLE_PRICE_INPUT_LINEAGE,
        TABLE_WEATHER_RUN_REJECTIONS,
        PV_WEATHER_SERIES_IDS,
        WIND_WEATHER_SERIES_IDS,
        DEMAND_FORECAST_WEATHER_SERIES_IDS,
        PV_CLUSTER_LOCATIONS,
        WIND_CLUSTER_LOCATIONS,
        SELECTED_CITIES,
        CITY_POPULATION,
    )

from util.smard_client import SmardClient
from util.openmeteo_client import OpenMeteoClient
from util.weather_weighted import build_yearly_weights

try:
    from src.config import (
        DATABASE_PATH,
        PV_CLUSTER_YEARLY_CAPACITY_PATH,
        WIND_CLUSTER_YEARLY_CAPACITY_PATH,
    )
except ImportError:
    from config import (
        DATABASE_PATH,
        PV_CLUSTER_YEARLY_CAPACITY_PATH,
        WIND_CLUSTER_YEARLY_CAPACITY_PATH,
    )

#PROJECT_ROOT = Path(__file__).parent.parent
#DB_DIR = PROJECT_ROOT / "db"
#DEFAULT_DB_PATH = DB_DIR / "energy_demand.db"
#PV_CLUSTER_YEARLY_CAPACITY_PATH = PROJECT_ROOT / "data" / "processed" / "pv_cluster_yearly_capacity_since_2019.csv"
#WIND_CLUSTER_YEARLY_CAPACITY_PATH = PROJECT_ROOT / "data" / "processed" / "wind_cluster_yearly_capacity_since_2019.csv"


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
    {
        "series_id": "forecast_wind_onshore_mwh",
        "source": "smard",
        "filter_id": SMARD_FILTER_WIND_ONSHORE_FC,
        "region": SMARD_REGION,
        "resolution": SMARD_RESOLUTION,
        "unit": "MWh",
        "active": 1,
        "description": "SMARD forecasted generation wind onshore",
    },
    {
        "series_id": "forecast_wind_offshore_mwh",
        "source": "smard",
        "filter_id": SMARD_FILTER_WIND_OFFSHORE_FC,
        "region": SMARD_REGION,
        "resolution": SMARD_RESOLUTION,
        "unit": "MWh",
        "active": 1,
        "description": "SMARD forecasted generation wind offshore",
    },
    {
        "series_id": "forecast_pv_mwh",
        "source": "smard",
        "filter_id": SMARD_FILTER_PV_FC,
        "region": SMARD_REGION,
        "resolution": SMARD_RESOLUTION,
        "unit": "MWh",
        "active": 1,
        "description": "SMARD forecasted generation photovoltaics",
    },
]


_OPENMETEO_UNITS = {
    "shortwave_radiation": "W/m^2",
    "direct_radiation": "W/m^2",
    "diffuse_radiation": "W/m^2",
    "cloud_cover": "%",
    "wind_speed_100m": "km/h",
    "wind_direction_100m": "deg",
}


for weather_var, series_id in PV_WEATHER_SERIES_IDS.items():
    SERIES_CATALOG_SEED.append(
        {
            "series_id": series_id,
            "source": "openmeteo",
            "filter_id": None,
            "region": SMARD_REGION,
            "resolution": SMARD_RESOLUTION,
            "unit": _OPENMETEO_UNITS[weather_var],
            "active": 1,
            "description": f"Open-Meteo PV weighted weather: {weather_var}",
        }
    )


for weather_var, series_id in WIND_WEATHER_SERIES_IDS.items():
    SERIES_CATALOG_SEED.append(
        {
            "series_id": series_id,
            "source": "openmeteo",
            "filter_id": None,
            "region": SMARD_REGION,
            "resolution": SMARD_RESOLUTION,
            "unit": _OPENMETEO_UNITS[weather_var],
            "active": 1,
            "description": f"Open-Meteo Wind weighted weather: {weather_var}",
        }
    )


for weather_var, series_id in DEMAND_FORECAST_WEATHER_SERIES_IDS.items():
    SERIES_CATALOG_SEED.append(
        {
            "series_id": series_id,
            "source": "openmeteo_forecast",
            "filter_id": None,
            "region": SMARD_REGION,
            "resolution": SMARD_RESOLUTION,
            "unit": _OPENMETEO_UNITS.get(weather_var, "unknown"),
            "active": 1,
            "description": f"Open-Meteo archived demand forecast weather: {weather_var}",
        }
    )


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


def _ddl_weather_forecast_runs() -> str:
    """Model-run metadata for point-in-time weather forecasts.

    ``available_at_utc`` is intentionally distinct from initialization: a
    numerical weather model needs time to finish and be published.
    """
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_WEATHER_FORECAST_RUNS} (
    forecast_run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    provider              TEXT NOT NULL,
    model                 TEXT NOT NULL,
    initialized_at_utc    TEXT NOT NULL,
    available_at_utc      TEXT NOT NULL,
    aggregation_key       TEXT NOT NULL,
    aggregation_version   TEXT NOT NULL,
    source_url            TEXT,
    fetched_at_utc        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(provider, model, initialized_at_utc, aggregation_key, aggregation_version)
)
"""


def _ddl_weather_forecast_values() -> str:
    """Weather features of one model run, indexed by their valid time."""
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_WEATHER_FORECAST_VALUES} (
    forecast_run_id       INTEGER NOT NULL,
    valid_time_utc        TEXT NOT NULL,
    series_id             TEXT NOT NULL,
    value                 REAL,
    PRIMARY KEY (forecast_run_id, valid_time_utc, series_id),
    FOREIGN KEY (forecast_run_id)
        REFERENCES {TABLE_WEATHER_FORECAST_RUNS}(forecast_run_id)
        ON DELETE CASCADE,
    FOREIGN KEY (series_id) REFERENCES {TABLE_SERIES_CATALOG}(series_id)
)
"""


def _ddl_external_forecast_snapshots() -> str:
    """Point-in-time snapshots of externally published forecasts (e.g. SMARD)."""
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_EXTERNAL_FORECAST_SNAPSHOTS} (
    source                TEXT NOT NULL,
    series_id             TEXT NOT NULL,
    issued_at_utc         TEXT NOT NULL,
    valid_time_utc        TEXT NOT NULL,
    value                 REAL,
    retrieved_at_utc      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (source, series_id, issued_at_utc, valid_time_utc),
    FOREIGN KEY (series_id) REFERENCES {TABLE_SERIES_CATALOG}(series_id)
)
"""


def _ddl_price_input_lineage() -> str:
    """Audit trail for the concrete inputs selected for one price target."""
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_PRICE_INPUT_LINEAGE} (
    evaluation_id              TEXT NOT NULL,
    target_date                TEXT NOT NULL,
    input_group                TEXT NOT NULL,
    source                     TEXT NOT NULL,
    selected_forecast_run_id   INTEGER,
    preferred_initialized_at_utc TEXT,
    fallback_offset_hours      INTEGER NOT NULL DEFAULT 0,
    availability_status        TEXT NOT NULL,
    fallback_type              TEXT,
    selection_reason           TEXT,
    record_origin              TEXT NOT NULL DEFAULT 'live',
    recorded_at_utc            TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (evaluation_id, target_date, input_group),
    FOREIGN KEY (selected_forecast_run_id)
        REFERENCES {TABLE_WEATHER_FORECAST_RUNS}(forecast_run_id)
)
"""


def _ddl_weather_run_rejections() -> str:
    """Candidates rejected while choosing a historical weather forecast run."""
    return f"""
CREATE TABLE IF NOT EXISTS {TABLE_WEATHER_RUN_REJECTIONS} (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id              TEXT,
    target_date                TEXT NOT NULL,
    aggregation_key            TEXT NOT NULL,
    candidate_initialized_at_utc TEXT NOT NULL,
    rejection_reason           TEXT NOT NULL,
    details                    TEXT,
    record_origin              TEXT NOT NULL DEFAULT 'live',
    recorded_at_utc            TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def create_price_tables(db_path: Path | str = DATABASE_PATH) -> sqlite3.Connection:
    """Create all price-pipeline tables and return an open DB connection."""
    if str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    cur.execute(_ddl_series_catalog())
    cur.execute(_ddl_timeseries_values())
    cur.execute(_ddl_ingestion_runs())
    cur.execute(_ddl_data_quality_log())
    cur.execute(_ddl_weather_forecast_runs())
    cur.execute(_ddl_weather_forecast_values())
    cur.execute(_ddl_external_forecast_snapshots())
    cur.execute(_ddl_price_input_lineage())
    cur.execute(_ddl_weather_run_rejections())

    # Helpful indexes for future fetch/query steps.
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_TIMESERIES_VALUES}_series_time "
        f"ON {TABLE_TIMESERIES_VALUES}(series_id, time)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_TIMESERIES_VALUES}_time "
        f"ON {TABLE_TIMESERIES_VALUES}(time)"
    )

    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_WEATHER_FORECAST_VALUES}_series_valid "
        f"ON {TABLE_WEATHER_FORECAST_VALUES}(series_id, valid_time_utc)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_WEATHER_FORECAST_RUNS}_available "
        f"ON {TABLE_WEATHER_FORECAST_RUNS}(available_at_utc)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_EXTERNAL_FORECAST_SNAPSHOTS}_series_valid_issued "
        f"ON {TABLE_EXTERNAL_FORECAST_SNAPSHOTS}(series_id, valid_time_utc, issued_at_utc)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_PRICE_INPUT_LINEAGE}_target "
        f"ON {TABLE_PRICE_INPUT_LINEAGE}(target_date, input_group)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_WEATHER_RUN_REJECTIONS}_target "
        f"ON {TABLE_WEATHER_RUN_REJECTIONS}(target_date, aggregation_key)"
    )
    cur.execute(f"""
        CREATE VIEW IF NOT EXISTS v_price_walk_forward_input_audit AS
        SELECT lineage.*, weather.initialized_at_utc AS selected_initialized_at_utc,
               weather.available_at_utc AS selected_available_at_utc
        FROM {TABLE_PRICE_INPUT_LINEAGE} AS lineage
        LEFT JOIN {TABLE_WEATHER_FORECAST_RUNS} AS weather
          ON weather.forecast_run_id = lineage.selected_forecast_run_id
    """)

    conn.commit()
    return conn


def store_weather_forecast_run(
    conn: sqlite3.Connection,
    forecast: pd.DataFrame,
    *,
    provider: str,
    model: str,
    initialized_at_utc: str,
    available_at_utc: str,
    aggregation_key: str,
    aggregation_version: str,
    series_ids: dict[str, str],
    source_url: str | None = None,
) -> int:
    """Store one aggregated weather run idempotently and return its identifier."""
    required = {"time", *series_ids}
    if missing := required - set(forecast.columns):
        raise ValueError(f"forecast lacks columns: {sorted(missing)}")
    cur = conn.cursor()
    cur.execute(
        f"""INSERT INTO {TABLE_WEATHER_FORECAST_RUNS}
        (provider, model, initialized_at_utc, available_at_utc, aggregation_key,
         aggregation_version, source_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, model, initialized_at_utc, aggregation_key, aggregation_version)
        DO UPDATE SET available_at_utc=excluded.available_at_utc, source_url=excluded.source_url
        """,
        (provider, model, initialized_at_utc, available_at_utc, aggregation_key, aggregation_version, source_url),
    )
    cur.execute(
        f"""SELECT forecast_run_id FROM {TABLE_WEATHER_FORECAST_RUNS}
        WHERE provider=? AND model=? AND initialized_at_utc=? AND aggregation_key=? AND aggregation_version=?""",
        (provider, model, initialized_at_utc, aggregation_key, aggregation_version),
    )
    run_id = int(cur.fetchone()[0])
    rows = []
    for row in forecast[["time", *series_ids]].itertuples(index=False):
        valid_time = pd.Timestamp(row[0]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        for index, series_id in enumerate(series_ids.values(), start=1):
            rows.append((run_id, valid_time, series_id, float(row[index])))
    cur.executemany(
        f"""INSERT OR REPLACE INTO {TABLE_WEATHER_FORECAST_VALUES}
        (forecast_run_id, valid_time_utc, series_id, value) VALUES (?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return run_id


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


def _resolve_weather_series(series_id: str) -> tuple[str, str]:
    """Return (technology, weather_variable) for a weather series_id."""
    for var, sid in PV_WEATHER_SERIES_IDS.items():
        if sid == series_id:
            return "pv", var
    for var, sid in WIND_WEATHER_SERIES_IDS.items():
        if sid == series_id:
            return "wind", var
    raise ValueError(f"series_id is not a configured weather series: {series_id}")


def _load_yearly_cluster_weights(csv_path: Path, technology: str) -> dict[int, dict[str, float]]:
    """Load normalized per-year cluster capacity weights from CSV export."""
    return build_yearly_weights(capacity_csv=csv_path, technology_prefix=technology)


def _fetch_weighted_weather_timeseries(
    technology: str,
    weather_variable: str,
    start_date: str,
    end_date: str,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """Fetch one weighted weather timeseries using yearly cluster capacity weights."""
    if technology == "pv":
        locations = PV_CLUSTER_LOCATIONS
        weights_by_year = _load_yearly_cluster_weights(PV_CLUSTER_YEARLY_CAPACITY_PATH, technology="pv")
    elif technology == "wind":
        locations = WIND_CLUSTER_LOCATIONS
        weights_by_year = _load_yearly_cluster_weights(WIND_CLUSTER_YEARLY_CAPACITY_PATH, technology="wind")
    else:
        raise ValueError(f"unsupported technology: {technology}")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date must be >= start_date")

    # OpenMeteoClient requires baseline cities/population in constructor.
    # For weighted cluster fetch, only the generic weighted-location method is used.
    client = OpenMeteoClient(
        cities=SELECTED_CITIES,
        city_population=CITY_POPULATION,
        weather_variables=[weather_variable],
        city_sleep=sleep,
    )

    frames: list[pd.DataFrame] = []
    for year in range(start_ts.year, end_ts.year + 1):
        if year not in weights_by_year:
            continue
        chunk_start = max(start_ts, pd.Timestamp(f"{year}-01-01"))
        chunk_end = min(end_ts, pd.Timestamp(f"{year}-12-31"))
        if chunk_end < chunk_start:
            continue

        df_chunk = client.fetch_archive_weighted_locations(
            locations=locations,
            location_weights=weights_by_year[year],
            start_date=chunk_start.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
            weather_variables=[weather_variable],
        )
        frames.append(df_chunk[["time", weather_variable]])

    if not frames:
        return pd.DataFrame(columns=["time", "value"])

    df_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    df_all["value"] = pd.to_numeric(df_all[weather_variable], errors="coerce")
    df_all = df_all.dropna(subset=["value"])
    return df_all[["time", "value"]].reset_index(drop=True)


def fetch_and_store_openmeteo_series(
    conn: sqlite3.Connection,
    series_id: str,
    start_date: str,
    end_date: str,
    sleep: float = 0.3,
    version: int = 1,
) -> int:
    """
    Fetch one configured Open-Meteo weather series and store into timeseries_values.

    Returns inserted row count (idempotent: duplicates are ignored).
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT source, active
        FROM {TABLE_SERIES_CATALOG}
        WHERE series_id = ?
        """,
        (series_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"series_id not found in {TABLE_SERIES_CATALOG}: {series_id}")

    source, active = row
    if int(active) != 1:
        raise ValueError(f"series_id is inactive: {series_id}")
    if source != "openmeteo":
        raise ValueError(f"series_id source is not 'openmeteo': {series_id}")

    technology, weather_variable = _resolve_weather_series(series_id)
    df = _fetch_weighted_weather_timeseries(
        technology=technology,
        weather_variable=weather_variable,
        start_date=start_date,
        end_date=end_date,
        sleep=sleep,
    )
    if df.empty:
        return 0

    df = df.copy()
    df["time"] = (
        pd.to_datetime(df["time"], utc=True)
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    rows = [
        (t, series_id, float(v), "openmeteo_archive", int(version))
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


def fetch_and_store_openmeteo_batch(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    series_ids: list[str] | None = None,
    sleep: float = 0.3,
    version: int = 1,
    continue_on_error: bool = False,
) -> dict:
    """
    Fetch and store multiple Open-Meteo weather series with ingestion run logging.

    Returns dict with run_id, status, total_inserted, per_series, errors.
    """
    if series_ids is None:
        series_ids = list(PV_WEATHER_SERIES_IDS.values()) + list(WIND_WEATHER_SERIES_IDS.values())

    run_id = _start_ingestion_run(conn, source="openmeteo")
    per_series: dict[str, int] = {}
    errors: dict[str, str] = {}
    total_inserted = 0

    try:
        for series_id in series_ids:
            try:
                inserted = fetch_and_store_openmeteo_series(
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


def check_price_data_status(conn: sqlite3.Connection, source: str | None = None) -> dict:
    """
    Query timeseries_values for max timestamp and row count per series.
    Returns dict: {series_id: {max_time, rows}}
    """
    cur = conn.cursor()
    if source is None:
        cur.execute(f"SELECT series_id FROM {TABLE_SERIES_CATALOG} WHERE active=1")
    else:
        cur.execute(
            f"SELECT series_id FROM {TABLE_SERIES_CATALOG} WHERE active=1 AND source=?",
            (source,),
        )
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


def create_lag_rolling_features(
    in_df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int] = (24, 168),
    rolling_windows: Sequence[int] = (24, 168),
    rolling_shift: int = 1,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Create lag and rolling-mean features for multiple numeric columns.

    Feature naming:
    - ``{column}_lag_{N}h``
    - ``{column}_rolling_mean_{N}h``

    Rolling features are computed on ``shift(rolling_shift)`` to avoid leakage
    from the current timestamp (default behavior uses only past observations).
    """
    if not columns:
        raise ValueError("columns must contain at least one column name")
    if rolling_shift < 0:
        raise ValueError("rolling_shift must be >= 0")

    missing_cols = [c for c in columns if c not in in_df.columns]
    if missing_cols:
        raise ValueError(f"missing columns in input DataFrame: {missing_cols}")

    bad_lags = [x for x in lags if int(x) <= 0]
    bad_windows = [x for x in rolling_windows if int(x) <= 0]
    if bad_lags:
        raise ValueError(f"all lags must be positive integers, got: {bad_lags}")
    if bad_windows:
        raise ValueError(f"all rolling_windows must be positive integers, got: {bad_windows}")

    out_df = in_df.copy()
    created_cols: list[str] = []

    for col in columns:
        series = pd.to_numeric(out_df[col], errors="coerce")

        for lag in lags:
            lag = int(lag)
            lag_col = f"{col}_lag_{lag}h"
            out_df[lag_col] = series.shift(lag)
            created_cols.append(lag_col)

        for window in rolling_windows:
            window = int(window)
            roll_col = f"{col}_rolling_mean_{window}h"
            out_df[roll_col] = series.shift(rolling_shift).rolling(window=window).mean()
            created_cols.append(roll_col)

    if dropna and created_cols:
        out_df = out_df.dropna(subset=created_cols)

    return out_df

def update_price_database(db_path: Path = DATABASE_PATH, start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
    """
    Orchestrator for price ETL: ensures tables, seeds catalog, fetches and stores all active series.
    """
    print("\n" + "-"*10 + " Updating price database " + "-"*10)
    conn = create_price_tables(db_path)
    try:
        changed = seed_series_catalog(conn)
        #print(f"Series catalog seeded/updated rows: {changed}")
        smard_status = check_price_data_status(conn, source="smard")
        #print("\nCurrent SMARD data status:")
        #for sid, s in smard_status.items():
        #    print(f"  {sid:28s}: {s['rows']:>6} rows | max: {s['max_time']}")

        # SMARD up-to-date check.
        # For price and generation, we often want the day-ahead values if available today.
        today = datetime.utcnow().date()
        tomorrow = today + timedelta(days=1)
        
        # Series that should have data for tomorrow (forecasts)
        forecast_series = [
            "forecast_wind_onshore_mwh", 
            "forecast_wind_offshore_mwh", 
            "forecast_pv_mwh",
            "price_de_lu_eur_mwh" # Day-ahead price is also a "forecast" for tomorrow usually published around 13:00
        ]

        smard_all_up_to_date = True
        for sid, s in smard_status.items():
            if not s["max_time"]:
                smard_all_up_to_date = False
                break
            max_time = datetime.strptime(s["max_time"], "%Y-%m-%dT%H:%M:%SZ").date()
            
            # If it's a forecast series, it's up to date if it has data for tomorrow
            if sid in forecast_series:
                 if max_time < tomorrow:
                     smard_all_up_to_date = False
                     break
            else:
                # Actual values are up to date if they have data for today
                if max_time < today:
                    smard_all_up_to_date = False
                    break

        if smard_all_up_to_date:
            print("\nSMARD series up to date — skip SMARD fetch.")
        else:
            # Series to fetch
            active_smard_series = [sid for sid, s in smard_status.items()]
            
            # Batch process them. 
            # We don't want to backfill forecasts for years if they are missing.
            # Forecasts are mostly useful for the "head" (today/tomorrow).
            
            smard_start = start_date
            smard_end = end_date
            
            if smard_start is None:
                # Calculate required start for each series type
                series_starts = {}
                for sid in active_smard_series:
                    s = smard_status[sid]
                    if s["rows"] == 0:
                        if sid in forecast_series:
                            # For forecasts, 30 days lookback is enough for any recent feature gaps
                            series_starts[sid] = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
                        else:
                            # For actuals/prices, fetch from the beginning
                            series_starts[sid] = "2019-01-01"
                    else:
                        series_starts[sid] = (datetime.strptime(s["max_time"], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=1)).strftime("%Y-%m-%d")
                
                # To keep using the batch fetcher, we take the minimum start date
                # but we filter out far-future starts
                smard_start = min(series_starts.values())
            
            if smard_end is None:
                smard_end = tomorrow.strftime("%Y-%m-%d")

            if smard_start <= smard_end:
                result = fetch_and_store_smard_batch(conn, smard_start, smard_end)
                #print(f"Batch ingestion result: {result}")
                print(f"\nDone with fetching and storing SMARD price/generation series: {smard_start} -> {smard_end}")
            else:
                print("\nSMARD start_date is after end_date — skip SMARD fetch.")

        weather_status = check_price_data_status(conn, source="openmeteo")
        #print("\nCurrent Open-Meteo data status:")
        #for sid, s in weather_status.items():
        #    print(f"  {sid:28s}: {s['rows']:>6} rows | max: {s['max_time']}")

        weather_target_end = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        weather_start = start_date
        weather_end = end_date if end_date is not None else weather_target_end
        if weather_start is None:
            if any(s["rows"] == 0 for s in weather_status.values()):
                weather_start = "2019-01-01"
            else:
                latest_weather = max(
                    [s["max_time"] for s in weather_status.values() if s["max_time"]],
                    default=None,
                )
                if latest_weather:
                    from dateutil import parser
                    dtw = parser.isoparse(latest_weather)
                    # Open-Meteo archive requests are date-based. If the latest UTC hour
                    # corresponds to a complete local day end (typically 21:00/22:00 UTC
                    # depending on DST), continue from next calendar day to avoid refetch.
                    if dtw.hour >= 21:
                        weather_start = (dtw + timedelta(days=1)).strftime("%Y-%m-%d")
                    else:
                        weather_start = dtw.strftime("%Y-%m-%d")
                else:
                    weather_start = "2019-01-01"

        if weather_start <= weather_end:
            weather_result = fetch_and_store_openmeteo_batch(conn, weather_start, weather_end)
            #print(f"Weather ingestion result: {weather_result}")
            print(f"\nDone with fetching and storing Open-Meteo weighted weather series: {weather_start} -> {weather_end}")
        else:
            print("\nOpen-Meteo weather is up to date (target: yesterday).")

        final = check_price_data_status(conn)
        print("\n" + "-"*10 + " Updating price database done " + "-"*10)
        #for sid, s in final.items():
        #    print(f"  {sid:25s}: {s['rows']:>6} rows | max: {s['max_time']}")
    finally:
        conn.close()
