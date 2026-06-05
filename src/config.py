"""
config.py — project-specific configuration for the Germany electricity demand project.

All domain constants live here so that:
- util/ classes remain fully generic (no hardcoded country/city defaults)
- etl_demand.py and fetch_demand_data.py import from one place instead of each defining their own
"""

import pandas as pd

# ---------------------------------------------------------------------------
# SMARD API
# ---------------------------------------------------------------------------
SMARD_BASE              = "https://www.smard.de/app/chart_data"
SMARD_HEADERS           = {"User-Agent": "Mozilla/5.0 (compatible; smard-fetcher/1.0)"}
SMARD_REGION            = "DE"
SMARD_RESOLUTION        = "hour"
SMARD_FILTER_NETZLAST   = 410   # Realisierter Stromverbrauch – Netzlast (actual)
SMARD_FILTER_FORECAST   = 411   # Prognostizierter Stromverbrauch – Netzlast (SMARD official forecast)

# ---------------------------------------------------------------------------
# Dataset date boundaries
# ---------------------------------------------------------------------------
KAGGLE_END_DATE  = "2025-09-30"   # last date fully covered by the Kaggle dataset
SMARD_START_DATE = "2025-10-01"   # first date fetched from SMARD actual

# ---------------------------------------------------------------------------
# Price project — initial scope (phase 1)
# ---------------------------------------------------------------------------
# Target series
SMARD_FILTER_PRICE_DE_LU = 4169  # Day-ahead market price DE/LU (EUR/MWh)
SMARD_REGION_PRICE_DE_LU = "DE-LU"

# Initial generation predictors from SMARD
SMARD_FILTER_WIND_ONSHORE = 4067
SMARD_FILTER_WIND_OFFSHORE = 1225
SMARD_FILTER_PV = 4068
SMARD_FILTER_OTHER_CONVENTIONAL = 1227

# Forecast generation predictors from SMARD
SMARD_FILTER_WIND_ONSHORE_FC = 123
SMARD_FILTER_WIND_OFFSHORE_FC = 3791
SMARD_FILTER_PV_FC = 125

# Initial fetch plan (series_id -> SMARD filter)
SMARD_PRICE_START_FILTERS = {
    "price_de_lu_eur_mwh": SMARD_FILTER_PRICE_DE_LU,
    "gen_wind_onshore_mwh": SMARD_FILTER_WIND_ONSHORE,
    "gen_wind_offshore_mwh": SMARD_FILTER_WIND_OFFSHORE,
    "gen_pv_mwh": SMARD_FILTER_PV,
    "gen_other_conventional_mwh": SMARD_FILTER_OTHER_CONVENTIONAL,
    "forecast_wind_onshore_mwh": SMARD_FILTER_WIND_ONSHORE_FC,
    "forecast_wind_offshore_mwh": SMARD_FILTER_WIND_OFFSHORE_FC,
    "forecast_pv_mwh": SMARD_FILTER_PV_FC,
}

# New normalized tables for price pipeline
TABLE_SERIES_CATALOG = "series_catalog"
TABLE_TIMESERIES_VALUES = "timeseries_values"
TABLE_INGESTION_RUNS = "ingestion_runs"
TABLE_DATA_QUALITY_LOG = "data_quality_log"

# ---------------------------------------------------------------------------
# Calendar / holiday features
# ---------------------------------------------------------------------------
DE_STATE_CODES = [
    'BB', 'BE', 'BW', 'BY', 'HB', 'HE',
    'HH', 'MV', 'NI', 'NW', 'RP', 'SH',
    'SL', 'SN', 'ST', 'TH',
]

PANDEMIC_START = pd.Timestamp('2020-03-01', tz='Europe/Berlin')
PANDEMIC_END   = pd.Timestamp('2021-12-31', tz='Europe/Berlin')

# ---------------------------------------------------------------------------
# Weather — Open-Meteo
# ---------------------------------------------------------------------------
WEATHER_VARIABLES = [
    'apparent_temperature',
    'rain',
    'snowfall',
    'wind_speed_10m',
    'shortwave_radiation',
    # 'temperature_2m' dropped: high correlation with apparent_temperature (see notebook 02 EDA)
]

# Population-weighted average over 5 major German cities
SELECTED_CITIES = {
    'Berlin':    {'latitude': 52.5200, 'longitude': 13.4050},
    'Hamburg':   {'latitude': 53.5511, 'longitude':  9.9937},
    'München':   {'latitude': 48.1351, 'longitude': 11.5820},
    'Köln':      {'latitude': 50.9375, 'longitude':  6.9603},
    'Frankfurt': {'latitude': 50.1109, 'longitude':  8.6821},
}

CITY_POPULATION = {
    'Berlin':    3_644_826,
    'Hamburg':   1_841_179,
    'München':   1_471_508,
    'Köln':      1_085_664,
    'Frankfurt':   753_056,
}

# ---------------------------------------------------------------------------
# Weather feature engineering thresholds
# ---------------------------------------------------------------------------
BASE_TEMPERATURE_HEATING = 18   # °C — heating degree days threshold
BASE_TEMPERATURE_COOLING = 25   # °C — cooling degree days threshold

# ---------------------------------------------------------------------------
# Price weather pipeline (PV/Wind separated to avoid leakage)
# ---------------------------------------------------------------------------
PV_WEATHER_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
]

WIND_WEATHER_VARIABLES = [
    "wind_speed_100m",
    "wind_direction_100m",
]

# Canonical series ids for technology-separated weather ingestion.
# These ids are used in series_catalog and timeseries_values.
PV_WEATHER_SERIES_IDS = {
    var: f"pv_weather_{var}" for var in PV_WEATHER_VARIABLES
}

WIND_WEATHER_SERIES_IDS = {
    var: f"wind_weather_{var}" for var in WIND_WEATHER_VARIABLES
}

WIND_LAND_WEATHER_SERIES_IDS = {
    var: f"wind_land_weather_{var}" for var in WIND_WEATHER_VARIABLES
}

WIND_SEA_WEATHER_SERIES_IDS = {
    var: f"wind_sea_weather_{var}" for var in WIND_WEATHER_VARIABLES
}

# Cluster centroid coordinates derived from notebook 02 exports.
# Keep cluster_id stable because yearly capacity CSVs depend on it.
PV_CLUSTER_CENTROIDS = [
    {"cluster_id": 0, "latitude": 48.254208117021278, "longitude": 9.95483580319149},
    {"cluster_id": 1, "latitude": 53.2118939516129, "longitude": 12.086464116935483},
    {"cluster_id": 2, "latitude": 50.086581113207551, "longitude": 7.0442732877358489},
    {"cluster_id": 3, "latitude": 51.5458006056338, "longitude": 13.807042591549296},
    {"cluster_id": 4, "latitude": 48.983946330316741, "longitude": 11.90809092760181},
    {"cluster_id": 5, "latitude": 49.92590152147239, "longitude": 10.256971736196318},
    {"cluster_id": 6, "latitude": 54.199199412698412, "longitude": 9.64202534920635},
    {"cluster_id": 7, "latitude": 52.316793511627907, "longitude": 8.448149488372092},
    {"cluster_id": 8, "latitude": 53.1419551959799, "longitude": 13.704316376884423},
    {"cluster_id": 9, "latitude": 51.541309815384615, "longitude": 11.716682680769232},
]

WIND_CLUSTER_CENTROIDS = [
    {"cluster_id": 0, "latitude": 51.843001675603219, "longitude": 8.4977388927613937},
    {"cluster_id": 1, "latitude": 54.720160583333332, "longitude": 13.918991070175441},
    {"cluster_id": 2, "latitude": 53.93755020402299, "longitude": 7.2917516408045975},
    {"cluster_id": 3, "latitude": 52.208402816849819, "longitude": 10.491895989010988},
    {"cluster_id": 4, "latitude": 54.01242887841191, "longitude": 9.7625609503722082},
    {"cluster_id": 5, "latitude": 52.437984534883718, "longitude": 13.9643738},
    {"cluster_id": 6, "latitude": 50.770402399159664, "longitude": 6.9432348655462182},
    {"cluster_id": 7, "latitude": 54.216298573584908, "longitude": 6.2127478999999992},
    {"cluster_id": 8, "latitude": 48.820484169014087, "longitude": 9.7376546760563372},
    {"cluster_id": 9, "latitude": 52.474594870833336, "longitude": 12.150012475},
]

WIND_CLUSTER_CENTROIDS_SEA = [
    {"cluster_id": 0, "latitude": 54.192663, "longitude": 6.572168},
    {"cluster_id": 1, "latitude": 54.813840, "longitude": 13.979021},
]

WIND_CLUSTER_CENTROIDS_LAND = [
    {"cluster_id": 0, "latitude": 54.144150, "longitude": 9.865936},
    {"cluster_id": 1, "latitude": 52.253366, "longitude": 7.417368},
    {"cluster_id": 2, "latitude": 52.544914, "longitude": 13.916800},
    {"cluster_id": 3, "latitude": 48.735615, "longitude": 10.063631},
    {"cluster_id": 4, "latitude": 53.076917, "longitude": 8.966918},
    {"cluster_id": 5, "latitude": 52.245920, "longitude": 10.581706},
    {"cluster_id": 6, "latitude": 49.664907, "longitude": 7.580589},
    {"cluster_id": 7, "latitude": 51.037630, "longitude": 6.355913},
    {"cluster_id": 8, "latitude": 52.488807, "longitude": 12.155722},
    {"cluster_id": 9, "latitude": 51.414065, "longitude": 8.853566},
]

PV_CLUSTER_LOCATIONS = {
    f"pv_cluster_{row['cluster_id']}": {
        "latitude": row["latitude"],
        "longitude": row["longitude"],
    }
    for row in PV_CLUSTER_CENTROIDS
}

WIND_CLUSTER_LOCATIONS = {
    f"wind_cluster_{row['cluster_id']}": {
        "latitude": row["latitude"],
        "longitude": row["longitude"],
    }
    for row in WIND_CLUSTER_CENTROIDS
}

WIND_CLUSTER_LOCATIONS_LAND = {
    f"wind_land_cluster_{row['cluster_id']}": {
        "latitude": row["latitude"],
        "longitude": row["longitude"],
    }
    for row in WIND_CLUSTER_CENTROIDS_LAND
}

WIND_CLUSTER_LOCATIONS_SEA = {
    f"wind_sea_cluster_{row['cluster_id']}": {
        "latitude": row["latitude"],
        "longitude": row["longitude"],
    }
    for row in WIND_CLUSTER_CENTROIDS_SEA
}


DATABASE_URL = "sqlite:///../db/energy_demand.db"


