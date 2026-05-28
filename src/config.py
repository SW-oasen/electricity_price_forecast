"""
config.py — project-specific configuration for the Germany electricity demand project.

All domain constants live here so that:
- util/ classes remain fully generic (no hardcoded country/city defaults)
- etl_demand.py and fetch_prepare_data_demand.py import from one place instead of each defining their own
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

# Initial fetch plan (series_id -> SMARD filter)
SMARD_PRICE_START_FILTERS = {
    "price_de_lu_eur_mwh": SMARD_FILTER_PRICE_DE_LU,
    "gen_wind_onshore_mwh": SMARD_FILTER_WIND_ONSHORE,
    "gen_wind_offshore_mwh": SMARD_FILTER_WIND_OFFSHORE,
    "gen_pv_mwh": SMARD_FILTER_PV,
    "gen_other_conventional_mwh": SMARD_FILTER_OTHER_CONVENTIONAL,
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
