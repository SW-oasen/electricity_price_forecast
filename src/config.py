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

# TODO double check - old source till 2023
# Capacity-weighted average location of top 10 PV plants in Germany (for solar radiation feature) 
PV_PLANTS = [
    #{"name": "Energiepark Witznitz", "capacity_mw": 605, "latitude": 51.1638, "longitude": 12.4186}, # seit 2024, aber nicht auf Marktdatenregister, daher rausgenommen
    {"name": "Solarpark Weesow-Willmersdorf", "capacity_mw": 187, "latitude": 52.5000, "longitude": 14.0000},
    {"name": "Solarpark Senftenberg", "capacity_mw": 187, "latitude": 51.5333, "longitude": 14.0167},
    {"name": "Solarpark Meuro", "capacity_mw": 166, "latitude": 51.5000, "longitude": 14.0000},
    {"name": "Solarpark Lieberose", "capacity_mw": 165, "latitude": 51.5833, "longitude": 14.2000},
    {"name": "Solarpark Jänschwalde", "capacity_mw": 145, "latitude": 51.6500, "longitude": 14.3000},
    {"name": "Solarpark Schwarzheide", "capacity_mw": 120, "latitude": 51.5000, "longitude": 14.1000},
    {"name": "Solarpark Schipkau", "capacity_mw": 110, "latitude": 51.6000, "longitude": 14.1500},
    {"name": "Solarpark Finsterwalde", "capacity_mw": 100, "latitude": 51.7000, "longitude": 14.2500},
    {"name": "Solarpark Drebkau", "capacity_mw": 90, "latitude": 51.5500, "longitude": 14.0500},
    {"name": "Solarpark Großräschen", "capacity_mw": 80, "latitude": 51.4500, "longitude": 14.0000},
    {"name": "Solarpark Ruhland", "capacity_mw": 75, "latitude": 51.6500, "longitude": 14.2000},
]

# Capacity-weighted average location of top 10 onshore wind farms in Germany (for wind speed feature)
WIND_FARMS = [
    {"name": "Windpark Kölleda", "capacity_mw": 200, "latitude": 51.2000, "longitude": 11.5000},
    {"name": "Windpark Holtriem", "capacity_mw": 150, "latitude": 53.5000, "longitude": 7.5000},
    {"name": "Windpark Hohenwarsleben", "capacity_mw": 120, "latitude": 52.0000, "longitude": 11.0000},
    {"name": "Windpark Borkum Riffgrund 1", "capacity_mw": 112, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Borkum Riffgrund 2", "capacity_mw": 112, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Gode Wind 1", "capacity_mw": 111, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Gode Wind 2", "capacity_mw": 111, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Nordsee One", "capacity_mw": 110, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Amrumbank West", "capacity_mw": 80, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Trianel Borkum", "capacity_mw": 80, "latitude": 54.0000, "longitude": 6.5000},
]

# Capacity-weighted average location of top 10 offshore wind farms in Germany (for offshore wind speed feature)
OFFSHORE_WIND_FARMS = [ 
    {"name": "Windpark Borkum Riffgrund 1", "capacity_mw": 112, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Borkum Riffgrund 2", "capacity_mw": 112, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Gode Wind 1", "capacity_mw": 111, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Gode Wind 2", "capacity_mw": 111, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Nordsee One", "capacity_mw": 110, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Amrumbank West", "capacity_mw": 80, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Trianel Borkum", "capacity_mw": 80, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark Meerwind Süd/Ost", "capacity_mw": 80, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark EnBW Baltic 2", "capacity_mw": 48, "latitude": 54.0000, "longitude": 6.5000},
    {"name": "Windpark EnBW Baltic 1", "capacity_mw": 48, "latitude": 54.0000, "longitude": 6.5000},
]

