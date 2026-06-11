"""
fetch_price_data.py

Datenvorbereitung fuer die Preis-Pipeline:
- Zeitspalten-Vereinheitlichung
- Feature Engineering
- Cleaning

Funktionen sind so gehalten, dass sie direkt in Notebooks oder im ETL-Kontext genutzt werden koennen.
"""

import sqlite3
from pathlib import Path
from xml.sax.handler import all_features

import pandas as pd
import numpy as np
import requests
import sqlalchemy as sa
import sys
import time

from etl_price import update_price_database
from etl_demand import update_demand_database

try:
    from src.config import (
        DE_STATE_CODES, PANDEMIC_START, PANDEMIC_END,
        SMARD_BASE, SMARD_HEADERS, SMARD_REGION, SMARD_RESOLUTION,
        SMARD_FILTER_NETZLAST, SMARD_FILTER_FORECAST,
        SMARD_FILTER_PRICE_DE_LU, SMARD_REGION_PRICE_DE_LU,
        SMARD_FILTER_WIND_ONSHORE, SMARD_FILTER_WIND_OFFSHORE,
        SMARD_FILTER_PV, SMARD_FILTER_OTHER_CONVENTIONAL,
        PV_WEATHER_VARIABLES, WIND_WEATHER_VARIABLES,
        PV_CLUSTER_LOCATIONS, WIND_CLUSTER_LOCATIONS,
        PV_WEATHER_SERIES_IDS, WIND_WEATHER_SERIES_IDS
    )
except ImportError:
    from config import (
        DE_STATE_CODES, PANDEMIC_START, PANDEMIC_END,
        SMARD_BASE, SMARD_HEADERS, SMARD_REGION, SMARD_RESOLUTION,
        SMARD_FILTER_NETZLAST, SMARD_FILTER_FORECAST,
        SMARD_FILTER_PRICE_DE_LU, SMARD_REGION_PRICE_DE_LU,
        SMARD_FILTER_WIND_ONSHORE, SMARD_FILTER_WIND_OFFSHORE,
        SMARD_FILTER_PV, SMARD_FILTER_OTHER_CONVENTIONAL,
        PV_WEATHER_VARIABLES, WIND_WEATHER_VARIABLES,
        PV_CLUSTER_LOCATIONS, WIND_CLUSTER_LOCATIONS,
        PV_WEATHER_SERIES_IDS, WIND_WEATHER_SERIES_IDS
    )

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → util importable
from config import DATABASE_PATH, PROJECT_ROOT
from util.time_features import TimeFeatureCreator
from util.smard_client import SmardClient
from util.openmeteo_client import OpenMeteoClient
from util.weather_weighted import build_yearly_weights

# For demand prediction
try:
    from src.etl_demand import prepare_for_demand_prediction_tomorrow
    from src.train_predict_model import load_model_from_pickle
    from src.fetch_demand_data import prepare_weather_for_prediction
    from src.etl_price import update_price_database
    from src.etl_price import create_lag_rolling_features
except ImportError:
    from etl_demand import prepare_for_demand_prediction_tomorrow
    from train_predict_model import load_model_from_pickle
    from fetch_demand_data import prepare_weather_for_prediction
    from etl_price import update_price_database
    from etl_price import create_lag_rolling_features

#PROJECT_ROOT = Path(__file__).resolve().parents[1]
#DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "energy_demand.db"


def normalize_time_column(in_df: pd.DataFrame, col: str = "time_utc", to_utc: bool = True, freq: str = "min") -> pd.DataFrame:
    """
    Vereinheitlicht eine Zeitspalte:
    - Konvertiert zu pandas.Timestamp
    - Optional in UTC
    - Rundet auf gewünschte Auflösung (default: Minute)

    Args:
        in_df: DataFrame mit Zeitspalte (wird nicht verändert)
        col: Name der Zeitspalte
        to_utc: True → in UTC konvertieren
        freq: 'H' (Stunde), 'min' (Minute), '15min', etc.
    Returns:
        out_df: DataFrame mit vereinheitlichter Zeitspalte
    """
    out_df = in_df.copy()
    out_df[col] = pd.to_datetime(out_df[col], utc=to_utc)
    out_df[col] = out_df[col].dt.floor(freq)
    return out_df


#DEFAULT_DATABASE_URL = "sqlite:///../db/energy_demand.db"
def load_time_series_data_from_db(database_path = DATABASE_PATH) -> pd.DataFrame:
    """
    Load PV- and Wind data from database.
    Pivot the tables so that each series_id becomes a separate column, indexed by time.
    Args:
        database_path: Path to the SQLite database containing the tables timeseries_values and series_catalog.
    Returns:
        df: DataFrame with time series data, time column as Timestamp in UTC

    """
    #pv_ids = tuple(PV_WEATHER_SERIES_IDS.values())
    #wind_ids = tuple(WIND_WEATHER_SERIES_IDS.values())

    query = """
        SELECT time, series_id, value
        FROM timeseries_values
        ORDER BY time DESC, series_id
        """    
    #conn = sa.create_engine(database_url).connect()
    db_path = Path(database_path)
    conn = sa.create_engine(f"sqlite:///{db_path.as_posix()}").connect()
 
    df = pd.read_sql(query, conn)
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.pivot(index='time', columns='series_id', values='value')
    df = df.sort_index()

    return df


def load_energy_demand_table(database_path = DATABASE_PATH) -> pd.DataFrame:
    """
    Load demand table used for price modeling.

    Returns a DataFrame with UTC time and both actual demand + SMARD forecast.
    """
    db_path = Path(database_path)
    conn = sa.create_engine(f"sqlite:///{db_path.as_posix()}").connect()
    try:
        df_dem = pd.read_sql(
            """
            SELECT time, energy_demand_mwh, smard_forecast_mwh
            FROM energy_demand
            ORDER BY time
            """,
            conn,
        )
    finally:
        conn.close()

    df_dem["time"] = pd.to_datetime(df_dem["time"], utc=True)
    return df_dem


def build_price_feature_base(
    df_price_raw: pd.DataFrame,
    df_demand_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build base price-feature table.

    Rule:
    - historical rows use actual values
    - only missing values are filled with day-ahead forecast
    """
    df_base = df_price_raw.merge(df_demand_raw, on="time", how="inner")

    time_features = ["hour", "hour_sin", "hour_cos", "weekday", "month",
                     "is_weekend", "is_holiday", "is_bridge_day", "is_pandemic_time", "holiday_ratio", "holiday_weight"
                     ]  
    tfc = TimeFeatureCreator(
        country="DE",
        state_codes=DE_STATE_CODES,
        pandemic_start=PANDEMIC_START,
        pandemic_end=PANDEMIC_END,
        time_column="time",
        include_features=time_features,
    )
    df_base = tfc.create(df_base, year=int(df_base["time"].dt.year.max()))
    #print(f"Created time features df_base.shape: {df_base.shape}")

    # Actual generation channels
    if "gen_wind_onshore_mwh" in df_base.columns and "gen_wind_offshore_mwh" in df_base.columns:
        # only calculate if not already present or if components are available
        mask_wind_comp = df_base["gen_wind_onshore_mwh"].notna() | df_base["gen_wind_offshore_mwh"].notna()
        if "gen_wind_total_mwh" not in df_base.columns:
            df_base["gen_wind_total_mwh"] = np.nan
        df_base.loc[mask_wind_comp, "gen_wind_total_mwh"] = (
            df_base.loc[mask_wind_comp, "gen_wind_onshore_mwh"].fillna(0) + 
            df_base.loc[mask_wind_comp, "gen_wind_offshore_mwh"].fillna(0)
        )
    
    if "gen_pv_mwh" in df_base.columns:
        df_base["gen_pv_total_mwh"] = df_base["gen_pv_mwh"]

    # Forecast generation channels (SMARD official forecasts)
    df_base["gen_wind_total_forecast_mwh"] = np.nan
    if "forecast_wind_onshore_mwh" in df_base.columns and "forecast_wind_offshore_mwh" in df_base.columns:
        df_base["gen_wind_total_forecast_mwh"] = df_base["forecast_wind_onshore_mwh"] + df_base["forecast_wind_offshore_mwh"]
    
    df_base["gen_pv_total_forecast_mwh"] = df_base.get("forecast_pv_mwh", np.nan)

    # Day-ahead fallback helpers
    df_base["demand_forecast_mwh"] = pd.to_numeric(df_base.get("demand_forecast_mwh", np.nan), errors="coerce")
    # support legacy smard_forecast_mwh column if present and demand_forecast_mwh is empty
    if "smard_forecast_mwh" in df_base.columns:
        df_base["demand_forecast_mwh"] = df_base["demand_forecast_mwh"].fillna(df_base["smard_forecast_mwh"])

    df_base["demand_forecast_smard_mwh"] = df_base["demand_forecast_mwh"] # alias for backward compat if needed
    df_base["gen_wind_da_proxy_mwh"] = df_base["gen_wind_total_mwh"].shift(24)
    df_base["gen_pv_da_proxy_mwh"] = df_base["gen_pv_total_mwh"].shift(24)

    # Use SMARD forecasts to fill the proxy where available
    df_base["gen_wind_da_proxy_mwh"] = df_base["gen_wind_total_forecast_mwh"].fillna(df_base["gen_wind_da_proxy_mwh"])
    df_base["gen_pv_da_proxy_mwh"] = df_base["gen_pv_total_forecast_mwh"].fillna(df_base["gen_pv_da_proxy_mwh"])

    # Inputs used by model (actual-first)
    df_base["demand_input_mwh"] = pd.to_numeric(df_base["energy_demand_mwh"], errors="coerce")
    df_base["gen_wind_input_mwh"] = pd.to_numeric(df_base["gen_wind_total_mwh"], errors="coerce")
    df_base["gen_pv_input_mwh"] = pd.to_numeric(df_base["gen_pv_total_mwh"], errors="coerce")

    # Fallback only on missing actual values
    mask_dem_missing = df_base["demand_input_mwh"].isna()
    mask_wind_missing = df_base["gen_wind_input_mwh"].isna()
    mask_pv_missing = df_base["gen_pv_input_mwh"].isna()

    df_base.loc[mask_dem_missing, "demand_input_mwh"] = df_base.loc[mask_dem_missing, "demand_forecast_smard_mwh"]
    df_base.loc[mask_wind_missing, "gen_wind_input_mwh"] = df_base.loc[mask_wind_missing, "gen_wind_da_proxy_mwh"]
    df_base.loc[mask_pv_missing, "gen_pv_input_mwh"] = df_base.loc[mask_pv_missing, "gen_pv_da_proxy_mwh"]

    df_base["residual_load_input_mwh"] = df_base["demand_input_mwh"] - df_base["gen_wind_input_mwh"] - df_base["gen_pv_input_mwh"]

    df_base["renewable_share"] = (df_base["gen_pv_input_mwh"] + df_base["gen_wind_input_mwh"]) / df_base["demand_input_mwh"]

    df_base["pv_share"] = df_base["gen_pv_input_mwh"] / df_base["demand_input_mwh"]

    df_base["wind_share"] = df_base["gen_wind_input_mwh"] / df_base["demand_input_mwh"]

    df_base["residual_load_ratio"] = df_base["residual_load_input_mwh"] / df_base["demand_input_mwh"]

    df_base["holiday_renewable_share"] = df_base["is_holiday"] * df_base["renewable_share"]

    # Lags (Richer structure from 'plus' model)
    lag_cols = [
        "price_de_lu_eur_mwh",
        "demand_input_mwh",
        #"energy_demand_mwh", # redundant with demand_input_mwh and causes issues with missing values, consider adding back if needed
        "gen_wind_input_mwh",
        #"gen_wind_total_mwh",
        "gen_pv_input_mwh",
        #"gen_pv_total_mwh",
        "gen_other_conventional_mwh",
        "residual_load_input_mwh",
    ]
    # Ensure all lag columns exist (even if as NaNs) to maintain consistent feature set
    for col in lag_cols:
        l24, l168 = f"{col}_lag_24h", f"{col}_lag_168h"
        if l24 not in df_base.columns:
            df_base[l24] = df_base[col].shift(24)
        if l168 not in df_base.columns:
            df_base[l168] = df_base[col].shift(168)

    # Extra regime/interaction features
    df_base["wind_pv_ratio_input"] = df_base["gen_wind_input_mwh"] / (df_base["gen_pv_input_mwh"].abs() + 1.0)
    
    # Ensure residual_vs_conv_gap exists even if underlying data is missing
    if "gen_other_conventional_mwh_lag_24h" in df_base.columns:
        df_base["residual_vs_conv_gap"] = df_base["residual_load_input_mwh"] - df_base["gen_other_conventional_mwh_lag_24h"]
    else:
        df_base["residual_vs_conv_gap"] = np.nan
        
    df_base["price_weekly_delta"] = df_base["price_de_lu_eur_mwh_lag_24h"] - df_base["price_de_lu_eur_mwh_lag_168h"]

    # Weather features (cyclical wind direction + lags)
    weather_lag_cols = [
        "pv_weather_shortwave_radiation",
        "pv_weather_cloud_cover",
        "pv_weather_diffuse_radiation",
        "pv_weather_direct_radiation",
        "wind_weather_wind_speed_100m",
        #"wind_weather_wind_direction_100m",  # wind direction seems to have little predictive power and causes issues with missing values, consider adding back as cyclical features if needed
    ]
    existing_weather_cols = [c for c in weather_lag_cols if c in df_base.columns]
    if existing_weather_cols:
        #if "wind_weather_wind_direction_100m" in df_base.columns:
        #    wind_dir_rad = np.deg2rad(df_base["wind_weather_wind_direction_100m"] % 360)
        #    df_base["wind_dir_sin"] = np.sin(wind_dir_rad)
        #    df_base["wind_dir_cos"] = np.cos(wind_dir_rad)
        for col in existing_weather_cols:
            df_base[f"{col}_lag_24h"] = df_base[col].shift(24)
            df_base[f"{col}_lag_168h"] = df_base[col].shift(168)
        if "wind_weather_wind_speed_100m" in df_base.columns:
            # Physical wind power features (mirroring logic in aggregate_weighted_wind_vector_features)
            v = df_base["wind_weather_wind_speed_100m"]
            df_base["wind_speed_clipped"] = v.clip(lower=3.0, upper=25.0)
            v_rated = v.clip(upper=13.0)
            df_base["wind_speed_pow2"] = v_rated ** 2
            df_base["wind_speed_pow3"] = v_rated ** 3

            df_base["residual_x_wind_speed"] = (
                df_base["residual_load_input_mwh"] * df_base["wind_weather_wind_speed_100m"]
            )

    return df_base


def prepare_price_model_dataset():
    """
    One-liner data preparation for the price model.

    Returns:
        df_base: merged table with engineered intermediate features
        df_price_model: training-ready dataframe (time, target, model features)
        feature_cols: ordered list of model feature columns
    """
    time_features = ["hour", "hour_sin", "hour_cos", "weekday", "month",
                    "is_weekend", "is_holiday", "is_bridge_day", "is_pandemic_time", "holiday_ratio", "holiday_weight"
                    ]

    prediction_cols = [
        "demand_input_mwh", #'energy_demand_mwh', 
        "gen_wind_input_mwh", 
        #'gen_wind_total_mwh', 
        'gen_wind_da_proxy_mwh', 
        "gen_pv_input_mwh", 
        'gen_pv_mwh',
        #'gen_pv_total_mwh', 
        'gen_pv_da_proxy_mwh', 
        'gen_other_conventional_mwh', 
        "residual_load_input_mwh",
        'residual_x_wind_speed', 
    ]

    prediction_lag_features = [ 
        "price_de_lu_eur_mwh_lag_24h", "price_de_lu_eur_mwh_lag_168h",
        "demand_input_mwh_lag_24h", "demand_input_mwh_lag_168h",
        #"energy_demand_mwh_lag_24h", "energy_demand_mwh_lag_168h",
        "gen_pv_input_mwh_lag_24h", "gen_pv_input_mwh_lag_168h",
        #'gen_pv_total_mwh_lag_168h', 'gen_pv_total_mwh_lag_24h', 
        "gen_wind_input_mwh_lag_24h", "gen_wind_input_mwh_lag_168h",
        #'gen_wind_total_mwh_lag_24h', 'gen_wind_total_mwh_lag_168h',
        "gen_other_conventional_mwh_lag_24h", "gen_other_conventional_mwh_lag_168h",
        "residual_load_input_mwh_lag_24h", "residual_load_input_mwh_lag_168h",
        #'gen_wind_offshore_mwh', 'gen_wind_onshore_mwh',  
        ]

    engineered_features = ["wind_pv_ratio_input", "residual_vs_conv_gap", "price_weekly_delta", 
                           "renewable_share", "pv_share", "wind_share", "residual_load_ratio", "holiday_renewable_share"
                           ]

    weather_features = [
        "wind_weather_wind_speed_100m", 
        #'wind_weather_wind_direction_100m', 
        'wind_speed_pow2', 'wind_speed_clipped', 'wind_speed_pow3',
        "pv_weather_shortwave_radiation", "pv_weather_cloud_cover",
        "pv_weather_diffuse_radiation", "pv_weather_direct_radiation",
        #"wind_dir_sin", "wind_dir_cos",
    ]

    weather_lag_features = [
        "pv_weather_shortwave_radiation_lag_24h", "pv_weather_shortwave_radiation_lag_168h",
        "pv_weather_cloud_cover_lag_24h", "pv_weather_cloud_cover_lag_168h",
        "pv_weather_diffuse_radiation_lag_24h", "pv_weather_diffuse_radiation_lag_168h",
        "pv_weather_direct_radiation_lag_24h", "pv_weather_direct_radiation_lag_168h",
        "wind_weather_wind_speed_100m_lag_24h", "wind_weather_wind_speed_100m_lag_168h",
        #"wind_weather_wind_direction_100m_lag_24h", 
        #"wind_weather_wind_direction_100m_lag_168h",
    ]

    update_price_database() # ensure we have the latest data in the database before loading
    df_price_raw = load_time_series_data_from_db().reset_index()
    df_price_raw["time"] = pd.to_datetime(df_price_raw["time"], utc=True).dt.tz_convert("Europe/Berlin")
    #print(f"Loaded price raw data with {len(df_price_raw)} rows, time range: {df_price_raw['time'].min()} -> {df_price_raw['time'].max()}")

    update_demand_database() # ensure we have the latest demand data in the database before loading
    df_demand_raw = load_energy_demand_table()
    df_demand_raw["time"] = pd.to_datetime(df_demand_raw["time"], utc=True).dt.tz_convert("Europe/Berlin")
    #print(f"Loaded demand raw data with {len(df_demand_raw)} rows, time range: {df_demand_raw['time'].min()} -> {df_demand_raw['time'].max()}")

    df_base = build_price_feature_base(df_price_raw=df_price_raw, df_demand_raw=df_demand_raw)
    #print(f"Built base price feature table with {len(df_base)} rows, time range: {df_base['time'].min()} -> {df_base['time'].max()}")

    # Define the core features that the model expects
    all_prediction_features = (
        prediction_cols 
        + prediction_lag_features 
        + engineered_features 
        + time_features
        + weather_features 
        + weather_lag_features
    )

    df_price_model = df_base[["time", "price_de_lu_eur_mwh", *all_prediction_features]].dropna().reset_index(drop=True)
    #df_price_model = df_base.dropna().reset_index(drop=True)
    #print(f"Prepared price model dataset with {len(df_price_model)} rows, time range: {df_price_model['time'].min()} -> {df_price_model['time'].max()}")

    return df_price_model


def fetch_smard_data(
    in_start_date: str,
    in_end_date: str,
    filter_id: int,
    region: str = SMARD_REGION,
    resolution: str = SMARD_RESOLUTION,
) -> pd.DataFrame:
    """Helper to fetch from SMARD."""
    client = SmardClient(
        filter_id=filter_id,
        region=region,
        base_url=SMARD_BASE,
        headers=SMARD_HEADERS,
        resolution=resolution,
    )
    return client.fetch(in_start_date, in_end_date)


def _fetch_weighted_weather(
    locations: dict,
    variables: list[str],
    weights_path: Path,
    prefix: str,
    start_date: str,
    end_date: str,
    forecast_days: int = 3
) -> pd.DataFrame:
    """Fetch and weight weather for clusters (archive + forecast)."""
    weights_total = build_yearly_weights(weights_path, prefix)
    target_year = pd.to_datetime(start_date).year
    if target_year not in weights_total:
        target_year = max(weights_total.keys())
    
    weights = weights_total[target_year]
    # filter locations to match weights
    relevant_locations = {k: locations[k] for k in weights if k in locations}
    
    client = OpenMeteoClient(
        cities=relevant_locations,
        city_population={k: 1 for k in relevant_locations},
        weather_variables=variables,
    )
    
    # 1. Archive
    df_archive = client.fetch_archive_weighted_locations(
        locations=relevant_locations,
        location_weights=weights,
        start_date=start_date,
        end_date=end_date,
        weather_variables=variables
    )
    
    # 2. Forecast
    vars_str = ','.join(variables)
    today_midnight = pd.Timestamp.now(tz='Europe/Berlin').normalize()
    
    loc_dict: dict[str, pd.DataFrame] = {}
    for name, coords in relevant_locations.items():
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords['latitude']}&longitude={coords['longitude']}"
            f"&hourly={vars_str}&forecast_days={forecast_days}&past_days=1&timezone=UTC"
        )
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30); r.raise_for_status()
                data = r.json(); break
            except:
                if attempt == 2: raise
                time.sleep(5)

        df_loc = pd.DataFrame(data['hourly'])
        df_loc['time'] = ensure_berlin_time(df_loc['time'])
        df_loc = df_loc[df_loc['time'] >= today_midnight].reset_index(drop=True)
        loc_dict[name] = df_loc
        time.sleep(0.1)
        
    df_forecast = client._merge_weighted(loc_dict, weights, variables)
    
    # Combine
    df_combined = pd.concat([df_archive, df_forecast], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    # Rename variables to include prefix
    rename_map = {v: f"{prefix}_weather_{v}" for v in variables}
    return df_combined.rename(columns=rename_map)

# prepare features for PV generation model from dataframe loaded from time series table in database, which will be used as input for the price model
def prepare_historical_pv_features(df):
    cols_pv_features = ['gen_pv_mwh', 
                        'pv_weather_cloud_cover',
                        'pv_weather_diffuse_radiation',
                        'pv_weather_direct_radiation',
                        'pv_weather_shortwave_radiation']

    df_pv = df[cols_pv_features].copy().reset_index()

    # Create lag and rolling features for the weather variables, defaults to lags of 24 and 168 hours and rolling windows of 24 and 168 hours
    df_pv = create_lag_rolling_features(df_pv, cols_pv_features)

    # Create time features
    tfc = TimeFeatureCreator(
        country="DE",
        state_codes=DE_STATE_CODES,
        pandemic_start=PANDEMIC_START,
        pandemic_end=PANDEMIC_END,
        time_column="time",
        include_features=["year", "hour", "month"],  # add more if needed
    )

    df_pv = tfc.create(df_pv, year=int(df_pv["time"].dt.year.max()))
    df_pv = df_pv.dropna().reset_index(drop=True)

    return df_pv

# prepare features for Wind generation model from dataframe loaded from time series table in database, which will be used as input for the price model
def prepare_historical_wind_features(df):
    cols_wind_features = ['time', 
                        'gen_wind_offshore_mwh', 
                        'gen_wind_onshore_mwh', 
                        #'wind_weather_wind_direction_100m', 
                        'wind_weather_wind_speed_100m']

    df_wind = df[cols_wind_features].copy().reset_index()

    # Physical thresholds for wind generation
    # Cut-in: ~3m/s, Rated: ~13m/s, Cut-out: ~25m/s
    v = df_wind['wind_weather_wind_speed_100m']
    df_wind['wind_speed_clipped'] = v.clip(lower=3.0, upper=25.0)

    # Power is proportional to v^3, but levels off at rated speed (~13m/s)
    v_rated = v.clip(upper=13.0)
    df_wind['wind_speed_pow2'] = v_rated ** 2
    df_wind['wind_speed_pow3'] = v_rated ** 3

    df_wind['gen_wind_total_mwh'] = df_wind['gen_wind_offshore_mwh'] + df_wind['gen_wind_onshore_mwh']
    df_wind = df_wind.drop(['gen_wind_offshore_mwh', 'gen_wind_onshore_mwh'], axis=1)
    
    return df_wind

def _predict_generation_tomorrow(
    target_series: str,
    model_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Helper to predict generation (PV or Wind) for a specific range using custom ML models."""
    model_path = PROJECT_ROOT / "models" / f"{model_name}.pkl"
    if not model_path.exists():
        return pd.DataFrame()
        
    model = load_model_from_pickle(model_path)
    
    # 1. Load history from DB (need enough for lags)
    df_hist = load_time_series_data_from_db().reset_index()
    df_hist['time'] = pd.to_datetime(df_hist['time'], utc=True).dt.tz_convert("Europe/Berlin")
    
    # 2. Get weather forecast for the range
    # Ensure weather covers first hour of start_date
    df_weather = prepare_weather_for_prediction(start_date, forecast_days=3)
    
    # 3. Features
    # PV and Wind models use weather series + generation lags
    # Combine history and weather
    df_combined = pd.concat([df_hist, df_weather], ignore_index=True)
    df_combined = df_combined.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    
    # 4. Feature Engineering (Lags/Rolling)
    if "pv" in target_series:
        cols_base = ['gen_pv_mwh', 
                     'pv_weather_cloud_cover', 
                     'pv_weather_diffuse_radiation', 
                     'pv_weather_direct_radiation', 
                     'pv_weather_shortwave_radiation']
    else:
        # Handle wind_total
        if 'gen_wind_onshore_mwh' in df_combined.columns and 'gen_wind_offshore_mwh' in df_combined.columns:
            df_combined['gen_wind_total_mwh'] = df_combined['gen_wind_onshore_mwh'].fillna(0) + df_combined['gen_wind_offshore_mwh'].fillna(0)
        cols_base = ['gen_wind_total_mwh', 
                     'wind_weather_wind_speed_100m', 
                     'wind_speed_clipped', 
                     'wind_speed_pow2',
                     'wind_speed_pow3']
        df_combined = prepare_historical_wind_features(df_combined)

    for c in cols_base:
        if c not in df_combined.columns:
            df_combined[c] = np.nan
        
    df_feat = create_lag_rolling_features(df_combined, cols_base, lags=(24, 168), rolling_windows=(24, 168))
    
    # Ensure time column is in correct format and timezone
    df_feat["time"] = pd.to_datetime(df_feat["time"], utc=True).dt.tz_convert("Europe/Berlin")

    tfc = TimeFeatureCreator(
        country="DE",
        state_codes=DE_STATE_CODES,
        pandemic_start=PANDEMIC_START,
        pandemic_end=PANDEMIC_END,
        time_column="time",
        include_features=["year", "month", "hour"]
    )
    df_feat = tfc.create(df_feat, year=pd.Timestamp.now().year)
    
    # 5. Filter for target range
    ts_start = pd.Timestamp(start_date, tz="Europe/Berlin").normalize()
    ts_end = pd.Timestamp(end_date, tz="Europe/Berlin").normalize() + pd.Timedelta(days=1)
    df_range = df_feat[(df_feat['time'] >= ts_start) & (df_feat['time'] < ts_end)].copy()
    
    if df_range.empty:
        return pd.DataFrame()
        
    X = df_range.drop(columns=['time', 'gen_pv_mwh', 'gen_wind_total_mwh'], errors='ignore')
    
    if hasattr(model, 'feature_name_'):
        model_features = model.feature_name_
        available_features = [f for f in model_features if f in X.columns]
        X = X[available_features]
    
    print(f"\nDataset prepared for predicting {target_series} \n time range {ts_start} -> {ts_end} \n shape: {X.shape} \n features used: {X.columns.tolist()}")
    preds = model.predict(X)
    
    return pd.DataFrame({
        'time': df_range['time'],
        target_series: preds
    })


def prepare_data_for_price_prediction_tomorrow(history_days=15):
    """
    Prepare the dataset for tomorrow's price prediction context.
    Uses database for historical data and a custom LGBM model for demand forecast.
    """
    # 1. Ensure database is updated (historical data)
    update_price_database()
    
    today = pd.Timestamp.now(tz="Europe/Berlin")
    tomorrow = today + pd.Timedelta(days=1)
    yesterday = today - pd.Timedelta(days=1)

    prediction_date = tomorrow.strftime("%Y-%m-%d")
    today_date = today.strftime("%Y-%m-%d")
    pred_ts = tomorrow.normalize() # 00:00:00 of tomorrow
    today_ts = today.normalize() # 00:00:00 of today
    
    # 2. Load historical data from database
    # load_time_series_data_from_db returns pivoted [price, generations, weighted weather]
    df_ts_hist = load_time_series_data_from_db().reset_index()
    df_ts_hist['time'] = pd.to_datetime(df_ts_hist['time'], utc=True).dt.tz_convert("Europe/Berlin")
    print(f"\nLoaded historical time series data shape: {df_ts_hist.shape}\n  time range: {df_ts_hist['time'].min()} -> {df_ts_hist['time'].max()}")

    # Load demand actuals
    df_dem_hist = load_energy_demand_table()
    df_dem_hist = df_dem_hist.dropna(subset=["energy_demand_mwh"]) # ensure we have actual demand values for the historical part (model training relies on this)
    df_dem_hist['time'] = pd.to_datetime(df_dem_hist['time'], utc=True).dt.tz_convert("Europe/Berlin")
    print(f"\nLoaded historical demand data shape: {df_dem_hist.shape}\n  time range: {df_dem_hist['time'].min()} -> {df_dem_hist['time'].max()}")

    # 3. Weather Forecast (needed for both demand and price models)
    # Get weather covering today and tomorrow
    df_weather_fc = prepare_weather_for_prediction(prediction_date, forecast_days=3)
    print(f"\nPrepared weather forecast data shape: {df_weather_fc.shape}\n  time range: {df_weather_fc['time'].min()} -> {df_weather_fc['time'].max()}")
    
    # 4. Custom Demand Forecast (Today + Tomorrow to fill the gap)
    # Prepare features for demand model using ETL-based logic
    df_dem_feat_today = prepare_for_demand_prediction_tomorrow(today_date)
    print(f"\nPrepared demand features for today shape: {df_dem_feat_today.shape}\n  time range: {df_dem_feat_today['time'].min()} -> {df_dem_feat_today['time'].max()}")
    
    df_dem_feat_tomorrow = prepare_for_demand_prediction_tomorrow(prediction_date)
    print(f"\nPrepared demand features for tomorrow shape: {df_dem_feat_tomorrow.shape}\n  time range: {df_dem_feat_tomorrow['time'].min()} -> {df_dem_feat_tomorrow['time'].max()}")
    
    df_demand_features = pd.concat([df_dem_feat_today, df_dem_feat_tomorrow], ignore_index=True).drop_duplicates('time').sort_values('time')
    print(f"\nCombined demand features shape: {df_demand_features.shape}\n  time range: {df_demand_features['time'].min()} -> {df_demand_features['time'].max()}")
    
    # Load and run the demand model
    model_path = PROJECT_ROOT / "models" / "energy_demand_lgbm_model.pkl"
    if not model_path.exists():
        # fallback to another known name if the user's name is slightly different in the filesystem
        model_path = PROJECT_ROOT / "models" / "best_lgbm_model_bayesian_etl.pkl"
        
    demand_model = load_model_from_pickle(model_path)
    print(f"\nLoaded demand model from {model_path}, features: {demand_model.feature_name_ if hasattr(demand_model, 'feature_name_') else 'unknown'}")

    # Predict demand
    X_demand = df_demand_features.drop(columns=['time'], errors='ignore')

    # Enforce feature order if needed
    if hasattr(demand_model, 'feature_name_'):
        model_features_demand = demand_model.feature_name_
        available_features_demand = [f for f in model_features_demand if f in X_demand.columns]
        X_demand = X_demand[available_features_demand]
    
    demand_pred = demand_model.predict(X_demand)
    print(f"\nPredicted demand shape: {demand_pred.shape}\n  time range: {df_demand_features['time'].min()} -> {df_demand_features['time'].max()}")

    df_demand_pred = pd.DataFrame({
        'time': df_demand_features['time'],
        'energy_demand_mwh': np.nan, # actual is unknown for the future
        'demand_forecast_mwh': demand_pred # we inject our custom forecast here
    })
    
    # NEW: Custom PV and Wind forecasts (Today + Tomorrow)
    df_pv_pred = _predict_generation_tomorrow('gen_pv_mwh', 'pv_lgbm_model', today_date, prediction_date)
    print(f"\nPredicted PV generation shape: {df_pv_pred.shape}\n  time range: {df_pv_pred['time'].min()} -> {df_pv_pred['time'].max()}")
    
    df_wind_pred = _predict_generation_tomorrow('gen_wind_total_mwh', 'wind_lgbm_model', today_date, prediction_date)
    print(f"\nPredicted Wind generation shape: {df_wind_pred.shape}\n  time range: {df_wind_pred['time'].min()} -> {df_wind_pred['time'].max()}")

    # 5. Combine everything for the price model base
    # Historical part
    df_price_raw_hist = df_ts_hist.copy()
    df_demand_raw_hist = df_dem_hist[['time', 'energy_demand_mwh', 'smard_forecast_mwh']].copy()
    # rename for internal consistency in the pipeline
    df_demand_raw_hist = df_demand_raw_hist.rename(columns={'smard_forecast_mwh': 'demand_forecast_mwh'})
    
    # Future part (today/tomorrow)
    # We define future rows starting from the beginning of today to the end of tomorrow
    future_times = pd.date_range(today_ts, periods=48, freq="h", tz="Europe/Berlin")
    df_future_rows = pd.DataFrame({'time': future_times})

    # Inject custom generation forecasts into future rows if available
    if not df_pv_pred.empty:
        df_future_rows = df_future_rows.merge(df_pv_pred, on='time', how='left')
    if not df_wind_pred.empty:
        df_future_rows = df_future_rows.merge(df_wind_pred, on='time', how='left')
    
    # Combine weather (history from DB + forecast from OpenMeteo)
    # Note: df_weather_fc from prepare_weather_for_prediction already includes treatment
    # but build_price_feature_base expects raw weather columns to apply its own lags.
    # So we take the weather forecast rows and join them.
    df_weather_fc_rows = df_weather_fc[df_weather_fc['time'] >= today_ts].copy()
    
    # Prepare the combined dataframes for build_price_feature_base
    df_price_raw = pd.concat([df_price_raw_hist, df_future_rows], ignore_index=True)
    df_price_raw = df_price_raw.merge(df_weather_fc_rows, on='time', how='left', suffixes=('', '_fc'))
    
    # Update weather columns with forecast where missing
    for col in df_weather_fc.columns:
        if col == 'time':
            continue
        
        col_fc = col + '_fc'
        if col_fc in df_price_raw.columns:
            if col not in df_price_raw.columns:
                # If column didn't exist in history, create it from forecast
                df_price_raw[col] = df_price_raw[col_fc]
            else:
                # If it existed, fill only NaNs (the future/gap)
                df_price_raw[col] = df_price_raw[col].fillna(df_price_raw[col_fc])
            
            df_price_raw = df_price_raw.drop(columns=[col_fc])

    df_demand_raw = pd.concat([df_demand_raw_hist, df_demand_pred], ignore_index=True)
    
    # Ensure sorted by time
    df_price_raw = df_price_raw.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    df_demand_raw = df_demand_raw.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    
    # 6. Apply Price Feature Engineering (including adding Time Features)
    # This will fill gaps using demand_forecast_mwh and gen_wind_total_mwh/gen_pv_mwh (our forecasts)
    df_base = build_price_feature_base(df_price_raw, df_demand_raw)
        
    # 8. Filter for target prediction date
    pred_end = pred_ts + pd.Timedelta(days=1)
    out_df = df_base[(df_base['time'] >= pred_ts) & (df_base['time'] < pred_end)].copy()
    
    return out_df.reset_index(drop=True)


def ensure_berlin_time(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s)

    if ts.dt.tz is None:
        # naive timestamps from API/database are treated as UTC
        ts = pd.to_datetime(s, utc=True).dt.tz_convert("Europe/Berlin")
    else:
        ts = ts.dt.tz_convert("Europe/Berlin")

    return ts.dt.as_unit("s")


def prepare_data_for_price_prediction_operational(
    target_date: str | None = None,
    price_model=None,
    price_model_path: Path | None = None,
    actual_available_until: str | None = None,
) -> pd.DataFrame:
    """
    Operational price-feature preparation.

    Simulates real forecasting situation:
    - only actual data up to actual_available_until is used
    - missing days before target_date are predicted recursively
    - predicted prices are inserted back so price_lag_24h is available
    - returns feature rows for target_date
    """

    update_price_database()
    update_demand_database()

    if target_date is None:
        target_ts = (pd.Timestamp.now(tz="Europe/Berlin") + pd.Timedelta(days=1)).normalize()
    else:
        target_ts = pd.Timestamp(target_date, tz="Europe/Berlin").normalize()

    target_date_str = target_ts.strftime("%Y-%m-%d")

    if price_model is None:
        if price_model_path is None:
            price_model_path = PROJECT_ROOT / "models" / "price_lgbm_model.pkl"
        price_model = load_model_from_pickle(price_model_path)

    # ------------------------------------------------------------
    # 1. Load historical raw data
    # ------------------------------------------------------------
    df_ts_hist_all = load_time_series_data_from_db().reset_index()
    df_ts_hist_all["time"] = pd.to_datetime(df_ts_hist_all["time"], utc=True).dt.tz_convert("Europe/Berlin")
    
    df_dem_hist_all = load_energy_demand_table()
    df_dem_hist_all["time"] = pd.to_datetime(df_dem_hist_all["time"], utc=True).dt.tz_convert("Europe/Berlin")
    
    df_dem_hist_all = df_dem_hist_all.rename(
        columns={"smard_forecast_mwh": "demand_forecast_mwh"}
    )

    # ------------------------------------------------------------
    # 2. Infer last complete actual day if not provided
    # ------------------------------------------------------------
    if actual_available_until is None:
        df_check = df_ts_hist_all.merge(
            df_dem_hist_all[["time", "energy_demand_mwh"]],
            on="time",
            how="inner",
        )

        required_cols = [
            "price_de_lu_eur_mwh",
            "gen_pv_mwh",
            "gen_wind_onshore_mwh",
            "gen_wind_offshore_mwh",
            "energy_demand_mwh",
        ]
        required_cols = [c for c in required_cols if c in df_check.columns]

        df_check["date"] = df_check["time"].dt.date
        complete_days = (
            df_check.dropna(subset=required_cols)
            .groupby("date")
            .size()
        )
        complete_days = complete_days[complete_days >= 24]

        if complete_days.empty:
            raise ValueError("No complete actual day found in database.")

        actual_until_date = pd.Timestamp(max(complete_days.index), tz="Europe/Berlin")
    else:
        actual_until_date = pd.Timestamp(actual_available_until, tz="Europe/Berlin").normalize()

    forecast_start = actual_until_date + pd.Timedelta(days=1)

    if forecast_start > target_ts:
        forecast_start = target_ts

    # ------------------------------------------------------------
    # 3. Keep actual history only up to actual_available_until
    # ------------------------------------------------------------
    actual_end_exclusive = actual_until_date + pd.Timedelta(days=1)

    df_ts_hist = df_ts_hist_all[df_ts_hist_all["time"] < actual_end_exclusive].copy()
    df_dem_hist = df_dem_hist_all[df_dem_hist_all["time"] < actual_end_exclusive].copy()

    # ------------------------------------------------------------
    # 4. Forecast demand for forecast_start ... target_date
    # ------------------------------------------------------------
    forecast_days = pd.date_range(
        forecast_start,
        target_ts,
        freq="D",
        tz="Europe/Berlin",
    )

    demand_frames = []

    demand_model_path = PROJECT_ROOT / "models" / "energy_demand_lgbm_model.pkl"
    if not demand_model_path.exists():
        demand_model_path = PROJECT_ROOT / "models" / "best_lgbm_model_bayesian_etl.pkl"

    demand_model = load_model_from_pickle(demand_model_path)

    for day in forecast_days:
        day_str = day.strftime("%Y-%m-%d")
        df_dem_feat = prepare_for_demand_prediction_tomorrow(day_str)

        X_dem = df_dem_feat.drop(columns=["time"], errors="ignore")
        if hasattr(demand_model, "feature_name_"):
            X_dem = X_dem.reindex(columns=list(demand_model.feature_name_))

        pred = demand_model.predict(X_dem)

        demand_frames.append(
            pd.DataFrame(
                {
                    "time": df_dem_feat["time"],
                    "energy_demand_mwh": np.nan,
                    "demand_forecast_mwh": pred,
                }
            )
        )

    df_dem_future = pd.concat(demand_frames, ignore_index=True)

    # ------------------------------------------------------------
    # 5. Forecast PV and Wind for forecast_start ... target_date
    # ------------------------------------------------------------
    start_str = forecast_start.strftime("%Y-%m-%d")
    end_str = target_ts.strftime("%Y-%m-%d")

    df_pv_pred = _predict_generation_tomorrow(
        target_series="gen_pv_mwh",
        model_name="pv_lgbm_model",
        start_date=start_str,
        end_date=end_str,
    )

    df_wind_pred = _predict_generation_tomorrow(
        target_series="gen_wind_total_mwh",
        model_name="wind_lgbm_model",
        start_date=start_str,
        end_date=end_str,
    )

    future_times = pd.date_range(
        forecast_start,
        target_ts + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        freq="h",
        tz="Europe/Berlin",
    )

    df_future_ts = pd.DataFrame({"time": future_times})

    if not df_pv_pred.empty:
        df_pv_pred["time"] = pd.to_datetime(df_pv_pred["time"]).dt.tz_convert("Europe/Berlin")
        df_future_ts = df_future_ts.merge(df_pv_pred, on="time", how="left")

    if not df_wind_pred.empty:
        df_wind_pred["time"] = pd.to_datetime(df_wind_pred["time"]).dt.tz_convert("Europe/Berlin")
        df_future_ts = df_future_ts.merge(df_wind_pred, on="time", how="left")

    # price initially unknown
    df_future_ts["price_de_lu_eur_mwh"] = np.nan

    # ------------------------------------------------------------
    # 6. Combine history + forecast rows
    # ------------------------------------------------------------
    df_price_raw = pd.concat([df_ts_hist, df_future_ts], ignore_index=True)
    df_price_raw = (
        df_price_raw
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )

    df_demand_raw = pd.concat([df_dem_hist, df_dem_future], ignore_index=True)
    df_demand_raw = (
        df_demand_raw
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # 7. Recursive price prediction for gap days before target_date
    # ------------------------------------------------------------
    recursive_days = pd.date_range(
        forecast_start,
        target_ts - pd.Timedelta(days=1),
        freq="D",
        tz="Europe/Berlin",
    )

    for day in recursive_days:
        day_start = day
        day_end = day + pd.Timedelta(days=1)

        df_base = build_price_feature_base(df_price_raw, df_demand_raw)

        df_day = df_base[
            (df_base["time"] >= day_start) &
            (df_base["time"] < day_end)
        ].copy()

        if df_day.empty:
            raise ValueError(f"No feature rows generated for recursive day {day.date()}")

        X = df_day.drop(columns=["time", "price_de_lu_eur_mwh"], errors="ignore")

        if hasattr(price_model, "feature_name_"):
            X = X.reindex(columns=list(price_model.feature_name_))
        elif hasattr(price_model, "feature_names_in_"):
            X = X.reindex(columns=list(price_model.feature_names_in_))

        pred_price = price_model.predict(X)

        mask = (
            (df_price_raw["time"] >= day_start) &
            (df_price_raw["time"] < day_end)
        )
        df_price_raw.loc[mask, "price_de_lu_eur_mwh"] = pred_price

    # ------------------------------------------------------------
    # 8. Build final target-day features
    # ------------------------------------------------------------
    df_base_final = build_price_feature_base(df_price_raw, df_demand_raw)

    out_df = df_base_final[
        (df_base_final["time"] >= target_ts) &
        (df_base_final["time"] < target_ts + pd.Timedelta(days=1))
    ].copy()

    if out_df.empty:
        raise ValueError(f"No target-day features generated for {target_date_str}")

    return out_df.reset_index(drop=True)