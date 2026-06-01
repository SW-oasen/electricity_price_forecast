# --------- fetch the data from Kaggle and save it to the raw data folder ----------
from functools import lru_cache
import shutil
import os
from typing import Literal

dataset_link = "dsersun/europe-electricity-load-hourly-20192025"  # just the owner/dataset part
destination = "../data/raw"
def fetch_kaggle_dataset(in_dataset_link=dataset_link, in_destination=destination):
    '''
    Fetch the dataset from Kaggle and save it to the raw data folder.
    '''
    import kagglehub  # pip install kagglehub
    cache_path = kagglehub.dataset_download(in_dataset_link)
    #print(f"Downloaded to cache: {cache_path}")

    # Copy all files from cache to destination
    for file in os.listdir(cache_path):
        shutil.copy(os.path.join(cache_path, file), in_destination)
        print(f"Copied: {file} → {in_destination}")


# ========== prepare the energy data from Kaggle file for modeling ============

from IPython.display import display
import numpy as np
import pandas as pd

orig_file_path = "../data/raw/MHLV_2019_2025_combined.csv"
processed_file_path = "../data/processed/energy_weather_2019_2025.csv"


def prepare_energy_data_for_modeling(file_path=orig_file_path):
    '''
    Prepare energy data for modeling: 
        read the energy data from the Kaggle dataset
        create time-based features
        return the prepared DataFrame along with the start and end date of the energy data.
    '''
    out_df = pd.read_csv(file_path)

    out_df = rename_time_column(out_df)
    # DateUTC column contains UTC-valued naive timestamps — localize UTC first, then convert to Berlin
    out_df['time'] = pd.to_datetime(out_df['time']).dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin").dt.as_unit('s')
    out_df = out_df.rename(columns={'Value': 'EnergyDemand'})

    out_start_date = out_df['time'].min().strftime("%Y-%m-%d")
    out_end_date = out_df['time'].max().strftime("%Y-%m-%d")

    out_df = out_df[out_df['CountryCode'] == 'DE']  # filter for Germany, since we want to predict German energy demand
    out_df = out_df.drop(columns=[col for col in out_df.columns if col not in ['time', 'EnergyDemand']], errors='ignore')  # keep only the relevant columns, ignore if they are not present

    out_df = out_df.sort_values('time').reset_index(drop=True)

    out_df = create_time_based_features(out_df, in_year=out_df['time'].dt.year.max())
    out_df = create_energy_features(out_df)

    return out_df, out_start_date, out_end_date


# --------- scrape SMARD data ----------

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# Project-specific config — single source of truth
from config import (
    SMARD_BASE, SMARD_HEADERS, SMARD_REGION, SMARD_RESOLUTION,
    SMARD_FILTER_NETZLAST, SMARD_FILTER_FORECAST,
    KAGGLE_END_DATE, SMARD_START_DATE,
    DE_STATE_CODES, PANDEMIC_START, PANDEMIC_END,
    WEATHER_VARIABLES, SELECTED_CITIES, CITY_POPULATION,
    BASE_TEMPERATURE_HEATING, BASE_TEMPERATURE_COOLING,
)

# Generic utility classes
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → util importable
from util.smard_client import SmardClient
from util.time_features import TimeFeatureCreator
from util.openmeteo_client import OpenMeteoClient

# Keep legacy aliases so any code that imports them directly still works
HEADERS = SMARD_HEADERS
FILTER_NETZLAST = SMARD_FILTER_NETZLAST


def fetch_smard_netzlast(
    in_start_date: str,
    in_end_date: str,
    output_file: str | None = None,
    region: str = SMARD_REGION,
    resolution: str = SMARD_RESOLUTION,
    filter_id: int = SMARD_FILTER_NETZLAST,
    sleep: float = 0.3
) -> pd.DataFrame:
    """
    Fetch Realisierter Stromverbrauch (Netzlast) from the SMARD chart_data API.
    Delegates to SmardClient (util/smard_client.py); signature unchanged.
    """
    client = SmardClient(
        filter_id=filter_id,
        region=region,
        base_url=SMARD_BASE,
        headers=SMARD_HEADERS,
        resolution=resolution,
        sleep=sleep,
    )
    df = client.fetch(in_start_date, in_end_date)
    df = df.rename(columns={'load_MWh': 'EnergyDemand'})

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        df.to_csv(output_file, index=False)

    return df

# --------- add time based features for the energy demand data ----------

import holidays

# rename the time column to 'time' for consistency across datasets
def rename_time_column(in_df):
    known_time_cols = ('time', 'timestamp', 'DateUTC')
    for col in known_time_cols:
        if col in in_df.columns and col != 'time':
            in_df = in_df.rename(columns={col: 'time'})
            break
    return in_df

# Keep module-level aliases so any code that imports them directly still works
pandemic_start = PANDEMIC_START
pandemic_end   = PANDEMIC_END

def holiday_ratio(date):
    '''Fraction of German states with a public holiday on date (delegates to TimeFeatureCreator).'''
    _tfc = TimeFeatureCreator(country='DE', state_codes=DE_STATE_CODES)
    return _tfc.holiday_ratio(date)

def create_time_based_features(in_df, in_year, time_column='time',
                               in_pandemic_start=PANDEMIC_START,
                               in_pandemic_end=PANDEMIC_END):
    '''Create time-based features. Delegates to TimeFeatureCreator (util/time_features.py).'''
    tfc = TimeFeatureCreator(
        country='DE',
        state_codes=DE_STATE_CODES,
        pandemic_start=in_pandemic_start,
        pandemic_end=in_pandemic_end,
        time_column=time_column,
    )
    return tfc.create(in_df, year=in_year)

def create_energy_features(in_df):
    out_df = in_df.copy()   

    known_load_cols = ('EnergyDemand', 'load_MWh', 'Value')
    for col in known_load_cols:
        if col in out_df.columns and col != 'EnergyDemand':
            out_df = out_df.rename(columns={col: 'EnergyDemand'})
            break

    # add lagged features for energy demand (shifted by 24 hours, 168 hours (1 week) to capture daily, weekly, and yearly patterns)
    out_df['EnergyDemand_lag_24h'] = out_df['EnergyDemand'].shift(24)   # 1 day
    out_df['EnergyDemand_lag_168h'] = out_df['EnergyDemand'].shift(168)   # 1 week
    # lag_8760h (1 year) is not useful, it leads to worse scoring and makes future prediction more difficult
    #out_df['EnergyDemand_lag_8760h'] = out_df['EnergyDemand'].shift(8760) # 1 year

    # rolling after shift(1): each row sees the mean of the 24/168 hours immediately before it,
    # without a 24-hour gap. This is safe because lag lookup (not shift+tail) is used for prediction.
    out_df['EnergyDemand_rolling_mean_24h'] = out_df['EnergyDemand'].shift(1).rolling(24).mean()   # daily pattern
    out_df['EnergyDemand_rolling_mean_168h'] = out_df['EnergyDemand'].shift(1).rolling(168).mean() # weekly pattern
    # rolling_mean_8760h (1 year) is not useful, it leads to worse scoring and makes future prediction more difficult
    #out_df['EnergyDemand_rolling_mean_8760h'] = out_df['EnergyDemand'].shift(1).rolling(8760).mean() # yearly pattern

    # drop nan rows after lagging and rolling calculations
    out_df = out_df.dropna()

    return out_df

# =========== prepare energy data for prediction ==============

def prepare_energy_data_for_prediction(prediction_date, history_days=15):
    '''
    Prepare energy data for prediction: 
        fetch the energy data from SMARD
        create time-based features, 
        and return the prepared DataFrame.
    '''
    # for prediction, we need to fetch the most recent energy data to create lagged features, since the Kaggle dataset only goes up to 2025-09-30
    start_date, end_date = get_start_end_date(prediction_date, history_days)
    #print('\n---- debugging output of prepare_energy_data_for_prediction ----')
    #print(start_date, end_date)
    
    out_df = fetch_smard_netzlast(start_date, end_date)
    #print('\nfetched from smard', out_df['time'].head(3))
    
    out_df = create_energy_features(out_df)
    #print('\nafter creating energy features', out_df['time'].head(3))

    # bug 2 corrected
    # create time hourly of end_date+1 features for the future prediction date, since the lagged features will be based on the past 24h and 168h of energy demand, we need to have at least 168h of energy data up to the day before the prediction date
    time_24h_after_end_date = pd.date_range(end_date, periods=24, freq='h').tz_localize("Europe/Berlin", nonexistent='shift_forward', ambiguous='infer') + pd.Timedelta(hours=24)
    out_df = out_df.sort_values('time').tail(24)
    out_df['time'] = time_24h_after_end_date
    #print('\nafter creating future time rows head:', out_df['time'].head(3))
    #print('\n---- end of debugging output of prepare_energy_data_for_prediction ----\n')

    out_df = create_time_based_features(out_df, in_year=pd.to_datetime(end_date).year)

    return out_df


# ----------- fetch weather data from open-meteo ----------


# Local aliases — kept for any code that still references them by the old names
weather_variables = WEATHER_VARIABLES
selected_cities   = SELECTED_CITIES
city_population   = CITY_POPULATION

raw_tmp_path = "../data/raw/tmp/"

def fetch_weather_data_for_cities(in_start_date=KAGGLE_END_DATE,
                                  in_end_date=KAGGLE_END_DATE,
                                  in_selected_cities=None,
                                  in_weather_variables=None):
    '''Fetch archive weather for cities. Delegates to OpenMeteoClient (util/openmeteo_client.py).'''
    cities    = in_selected_cities or SELECTED_CITIES
    variables = in_weather_variables or WEATHER_VARIABLES
    client = OpenMeteoClient(
        cities=cities,
        city_population={c: CITY_POPULATION[c] for c in cities},
        weather_variables=variables,
    )
    # Return the raw per-city dict for backward compat with callers that iterate over cities
    import requests as _requests, time as _time
    api_start = (pd.to_datetime(in_start_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    clip_start = pd.Timestamp(in_start_date, tz='Europe/Berlin')
    vars_str = ','.join(variables)
    city_dict = {}
    for city, coords in cities.items():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={coords['latitude']}&longitude={coords['longitude']}"
            f"&start_date={api_start}&end_date={in_end_date}"
            f"&hourly={vars_str}&timezone=UTC"
        )
        for attempt in range(3):
            try:
                r = _requests.get(url, timeout=30); r.raise_for_status()
                data = r.json(); break
            except _requests.exceptions.RequestException:
                if attempt == 2: raise
                _time.sleep(5)
        df_city = pd.DataFrame(data['hourly'])
        df_city['time'] = (
            pd.to_datetime(df_city['time'], utc=True)
            .dt.tz_convert('Europe/Berlin').dt.as_unit('s')
        )
        df_city = df_city[df_city['time'] >= clip_start].reset_index(drop=True)
        city_dict[city] = df_city
        _time.sleep(1)
    return city_dict

def merge_weather_data_with_city_weights(in_weather_city_dict,
                                         in_city_population=None,
                                         in_weather_variables=None):
    '''Merge city weather DataFrames with population weights. Delegates to OpenMeteoClient._merge_cities.'''
    pop  = in_city_population or CITY_POPULATION
    vars_ = in_weather_variables or WEATHER_VARIABLES
    cities_subset = {c: SELECTED_CITIES[c] for c in in_weather_city_dict if c in SELECTED_CITIES}
    client = OpenMeteoClient(
        cities=cities_subset,
        city_population={c: pop[c] for c in cities_subset},
        weather_variables=vars_,
    )
    return client._merge_cities(in_weather_city_dict)

# feature engineering: create new features based on existing ones, such as rolling averages, lagged variables, or interaction terms

base_temperature_heating = BASE_TEMPERATURE_HEATING
base_temperature_cooling = BASE_TEMPERATURE_COOLING

def create_weather_features(in_df,
                    in_base_temperature_heating=BASE_TEMPERATURE_HEATING,
                    in_base_temperature_cooling=BASE_TEMPERATURE_COOLING):
    '''
    Create new features based on existing ones, such as rolling averages, lagged variables, or interaction terms.
    '''
    out_df = in_df.copy()

    # add rolling average and lagged variable for apparent_temperature
    out_df['apparent_temperature_rolling_mean_24h'] = out_df['apparent_temperature'].shift(1).rolling(window=24).mean()
    out_df['apparent_temperature_lag_24h'] = out_df['apparent_temperature'].shift(24)

    # add rolling average and lagged varirable for shortwave_radiation_0m
    out_df['shortwave_radiation_0m_rolling_mean_24h'] = out_df['shortwave_radiation'].shift(1).rolling(window=24).mean()
    out_df['shortwave_radiation_0m_lag_24h'] =   out_df['shortwave_radiation'].shift(24)

    # add heating degree days (HDD) and cooling degree days (CDD) features
    out_df['heating_degree'] = out_df['apparent_temperature'].apply(lambda x: max(0, in_base_temperature_heating - x))  # HDD is calculated as the difference between a base temperature (e.g., 18°C) and the actual temperature, but only if the actual temperature is below the base temperature
    out_df['cooling_degree'] = out_df['apparent_temperature'].apply(lambda x: max(0, x - in_base_temperature_cooling))  # CDD is calculated as the difference between the actual temperature and a base temperature (e.g., 25°C), but only if the actual temperature is above the base temperature

    return out_df

# ============ prepare weather data ============

def prepare_weather_data(in_start_date,
                        in_end_date,
                        in_selected_cities=None,
                        in_weather_variables=None,
                        in_city_population=None):
    '''Fetch + merge + feature-engineer archive weather. Delegates to OpenMeteoClient.'''
    cities    = in_selected_cities or SELECTED_CITIES
    variables = in_weather_variables or WEATHER_VARIABLES
    pop       = in_city_population   or CITY_POPULATION
    client = OpenMeteoClient(cities=cities, city_population=pop, weather_variables=variables)
    out_df = client.fetch_archive(in_start_date, in_end_date)
    out_df = rename_time_column(out_df)
    out_df = out_df.sort_values('time').reset_index(drop=True)
    out_df = create_weather_features(out_df)
    return out_df

# ============ prepare weather forecast data ============

def fetch_weather_forecast_for_cities(
        in_selected_cities=None,
        in_weather_variables=None,
        forecast_days: int = 2):
    '''Fetch forecast weather per city. Delegates to OpenMeteoClient.fetch_forecast internals.'''
    cities    = in_selected_cities or SELECTED_CITIES
    variables = in_weather_variables or WEATHER_VARIABLES
    client = OpenMeteoClient(
        cities=cities,
        city_population={c: CITY_POPULATION[c] for c in cities},
        weather_variables=variables,
    )
    # Re-implement per-city dict return for backward compat
    import requests as _requests, time as _time
    vars_str = ','.join(variables)
    today_midnight = pd.Timestamp.now(tz='Europe/Berlin').normalize()
    out = {}
    for city, coords in cities.items():
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords['latitude']}&longitude={coords['longitude']}"
            f"&hourly={vars_str}&forecast_days={forecast_days}&past_days=1&timezone=UTC"
        )
        for attempt in range(3):
            try:
                r = _requests.get(url, timeout=30); r.raise_for_status()
                data = r.json(); break
            except _requests.exceptions.RequestException:
                if attempt == 2: raise
                _time.sleep(5)
        df_city = pd.DataFrame(data['hourly'])
        df_city['time'] = (
            pd.to_datetime(df_city['time'], utc=True)
            .dt.tz_convert('Europe/Berlin').dt.as_unit('s')
        )
        df_city = df_city[df_city['time'] >= today_midnight].reset_index(drop=True)
        out[city] = df_city
        _time.sleep(1)
    return out


def prepare_weather_forecast(
        in_selected_cities=None,
        in_weather_variables=None,
        in_city_population=None,
        forecast_days: int = 2):
    '''Prepare forecast weather DataFrame. Delegates to OpenMeteoClient.fetch_forecast.'''
    cities    = in_selected_cities or SELECTED_CITIES
    variables = in_weather_variables or WEATHER_VARIABLES
    pop       = in_city_population   or CITY_POPULATION
    client = OpenMeteoClient(cities=cities, city_population=pop, weather_variables=variables)
    out_df = client.fetch_forecast(forecast_days=forecast_days)
    out_df = rename_time_column(out_df)
    return out_df


def prepare_weather_for_prediction(prediction_date, lookback_days=2, forecast_days=3):
    '''Fetch archive + forecast weather combined for a prediction date. Delegates to OpenMeteoClient.'''
    client = OpenMeteoClient(
        cities=SELECTED_CITIES,
        city_population=CITY_POPULATION,
        weather_variables=WEATHER_VARIABLES,
    )
    out_df = client.prepare_for_prediction(
        prediction_date, lookback_days=lookback_days, forecast_days=forecast_days
    )
    out_df = rename_time_column(out_df)
    out_df = create_weather_features(out_df)
    return out_df


# ---------- comnbine energy and weather dataset for modeling ----------

def combine_energy_weather_dataset(in_energy_df, in_weather_df):
    '''
    Prepare the combined energy and weather dataset for modeling: merge the energy and weather datasets on the timestamp, 
    drop columns with high correlation, and save the combined dataset to the processed data folder.
    '''
    in_energy_df = in_energy_df.copy()
    in_weather_df = in_weather_df.copy()

    # may cause doubled time conversion
    #in_energy_df['time'] = pd.to_datetime(in_energy_df['time']).dt.tz_convert("Europe/Berlin")

    #if in_weather_df['time'].dt.tz is None:
    #    in_weather_df['time'] = pd.to_datetime(in_weather_df['time']).dt.tz_localize("Europe/Berlin", nonexistent='shift_forward', ambiguous='infer')
    #else:
   #     in_weather_df['time'] = pd.to_datetime(in_weather_df['time']).dt.tz_convert("Europe/Berlin")

    out_df = pd.merge(in_energy_df, in_weather_df, on='time', how='inner')
    out_df = out_df.sort_values('time').reset_index(drop=True)

    return out_df

# =========== prepare the combined energy and weather dataset for modeling ============

def prepare_data_for_modeling():
    '''
    Prepare the combined energy and weather dataset for modeling: merge the energy and weather datasets on the timestamp, 
    drop columns with high correlation, and save the combined dataset to the processed data folder.
    '''
    df_energy, start_date, end_date = prepare_energy_data_for_modeling()
    df_weather = prepare_weather_data(in_start_date=start_date, in_end_date=end_date)
    out_df = combine_energy_weather_dataset(df_energy, df_weather)
    out_df = out_df.sort_values('time').reset_index(drop=True)

    return out_df

# =========== prepare historical data for prediction and compare with actual data ============

def prepare_historical_data_for_prediction(prediction_date, history_days=15):
    '''
    Prepare historical data for prediction and compare with actual data: 
        fetch the energy data from SMARD for the past history_days, 
        create time-based features, and return the prepared DataFrame along with the actual energy demand for the prediction date.
    '''
    start_date, _ = get_start_end_date(prediction_date, history_days)
    # Include prediction_date itself so features (lag, rolling) are generated for that day.
    # get_start_end_date returns end_date = prediction_date - 1, which would exclude it.
    end_date = prediction_date
    df_energy = fetch_smard_netzlast(start_date, end_date)
    df_energy = create_time_based_features(df_energy, in_year=pd.to_datetime(prediction_date).year)
    df_energy = create_energy_features(df_energy)

    df_weather = prepare_weather_data(in_start_date=start_date, in_end_date=end_date)
    out_df = combine_energy_weather_dataset(df_energy, df_weather)
    out_df = out_df.sort_values('time').reset_index(drop=True)

    # Snap start to next clean midnight: lag/rolling dropna may leave a partial first day
    # (e.g. SMARD UTC→Berlin conversion shifts the series start to 23:00 instead of 00:00)
    first_row = out_df['time'].iloc[0]
    if first_row.hour != 0 or first_row.minute != 0:
        next_midnight = (first_row + pd.Timedelta(days=1)).normalize()
        out_df = out_df[out_df['time'] >= next_midnight].reset_index(drop=True)

    # use only the predictors 
    out_df = out_df.drop(columns=['EnergyDemand'], errors='ignore')
    
    return out_df


# ============= prepare future features for prediction ============ 

def get_start_end_date(prediction_date, history_days=15):
    # Need at least 168h (7 days) of history for lag/rolling features + buffer
    start_date = (pd.to_datetime(prediction_date).tz_localize("Europe/Berlin") - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")
    end_date   = (pd.to_datetime(prediction_date).tz_localize("Europe/Berlin") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return start_date, end_date


def create_tomorrow_time(prediction_date):
    pred_start = pd.Timestamp(prediction_date, tz="Europe/Berlin")
    future_times = pd.date_range(pred_start, periods=24, freq="h")

    #print(f'Appending future energy rows for prediction date {prediction_date}, future times: {future_times[0]} to {future_times[-1]}')

    return  future_times
    #(pd.concat([df_energy, df_future], ignore_index=True).sort_values("time").reset_index(drop=True)

def prepare_for_prediction_tomorrow(prediction_date, history_days=15):
    '''
    Prepare the dataset for predicting tomorrow's energy demand:
        fetch the most recent energy data from SMARD to create lagged features, 
        create time-based features, fetch forecast weather data, and combine them into a single DataFrame for prediction.
    '''
    start_date, end_date = get_start_end_date(prediction_date, history_days)

    df_history = fetch_smard_netzlast(start_date, end_date)
    df_history = df_history.sort_values('time').reset_index(drop=True)

    # Build energy lag features via direct timestamp lookup instead of shift(24)+tail(24).
    # The tail(24) approach was wrong in two ways:
    #   1. shift(24) on history rows gives lag values 24h too old once times are relabeled.
    #   2. If SMARD only has partial data for today (publication delay), tail(24) straddles
    #      two partial days, causing a visible ~12h pattern shift.
    energy_idx = df_history.set_index('time')['EnergyDemand']

    def _get(t):
        # Fall back to same hour last week if the exact timestamp is not yet published by SMARD
        v = energy_idx.get(t, np.nan)
        if pd.isna(v):
            v = energy_idx.get(t - pd.Timedelta(hours=168), np.nan)
        return v

    def _rolling_mean(t, n):
        # Matches training: EnergyDemand.shift(24).rolling(n).mean() at prediction time T
        # = mean of EnergyDemand from T-(n+23)h to T-24h  (n consecutive hours)
        window = energy_idx.loc[
            (energy_idx.index >= t - pd.Timedelta(hours=n + 23)) &
            (energy_idx.index <= t - pd.Timedelta(hours=24))
        ]
        return window.mean() if len(window) > 0 else np.nan

    future_times = create_tomorrow_time(prediction_date)

    df_energy = pd.DataFrame({
        'time':                           future_times,
        'EnergyDemand_lag_24h':           [_get(t - pd.Timedelta(hours=24))  for t in future_times],
        'EnergyDemand_lag_168h':          [_get(t - pd.Timedelta(hours=168)) for t in future_times],
        'EnergyDemand_rolling_mean_24h':  [_rolling_mean(t, 24)              for t in future_times],
        'EnergyDemand_rolling_mean_168h': [_rolling_mean(t, 168)             for t in future_times],
    })

    df_energy = create_time_based_features(
        df_energy,
        in_year=pd.to_datetime(prediction_date).year
    )

    pred_start = pd.Timestamp(prediction_date, tz="Europe/Berlin")
    pred_end   = pred_start + pd.Timedelta(days=1)

    df_energy = df_energy[
        (df_energy["time"] >= pred_start) &
        (df_energy["time"] < pred_end)
    ].copy()

    df_weather = prepare_weather_for_prediction(prediction_date)
    df_weather = df_weather[
        (df_weather["time"] >= pred_start) &
        (df_weather["time"] < pred_end)
    ].copy()

    out_df = combine_energy_weather_dataset(df_energy, df_weather)

    return out_df.drop(columns=["EnergyDemand"], errors="ignore")

