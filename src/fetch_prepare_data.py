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

SMARD_BASE = "https://www.smard.de/app/chart_data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; smard-fetcher/1.0)"}

# Filter IDs
FILTER_NETZLAST = 410          # Realisierter Stromverbrauch – Netzlast


def _get_index(filter_id: int, region: str = "DE", resolution: str = "hour") -> list[int]:
    """Return the list of weekly bucket timestamps (Unix ms) available for the given filter."""
    url = f"{SMARD_BASE}/{filter_id}/{region}/index_{resolution}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()["timestamps"]


def _fetch_week(filter_id: int, timestamp_ms: int, region: str = "DE", resolution: str = "hour") -> list:
    """Fetch the raw series [[ts_ms, value], ...] for one weekly bucket."""
    url = f"{SMARD_BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp_ms}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json().get("series", [])


def fetch_smard_netzlast(
    in_start_date: str,
    in_end_date: str,
    output_file: str | None = None,
    region: str = "DE",
    resolution: str = "hour",
    filter_id: int = FILTER_NETZLAST,
    sleep: float = 0.3
) -> pd.DataFrame:
    """
    Fetch Realisierter Stromverbrauch (Netzlast) from the SMARD chart_data API.

    Parameters
    ----------
    in_start_date : str
        Inclusive start in 'YYYY-MM-DD' format (local CET/CEST time).
    in_end_date : str
        Inclusive end in 'YYYY-MM-DD' format.
    output_file : str | None
        If given, save the result as CSV to this path.
    region : str
        SMARD region code, default 'DE'.
    resolution : str
        'hour' or 'quarterhour'.
    filter_id : int
        SMARD filter ID (410 = Netzlast).
    sleep : float
        Seconds to sleep between requests (be polite to the server).

    Returns
    -------
    pd.DataFrame with columns ['timestamp', 'load_MWh'].
    """
    # Convert date strings to UTC millisecond boundaries
    start_dt = pd.to_datetime(in_start_date).tz_localize("Europe/Berlin")
    end_dt   = pd.to_datetime(in_end_date).tz_localize("Europe/Berlin")
    # Include the full end day
    end_ms   = int(end_dt.timestamp() * 1000) + 86_400_000 - 1
    start_ms = int(start_dt.timestamp() * 1000)

    #print(f"Fetching index for filter {filter_id} / {region} / {resolution} ...")
    all_timestamps = _get_index(filter_id, region, resolution)

    # Keep only buckets that can overlap with [start_ms, end_ms].
    # Each bucket covers roughly one week (604_800_000 ms).
    week_ms = 7 * 24 * 3600 * 1000
    relevant = [ts for ts in all_timestamps if ts <= end_ms and ts + week_ms >= start_ms]

    if not relevant:
        print("No data available for the requested period.")
        return pd.DataFrame(columns=["timestamp", "load_MWh"])

    #print(f"Fetching {len(relevant)} weekly bucket(s) ...")
    rows = []
    for ts in relevant:
        series = _fetch_week(filter_id, ts, region, resolution)
        rows.extend(series)
        time.sleep(sleep)

    # Build DataFrame and clip to the exact requested range
    df = pd.DataFrame(rows, columns=["ts_ms", "load_MWh"])
    df = df.dropna(subset=["load_MWh"])
    df = df[(df["ts_ms"] >= start_ms) & (df["ts_ms"] <= end_ms)]
    # Downcast ms → s: hourly data needs no sub-second precision; aligns with open-meteo datetime64[s]
    df["timestamp"] = (pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
                       .dt.tz_convert("Europe/Berlin")
                       .dt.as_unit('s'))
    df = df[["timestamp", "load_MWh"]].sort_values("timestamp").reset_index(drop=True)

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        df.to_csv(output_file, index=False)
        #print(f"Saved to {output_file}")
    
    # timestamp -> time
    df = rename_time_column(df)     
    # rename load_MWh -> EnergyDemand to match training data column name
    df = df.rename(columns={'load_MWh': 'EnergyDemand'})

    return df

# --------- add time based features for the energy demand data ----------

# rename the time column to 'time' for consistency across datasets
def rename_time_column(in_df):  
    known_time_cols = ('time', 'timestamp', 'DateUTC')
    for col in known_time_cols:
        if col in in_df.columns and col != 'time':
            in_df = in_df.rename(columns={col: 'time'})
            break
    return in_df

# add holiday ratio depending the number of states in Germany with a holiday on that day
import holidays 

DE_STATE_CODES = ['BB', 'BE', 'BW', 'BY', 'HB', 'HE', 'HH', 'MV', 
                  'NI', 'NW', 'RP', 'SH', 'SL', 'SN', 'ST', 'TH']

# use lru_cache to cache the holiday data for each state and year, since the holiday data is static 
# and can be reused across multiple calls to the holiday_ratio function, this can improve performance 
# by avoiding redundant API calls or computations for the same state and year
@lru_cache(maxsize=None)
def _state_holidays(state_code, year):
    return holidays.Germany(subdiv=state_code, years=[year])

# calculate the holiday ratio for a given date, which is the number of states in Germany that have a holiday 
# on that date divided by the total number of states (16), this feature can help capture the impact of holidays 
# on energy demand, since holidays can lead to changes in energy consumption patterns due to factors 
# such as reduced industrial activity, increased residential usage, and changes in transportation demand
def holiday_ratio(date):
    '''
    Calculate the holiday ratio for a given date, which is the number of states in Germany that have a holiday 
    on that date divided by the total number of states (16).'''
    count = sum(1 for code in DE_STATE_CODES 
                if date in _state_holidays(code, date.year))
    return count / 16

pandemic_start = pd.to_datetime('2020-03-01', utc=True).tz_convert("Europe/Berlin")
pandemic_end = pd.to_datetime('2021-12-31', utc=True).tz_convert("Europe/Berlin")

def create_time_based_features(in_df, in_year, time_column='time', in_pandemic_start=pandemic_start, in_pandemic_end=pandemic_end):
    '''
    Create time-based features such as hour of day, day of week, and month of year.
    '''
    out_df = in_df.copy()
    out_df['year'] = out_df[time_column].dt.year.astype(int)
    out_df['hour'] = out_df[time_column].dt.hour.astype(int)
    out_df['weekday'] = out_df[time_column].dt.dayofweek.astype(int)
    out_df['month'] = out_df[time_column].dt.month.astype(int)
    out_df['is_weekend'] = out_df[time_column].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0).astype(int)

    de_holidays = holidays.Germany(years=range(2019, in_year + 1))
    out_df['is_holiday'] = out_df[time_column].dt.date.apply(lambda x: 1 if x in de_holidays else 0).astype(int)
    out_df['holiday_ratio'] = out_df[time_column].dt.date.apply(holiday_ratio).astype(float)

    # is_workday: 1 only if it is neither a weekend nor a public holiday — direct signal for high-demand days
    out_df['is_workday'] = ((out_df['is_weekend'] == 0) & (out_df['is_holiday'] == 0)).astype(int)

    # is_bridge_day: a working day sandwiched between a public holiday and a weekend (or another holiday).
    # These days typically see reduced industrial activity similar to a holiday.
    dates = out_df[time_column].dt.date
    out_df['is_bridge_day'] = dates.apply(
        lambda d: 1 if (
            d.weekday() not in (5, 6)           # the day itself is a weekday
            and d not in de_holidays             # but not already a holiday
            and (
                # previous day is holiday or weekend
                (
                    (pd.Timestamp(d) - pd.Timedelta(days=1)).date() in de_holidays
                    or (pd.Timestamp(d) - pd.Timedelta(days=1)).date().weekday() >= 5
                )
                # AND next day is holiday or weekend
                and (
                    (pd.Timestamp(d) + pd.Timedelta(days=1)).date() in de_holidays
                    or (pd.Timestamp(d) + pd.Timedelta(days=1)).date().weekday() >= 5
                )
            )
        ) else 0
    ).astype(int)

    # holiday_weight: combines holiday_ratio with the is_weekend flag so the model sees
    # a single continuous "non-working-day intensity" signal in [0, 1].
    # On a holiday with ratio=1.0 it equals 1.0; on a pure weekend it equals 0.5;
    # on a bridge day it picks up a small boost from holiday_ratio of adjacent days.
    out_df['holiday_weight'] = out_df[['holiday_ratio', 'is_weekend']].apply(
        lambda row: max(row['holiday_ratio'], row['is_weekend'] * 0.5), axis=1
    ).astype(float)

    # add pandemic feature
    out_df['is_pandemic_time'] = out_df[time_column].apply(lambda x: 1 if (x >= in_pandemic_start) and (x <= in_pandemic_end) else 0).astype(int)

    return out_df

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

    # rolling after shift(24) to provide the recent historical context for the future rows; forward-fill NaN to avoid rolling mean becoming NaN 
    out_df['EnergyDemand_rolling_mean_24h'] = out_df['EnergyDemand'].shift(24).rolling(24).mean()   # daily pattern
    out_df['EnergyDemand_rolling_mean_168h'] = out_df['EnergyDemand'].shift(24).rolling(168).mean() # weekly pattern
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

import time
import requests
import pandas as pd 

# TODO consider adding back 'weathercode'
weather_variables = ['apparent_temperature', 'rain', 'snowfall', 'wind_speed_10m', 'shortwave_radiation']  # temperature_2m dropped: high correlation with apparent_temperature (see notebook 02 EDA)

# get latitude and longitude of German cities: Berlin, Hamburg, München, Köln, Frankfurt
selected_cities = {  
    'Berlin': {'latitude': 52.5200, 'longitude': 13.4050},
    'Hamburg': {'latitude': 53.5511, 'longitude': 9.9937},
    'München': {'latitude': 48.1351, 'longitude': 11.5820},
    'Köln': {'latitude': 50.9375, 'longitude': 6.9603},
    'Frankfurt': {'latitude': 50.1109, 'longitude': 8.6821}
}

start_date = "2019-01-01" # Kaggle dataset starts from 2019-01-01
end_date = "2025-09-30" # Kaggle dataset ends at 2025-09-30

def fetch_weather_data_for_cities(in_start_date=start_date, 
                                  in_end_date=end_date, 
                                  in_selected_cities=selected_cities,
                                  in_weather_variables=weather_variables):
    '''
    Fetch weather data from open-meteo archive API for the selected cities 
    and return a dictionary of city name to weather DataFrame.
    '''
    # Fetch 1 day before in_start_date in UTC so Berlin midnight (00:00+02:00 = 22:00 UTC prior day) is included
    api_start_date = (pd.to_datetime(in_start_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    berlin_clip_start = pd.Timestamp(in_start_date, tz="Europe/Berlin")

    weather_city_dict = {}
    for city, coords in in_selected_cities.items():
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['latitude']}&longitude={coords['longitude']}&start_date={api_start_date}&end_date={in_end_date}&hourly={','.join(in_weather_variables)}&timezone=UTC"
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                weather_data = response.json()
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise
                time.sleep(5)
        df_weather_city = pd.DataFrame(weather_data['hourly'])

        # UTC has no DST ambiguity — safe to localize then convert to Berlin
        df_weather_city['time'] = pd.to_datetime(df_weather_city['time'], utc=True).dt.tz_convert("Europe/Berlin").dt.as_unit('s')
        # Clip to requested start in Berlin time (drop the extra day fetched for UTC offset coverage)
        df_weather_city = df_weather_city[df_weather_city['time'] >= berlin_clip_start].reset_index(drop=True)
        #print(f"weather for {city}: {len(df_weather_city)} rows")
        #print(df_weather_city.head(3))
        weather_city_dict.update({city:df_weather_city})
        time.sleep(1)  # sleep for 1 second to avoid hitting API rate limits
    return weather_city_dict

city_population = {
    'Berlin': 3644826, 
    'Hamburg': 1841179, 
    'München': 1471508,     
    'Köln': 1085664, 
    'Frankfurt': 753056 
}

raw_tmp_path = "../data/raw/tmp/"

# calculate the weight of the cities based on their population size and use it to create a weighted average of the weather variables for Germany
def merge_weather_data_with_city_weights(in_weather_city_dict, 
                                         in_city_population=city_population, 
                                         in_weather_variables=weather_variables):
    '''
    Merge the weather data for the selected cities into a single DataFrame for Germany, 
    using population weights to calculate a weighted average of the weather variables.
    '''
    total_population = sum(in_city_population.values())
    df_weather_germany = pd.DataFrame() 
    for city, df_city in in_weather_city_dict.items():
        weight = in_city_population[city] / total_population
        df_city_weighted = df_city.copy()
        for var in in_weather_variables:
            df_city_weighted[var] = df_city[var] * weight
        if df_weather_germany.empty:
            df_weather_germany = df_city_weighted
        else:
            df_weather_germany[in_weather_variables] += df_city_weighted[in_weather_variables]

    return df_weather_germany

# feature engineering: create new features based on existing ones, such as rolling averages, lagged variables, or interaction terms

base_temperature_heating = 18  # base temperature for heating degree days
base_temperature_cooling = 25  # base temperature for cooling degree days

def create_weather_features(in_df, 
                    in_base_temperature_heating=base_temperature_heating, 
                    in_base_temperature_cooling=base_temperature_cooling):
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

# fetch weather data for the selected cities, merge it with population weights to get a Germany-wide weather dataset, and save it to the processed data folder
def prepare_weather_data(in_start_date,
                        in_end_date, 
                        in_selected_cities=selected_cities,
                        in_weather_variables=weather_variables, 
                        in_city_population=city_population):
    '''
    Prepare weather data for modeling: fetch weather data for the selected cities, 
    merge it with population weights to get a Germany-wide weather dataset, and save it to the processed data folder.
    '''
    weather_city_dict = fetch_weather_data_for_cities(in_start_date, in_end_date)
    out_df = merge_weather_data_with_city_weights(weather_city_dict, in_city_population, in_weather_variables)
    out_df = rename_time_column(out_df)
    out_df = out_df.sort_values('time').reset_index(drop=True)
    out_df = create_weather_features(out_df)
 
    return out_df

# ============ prepare weather forecast data ============

def fetch_weather_forecast_for_cities(
        in_selected_cities=selected_cities,
        in_weather_variables=weather_variables,
        forecast_days: int = 2): # forecast_days is set to 2 by default, otherwise only the same day forecast is fetched
    '''
    Fetch hourly weather forecast from open-meteo forecast API for the selected cities.
    forecast_days: 1-16 (free tier max is 16)
    '''
    out_weather_city_dict = {}
    for city, coords in in_selected_cities.items():
        # past_days=1 ensures Berlin midnight (= UTC-2 the day before) is included.
        # After converting UTC→Berlin we clip to today's Berlin 00:00 so the series
        # always starts at midnight, mirroring the archive API behaviour.
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords['latitude']}&longitude={coords['longitude']}"
            f"&hourly={','.join(in_weather_variables)}"
            f"&forecast_days={forecast_days}"
            f"&past_days=1"
            f"&timezone=UTC"
        )
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        '''
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30)
                print('open-meteo API response status code:', response.status_code)
                response.raise_for_status()
                weather_data = response.json()
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise
                time.sleep(5)
        '''
        df_weather_city = pd.DataFrame(weather_data['hourly'])
        # UTC has no DST ambiguity — safe to localize then convert to Berlin
        df_weather_city['time'] = (
            pd.to_datetime(df_weather_city['time'], utc=True)
            .dt.tz_convert("Europe/Berlin")
            .dt.as_unit('s')
        )
        # Clip to today's Berlin midnight so the series starts at 00:00+02:00
        today_berlin_midnight = pd.Timestamp.now(tz="Europe/Berlin").normalize()
        df_weather_city = df_weather_city[
            df_weather_city['time'] >= today_berlin_midnight
        ].reset_index(drop=True)
        out_weather_city_dict[city] = df_weather_city
        time.sleep(1)
    return out_weather_city_dict


def prepare_weather_forecast(
        in_selected_cities=selected_cities,
        in_weather_variables=weather_variables,
        in_city_population=city_population,
        forecast_days: int = 2):
    '''
    Prepare forecast weather data: fetch forecast for cities, merge with 
    population weights, and create weather features.
    forecast_days=2 ensures tomorrow is always fully covered regardless of
    the current UTC hour when the API is called.
    '''
    weather_city_dict = fetch_weather_forecast_for_cities(forecast_days=forecast_days)
    out_df = merge_weather_data_with_city_weights(weather_city_dict)
    out_df = rename_time_column(out_df)
    
    return out_df


def prepare_weather_for_prediction(prediction_date, lookback_days=2, forecast_days=3):
    '''
    Fetch archive + forecast weather, combine them, and apply create_weather_features.
    lookback_days of archive data provides the lag/rolling context needed for tomorrow's
    weather features (apparent_temperature_lag_24h, rolling_mean_24h, etc.).
    The model was trained with these features; without them predictions are unreliable.
    '''
    # --- archive: lookback_days before prediction_date (for lag/rolling context) ---
    archive_start = (pd.to_datetime(prediction_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    archive_end   = (pd.to_datetime(prediction_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    archive_city_dict = fetch_weather_data_for_cities(archive_start, archive_end)
    df_archive = merge_weather_data_with_city_weights(archive_city_dict)
    df_archive = rename_time_column(df_archive)

    # --- forecast: covers prediction_date and beyond ---
    fc_city_dict = fetch_weather_forecast_for_cities(forecast_days=forecast_days)
    df_forecast = merge_weather_data_with_city_weights(fc_city_dict)
    df_forecast = rename_time_column(df_forecast)

    # Keep archive rows strictly before prediction_date to avoid overlap with forecast
    pred_start = pd.Timestamp(prediction_date, tz="Europe/Berlin")
    df_archive = df_archive[df_archive['time'] < pred_start].copy()

    # Combine, deduplicate (forecast wins for any overlap), sort
    df_combined = pd.concat([df_archive, df_forecast], ignore_index=True)
    df_combined = (df_combined
                   .sort_values('time')
                   .drop_duplicates(subset=['time'])
                   .reset_index(drop=True))

    # Apply weather feature engineering (same as training path)
    df_combined = create_weather_features(df_combined)

    return df_combined


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

