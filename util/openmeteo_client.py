"""
openmeteo_client.py — generic client for the Open-Meteo archive and forecast APIs.

No city list or variable defaults — all domain values must be supplied by the caller
(typically from src/config.py for this project).

Usage example (Germany):
    from util.openmeteo_client import OpenMeteoClient
    from config import SELECTED_CITIES, CITY_POPULATION, WEATHER_VARIABLES

    client = OpenMeteoClient(
        cities=SELECTED_CITIES,
        city_population=CITY_POPULATION,
        weather_variables=WEATHER_VARIABLES,
    )

    df_hist    = client.fetch_archive('2024-01-01', '2024-12-31')
    df_fc      = client.fetch_forecast(forecast_days=3)
    df_pred    = client.prepare_for_prediction('2026-05-27')

API reference: documents/open-meteo_api.md

Returned DataFrames always have:
    time — tz-aware datetime64[s] (Europe/Berlin)
    one column per weather variable (population-weighted aggregate)
"""

from __future__ import annotations

import time
from pathlib import Path
import json
import os
import subprocess

import pandas as pd
import requests


_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"


def _request_json(url: str, params: dict, timeout: int) -> dict:
    """Fetch JSON while retaining platform-native certificate validation."""
    def _curl() -> dict:
        command = ["curl.exe", "--fail", "--silent", "--show-error", "--get", url]
        for key, value in params.items():
            command.extend(["--data-urlencode", f"{key}={value}"])
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
        return json.loads(result.stdout)

    # In this Python 3.14 Windows build, requests/OpenSSL aborts the process on
    # the local certificate chain before an SSLError can be raised.  curl.exe
    # uses the Windows trust store and keeps verification enabled.
    if os.name == "nt":
        return _curl()
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.SSLError:
        return _curl()


class OpenMeteoClient:
    """
    Fetch and population-weight hourly weather data from Open-Meteo for a set
    of cities.

    Parameters
    ----------
    cities : dict
        Mapping of city name → {'latitude': float, 'longitude': float}.
    city_population : dict
        Mapping of city name → population (int).  Used to compute weights.
    weather_variables : list[str]
        Open-Meteo variable names to request (e.g. 'apparent_temperature').
    city_sleep : float
        Seconds to sleep between per-city requests (default 1 s).
    timeout : int
        HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        cities: dict,
        city_population: dict,
        weather_variables: list[str],
        city_sleep: float = 1.0,
        timeout: int = 30,
    ) -> None:
        if set(cities) != set(city_population):
            raise ValueError("cities and city_population must have the same keys.")
        self.cities            = cities
        self.city_population   = city_population
        self.weather_variables = list(weather_variables)
        self.city_sleep        = city_sleep
        self.timeout           = timeout

        total = sum(city_population.values())
        self._weights = {city: pop / total for city, pop in city_population.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_cities(self, city_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Population-weight the per-city DataFrames into one aggregate DataFrame.
        The 'time' column is taken from the first city; numeric columns are
        weighted-summed across cities.
        """
        return self._merge_weighted(
            location_dict=city_dict,
            weights=self._weights,
            variables=self.weather_variables,
        )

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        """Normalize arbitrary positive weights to sum to 1.0."""
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        return {key: float(value) / total for key, value in weights.items()}

    def _merge_weighted(
        self,
        location_dict: dict[str, pd.DataFrame],
        weights: dict[str, float],
        variables: list[str],
    ) -> pd.DataFrame:
        """
        Weighted merge for arbitrary locations.

        Parameters
        ----------
        location_dict : dict[str, pd.DataFrame]
            Mapping location name -> hourly DataFrame.
        weights : dict[str, float]
            Normalized weights keyed by location name.
        variables : list[str]
            Weather variables to weighted-sum.
        """
        out = pd.DataFrame()
        for location, df_loc in location_dict.items():
            w = weights[location]
            df_w = df_loc.copy()
            for var in variables:
                df_w[var] = df_loc[var] * w
            if out.empty:
                out = df_w
            else:
                out[variables] = out[variables].values + df_w[variables].values
        return out

    def _fetch_archive_per_location(
        self,
        locations: dict,
        start_date: str,
        end_date: str,
        variables: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Fetch archive weather for each location and return per-location DataFrames."""
        api_start = (pd.to_datetime(start_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        clip_start = pd.Timestamp(start_date, tz='Europe/Berlin')
        clip_end_exclusive = pd.Timestamp(end_date, tz='Europe/Berlin') + pd.Timedelta(days=1)
        vars_str = ','.join(variables)

        out: dict[str, pd.DataFrame] = {}
        for name, coords in locations.items():
            url = (
                f"{_ARCHIVE_URL}"
                f"?latitude={coords['latitude']}"
                f"&longitude={coords['longitude']}"
                f"&start_date={api_start}"
                f"&end_date={end_date}"
                f"&hourly={vars_str}"
                f"&timezone=UTC"
            )
            data = _request_json(url, {}, self.timeout)

            df_loc = pd.DataFrame(data['hourly'])
            df_loc['time'] = (
                pd.to_datetime(df_loc['time'], utc=True)
                .dt.tz_convert('Europe/Berlin')
                .dt.as_unit('s')
            )
            df_loc = df_loc[
                (df_loc['time'] >= clip_start)
                & (df_loc['time'] < clip_end_exclusive)
            ].reset_index(drop=True)
            out[name] = df_loc
            time.sleep(self.city_sleep)

        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def latest_available_run(
        as_of: pd.Timestamp, availability_delay_hours: int = 6
    ) -> pd.Timestamp:
        """Latest ECMWF six-hour run safely published by ``as_of``."""
        timestamp = pd.Timestamp(as_of)
        if timestamp.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        cutoff = timestamp.tz_convert("UTC") - pd.Timedelta(
            hours=availability_delay_hours
        )
        return cutoff.normalize() + pd.Timedelta(hours=(cutoff.hour // 6) * 6)

    def fetch_single_run(
        self,
        run: pd.Timestamp,
        forecast_days: int = 3,
        model: str = "ecmwf_ifs",
        cache_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Fetch one archived, population-weighted Open-Meteo model run."""
        run = pd.Timestamp(run)
        if run.tzinfo is None:
            raise ValueError("run must be timezone-aware")
        run = run.tz_convert("UTC")
        if run.minute or run.second or run.hour % 6:
            raise ValueError("run must be a 00/06/12/18 UTC model cycle")

        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / model / f"{run.strftime('%Y%m%dT%H%MZ')}_{forecast_days}d.csv"
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                if {"time", *self.weather_variables}.issubset(cached.columns):
                    cached["time"] = pd.to_datetime(cached["time"], utc=True).dt.tz_convert("Europe/Berlin")
                    return cached.sort_values("time").reset_index(drop=True)

        location_frames: dict[str, pd.DataFrame] = {}
        for name, coords in self.cities.items():
            data = _request_json(
                _SINGLE_RUNS_URL,
                {
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                    "hourly": ",".join(self.weather_variables),
                    "models": model,
                    "run": run.strftime("%Y-%m-%dT%H:%M"),
                    "timezone": "UTC",
                    "forecast_days": forecast_days,
                }, self.timeout,
            )
            frame = pd.DataFrame(data["hourly"])
            missing = set(self.weather_variables) - set(frame.columns)
            if missing:
                raise ValueError(f"Single Runs response lacks variables: {sorted(missing)}")
            frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Europe/Berlin")
            location_frames[name] = frame[["time", *self.weather_variables]]
            time.sleep(self.city_sleep)

        result = self._merge_cities(location_frames).sort_values("time").reset_index(drop=True)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.assign(time=result["time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")).to_csv(cache_path, index=False)
        return result

    def fetch_archive(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical (archive) weather from Open-Meteo for all cities,
        population-weight the result, and return a single DataFrame.

        Fetches one extra day before start_date in UTC to ensure Berlin midnight
        (= UTC-2 the prior day) is included, then clips to start_date Berlin time.

        Parameters
        ----------
        start_date : str  'YYYY-MM-DD' (Europe/Berlin local)
        end_date   : str  'YYYY-MM-DD' (Europe/Berlin local)

        Returns
        -------
        DataFrame with columns: ['time'] + weather_variables
        """
        city_dict = self._fetch_archive_per_location(
            locations=self.cities,
            start_date=start_date,
            end_date=end_date,
            variables=self.weather_variables,
        )

        df = self._merge_cities(city_dict)
        return df.sort_values('time').reset_index(drop=True)

    def fetch_archive_weighted_locations(
        self,
        locations: dict,
        location_weights: dict[str, float],
        start_date: str,
        end_date: str,
        weather_variables: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch archive weather for arbitrary locations and return weighted aggregate.

        This supports technology-specific aggregation (e.g. PV cluster weights,
        Wind cluster weights) while reusing the same Open-Meteo fetch flow.
        """
        variables = list(weather_variables) if weather_variables is not None else list(self.weather_variables)
        if set(locations) != set(location_weights):
            raise ValueError("locations and location_weights must have the same keys")

        norm_weights = self._normalize_weights(location_weights)
        location_dict = self._fetch_archive_per_location(
            locations=locations,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
        )

        df = self._merge_weighted(
            location_dict=location_dict,
            weights=norm_weights,
            variables=variables,
        )
        return df.sort_values('time').reset_index(drop=True)

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """
        Fetch hourly weather forecast from Open-Meteo for all cities,
        population-weight the result, and return a single DataFrame.

        Uses past_days=1 to ensure Berlin midnight is included, then clips to
        today's Berlin midnight so the series always starts at 00:00.

        Parameters
        ----------
        forecast_days : int  1–16 (Open-Meteo free tier max is 16).

        Returns
        -------
        DataFrame with columns: ['time'] + weather_variables
        """
        vars_str   = ','.join(self.weather_variables)
        today_midnight = pd.Timestamp.now(tz='Europe/Berlin').normalize()

        city_dict: dict[str, pd.DataFrame] = {}
        for city, coords in self.cities.items():
            url = (
                f"{_FORECAST_URL}"
                f"?latitude={coords['latitude']}"
                f"&longitude={coords['longitude']}"
                f"&hourly={vars_str}"
                f"&forecast_days={forecast_days}"
                f"&past_days=1"
                f"&timezone=UTC"
            )
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    break
                except requests.exceptions.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(5)

            df_city = pd.DataFrame(data['hourly'])
            df_city['time'] = (
                pd.to_datetime(df_city['time'], utc=True)
                .dt.tz_convert('Europe/Berlin')
                .dt.as_unit('s')
            )
            df_city = df_city[df_city['time'] >= today_midnight].reset_index(drop=True)
            city_dict[city] = df_city
            time.sleep(self.city_sleep)

        df = self._merge_cities(city_dict)
        return df.sort_values('time').reset_index(drop=True)

    def prepare_for_prediction(
        self,
        prediction_date: str,
        lookback_days: int = 2,
        forecast_days: int = 3,
    ) -> pd.DataFrame:
        """
        Combine archive (lookback context) and forecast data for a prediction date.

        The archive lookback provides the lag/rolling context rows that the
        feature engineering step (create_weather_features in fetch_demand_data.py)
        needs.  Without it, lag/rolling values for the prediction day would be NaN.

        Parameters
        ----------
        prediction_date : str  'YYYY-MM-DD'
        lookback_days   : int  Days of archive history before prediction_date.
        forecast_days   : int  Days of forecast to include (must cover prediction_date).

        Returns
        -------
        DataFrame covering [prediction_date - lookback_days, prediction_date + (forecast_days-1)]
        with columns: ['time'] + weather_variables
        """
        archive_start = (
            pd.to_datetime(prediction_date) - pd.Timedelta(days=lookback_days)
        ).strftime('%Y-%m-%d')
        archive_end = (
            pd.to_datetime(prediction_date) - pd.Timedelta(days=1)
        ).strftime('%Y-%m-%d')

        df_archive = self.fetch_archive(archive_start, archive_end)

        df_forecast = self.fetch_forecast(forecast_days=forecast_days)

        pred_start = pd.Timestamp(prediction_date, tz='Europe/Berlin')
        df_archive = df_archive[df_archive['time'] < pred_start].copy()

        df_combined = (
            pd.concat([df_archive, df_forecast], ignore_index=True)
            .sort_values('time')
            .drop_duplicates(subset=['time'])   # forecast wins on overlap
            .reset_index(drop=True)
        )
        return df_combined
