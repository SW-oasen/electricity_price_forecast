from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from util.openmeteo_client import OpenMeteoClient


def build_yearly_weights(capacity_csv: Path, technology_prefix: str) -> dict[int, dict[str, float]]:
    """Build normalized cluster weights by year from yearly capacity CSV."""
    if not capacity_csv.exists():
        raise FileNotFoundError(f"missing cluster yearly capacity file: {capacity_csv}")

    df = pd.read_csv(capacity_csv)
    required = {"year", "cluster_id", "capacity_kw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {capacity_csv}: {sorted(missing)}")

    weights_by_year: dict[int, dict[str, float]] = {}
    for year, df_year in df.groupby("year"):
        total_capacity = float(df_year["capacity_kw"].sum())
        if total_capacity <= 0:
            continue
        weights_by_year[int(year)] = {
            f"{technology_prefix}_cluster_{int(row.cluster_id)}": float(row.capacity_kw) / total_capacity
            for row in df_year.itertuples(index=False)
        }
    return weights_by_year


def fetch_weighted_weather_for_technology(
    technology_name: str,
    locations: Mapping[str, Mapping[str, float]],
    weather_variables: list[str],
    weights_by_year: dict[int, dict[str, float]],
    start_date: str,
    end_date: str,
    selected_cities: Mapping[str, Mapping[str, float]],
    city_population: Mapping[str, int],
    city_sleep: float = 0.2,
) -> pd.DataFrame:
    """Fetch weighted archive weather, applying yearly capacity weights per calendar year."""
    client = OpenMeteoClient(
        cities=selected_cities,
        city_population=city_population,
        weather_variables=weather_variables,
        city_sleep=city_sleep,
    )

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        raise ValueError("end_date must be >= start_date")

    frames: list[pd.DataFrame] = []
    for year in range(start_ts.year, end_ts.year + 1):
        year_weights = weights_by_year.get(year)
        if year_weights is None:
            continue

        chunk_start = max(start_ts, pd.Timestamp(f"{year}-01-01"))
        chunk_end = min(end_ts, pd.Timestamp(f"{year}-12-31"))
        if chunk_end < chunk_start:
            continue

        # Filter locations and weights to match each other exactly
        # (prevents ValueError in OpenMeteoClient if some clusters have 0 capacity in early years
        # or if config doesn't match CSV exactly)
        common_keys = set(locations.keys()) & set(year_weights.keys())
        relevant_locations = {k: locations[k] for k in common_keys}
        relevant_weights = {k: year_weights[k] for k in common_keys}

        if not relevant_locations:
            continue

        df_chunk = client.fetch_archive_weighted_locations(
            locations=relevant_locations,
            location_weights=relevant_weights,
            start_date=chunk_start.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
            weather_variables=weather_variables,
        )
        frames.append(df_chunk)

    if not frames:
        raise ValueError(f"No weather data fetched for {technology_name} in selected range")

    df_out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return df_out


def aggregate_weighted_wind_vector_features(
    location_dict: Mapping[str, pd.DataFrame],
    location_weights: Mapping[str, float],
    speed_col: str = "wind_speed_100m",
    direction_col: str = "wind_direction_100m",
    time_col: str = "time",
    direction_convention: str = "from",
    add_power_features: bool = True,
    power_exponents: tuple[int, ...] = (2, 3),
    clip_speed_min: float | None = 0.0,
    clip_speed_max: float | None = None,
) -> pd.DataFrame:
    """
    Aggregate wind speed/direction across locations using vector components.

    This avoids invalid arithmetic means for circular direction data
    (e.g., 359 deg and 1 deg).

    Parameters
    ----------
    location_dict : Mapping[str, pd.DataFrame]
        Mapping location name -> DataFrame containing time, speed, direction.
    location_weights : Mapping[str, float]
        Raw or normalized positive weights keyed by location name.
    speed_col : str
        Wind speed column name in each location DataFrame.
    direction_col : str
        Wind direction column name (degrees).
    time_col : str
        Timestamp column name.
    direction_convention : str
        "from" (meteorological) or "to".
    add_power_features : bool
        Whether to add speed power features from vector speed.
    power_exponents : tuple[int, ...]
        Exponents applied to vector speed (for example (2, 3)).
    clip_speed_min : float | None
        Optional lower clipping bound before power features.
    clip_speed_max : float | None
        Optional upper clipping bound before power features.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time
        - aggregated u/v components
        - vector-reconstructed speed and direction
        - optional speed power features
    """
    if not location_dict:
        raise ValueError("location_dict must not be empty")

    missing_weights = set(location_dict.keys()) - set(location_weights.keys())
    if missing_weights:
        raise ValueError(f"Missing weights for locations: {sorted(missing_weights)}")

    total_weight = float(sum(float(location_weights[name]) for name in location_dict.keys()))
    if total_weight <= 0:
        raise ValueError("location_weights must sum to a positive value")

    if direction_convention not in {"from", "to"}:
        raise ValueError("direction_convention must be 'from' or 'to'")

    normalized_weights = {
        name: float(location_weights[name]) / total_weight
        for name in location_dict.keys()
    }

    first_name = next(iter(location_dict))
    base_df = location_dict[first_name]
    required = {time_col, speed_col, direction_col}
    missing_cols = required - set(base_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for {first_name}: {sorted(missing_cols)}")

    base_time = base_df[time_col].reset_index(drop=True)
    n_rows = len(base_df)
    u_agg = np.zeros(n_rows, dtype=float)
    v_agg = np.zeros(n_rows, dtype=float)

    for name, df_loc in location_dict.items():
        missing_cols = required - set(df_loc.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for {name}: {sorted(missing_cols)}")

        if len(df_loc) != n_rows:
            raise ValueError("All location DataFrames must have the same row count")

        loc_time = df_loc[time_col].reset_index(drop=True)
        if not base_time.equals(loc_time):
            raise ValueError("All location DataFrames must share the same time index and order")

        speed = pd.to_numeric(df_loc[speed_col], errors="coerce").to_numpy(dtype=float)
        theta = np.deg2rad(pd.to_numeric(df_loc[direction_col], errors="coerce").to_numpy(dtype=float))

        if np.isnan(speed).any() or np.isnan(theta).any():
            raise ValueError(f"NaN found in speed/direction values for location: {name}")

        if direction_convention == "from":
            u = -speed * np.sin(theta)
            v = -speed * np.cos(theta)
        else:
            u = speed * np.sin(theta)
            v = speed * np.cos(theta)

        w = normalized_weights[name]
        u_agg += w * u
        v_agg += w * v

    speed_vector = np.sqrt(u_agg**2 + v_agg**2)

    if direction_convention == "from":
        direction_rad = np.arctan2(-u_agg, -v_agg)
    else:
        direction_rad = np.arctan2(u_agg, v_agg)
    direction_vector = (np.rad2deg(direction_rad) + 360.0) % 360.0

    speed_u_col = f"{speed_col}_u"
    speed_v_col = f"{speed_col}_v"
    speed_vec_col = f"{speed_col}_vector"
    dir_vec_col = f"{direction_col}_vector"

    out = pd.DataFrame(
        {
            time_col: base_time,
            speed_u_col: u_agg,
            speed_v_col: v_agg,
            speed_vec_col: speed_vector,
            dir_vec_col: direction_vector,
        }
    )

    if add_power_features and power_exponents:
        speed_for_power = out[speed_vec_col]
        if clip_speed_min is not None or clip_speed_max is not None:
            speed_for_power = speed_for_power.clip(lower=clip_speed_min, upper=clip_speed_max)

        for exponent in power_exponents:
            exp_int = int(exponent)
            if exp_int <= 0:
                raise ValueError("power_exponents must contain positive integers")
            out[f"{speed_vec_col}_pow{exp_int}"] = speed_for_power.pow(exp_int)

    return out
