"""Recursive frozen PV/wind forecasts shared by backtest and live price runs."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_protocol import PRICE_WALK_FORWARD_PROTOCOL, PriceWalkForwardProtocol


ROOT = Path(__file__).resolve().parents[1]


def _model_features(model) -> list[str]:
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(model.get_booster().feature_names)


def _predict_one(
    data: pd.DataFrame, model, target_col: str, weather_col: str,
    target_date: str, protocol: PriceWalkForwardProtocol,
) -> pd.DataFrame:
    target_start = protocol.target_start(target_date)
    origin = target_start - pd.DateOffset(days=1)
    end = target_start + pd.DateOffset(days=1)
    data = data.sort_values("time").copy()
    data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_convert(protocol.timezone)
    names = _model_features(model)
    weather_columns = [
        column for column in names
        if column.startswith(("pv_weather_", "wind_weather_"))
        and "_lag_" not in column and "_rolling_" not in column
    ]
    for column in weather_columns:
        weather_values = pd.to_numeric(data[column], errors="coerce")
        for lag in (24, 168):
            data[f"{column}_lag_{lag}h"] = weather_values.shift(lag)
            data[f"{column}_rolling_mean_{lag}h"] = weather_values.shift(1).rolling(lag).mean()
    weather = pd.to_numeric(data[weather_col], errors="coerce")
    if weather_col == "wind_weather_wind_speed_100m":
        data["wind_speed_clipped"] = weather.clip(3.0, 25.0)
        rated = weather.clip(upper=13.0)
        data["wind_speed_pow2"] = rated ** 2
        data["wind_speed_pow3"] = rated ** 3
    data["year"] = data.time.dt.year
    data["hour"] = data.time.dt.hour
    data["month"] = data.time.dt.month

    history = pd.to_numeric(data.loc[data.time < origin, target_col], errors="coerce").dropna().tolist()
    if len(history) < 168:
        raise ValueError(f"At least 168 prior {target_col} observations are required")
    rows = []
    for index, row in data.loc[(data.time >= origin) & (data.time < end)].iterrows():
        values = row.copy()
        values[f"{target_col}_lag_24h"] = history[-24]
        values[f"{target_col}_lag_168h"] = history[-168]
        values[f"{target_col}_rolling_mean_24h"] = float(np.mean(history[-24:]))
        values[f"{target_col}_rolling_mean_168h"] = float(np.mean(history[-168:]))
        frame = values.to_frame().T.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
        if frame.isna().any().any():
            missing = frame.columns[frame.isna().any()].tolist()
            raise ValueError(f"Generation forecast features unavailable: {missing}")
        prediction = float(np.asarray(model.predict(frame)).reshape(-1)[0])
        history.append(prediction)
        rows.append((row["time"], prediction))
    return pd.DataFrame(rows, columns=["time", target_col])


def predict_generation_for_price_target(
    price_data: pd.DataFrame,
    target_date: str,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> pd.DataFrame:
    """Forecast PV and total wind for D-1/D from the frozen generation models."""
    data = price_data.copy()
    data["gen_wind_total_mwh"] = (
        pd.to_numeric(data["gen_wind_onshore_mwh"], errors="coerce")
        + pd.to_numeric(data["gen_wind_offshore_mwh"], errors="coerce")
    )
    with (ROOT / "models" / "pv_lgbm_model.pkl").open("rb") as handle:
        pv_model = pickle.load(handle)
    with (ROOT / "models" / "wind_lgbm_model.pkl").open("rb") as handle:
        wind_model = pickle.load(handle)
    pv = _predict_one(data, pv_model, "gen_pv_mwh", "pv_weather_shortwave_radiation", target_date, protocol)
    wind = _predict_one(data, wind_model, "gen_wind_total_mwh", "wind_weather_wind_speed_100m", target_date, protocol)
    return pv.merge(wind, on="time").rename(columns={
        "gen_pv_mwh": "generation_ml_pv_mwh",
        "gen_wind_total_mwh": "generation_ml_wind_mwh",
    })
