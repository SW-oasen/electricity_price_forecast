"""Frozen-model, leakage-safe daily walk-forward predictions for prices."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.fetch_price_data import build_price_feature_base
from src.fetch_price_data import load_energy_demand_table, load_time_series_data_from_db
from src.demand_upstream import predict_demand_for_price_target_from_db
from src.forecast_protocol import PRICE_WALK_FORWARD_PROTOCOL, PriceWalkForwardProtocol
from src.generation_upstream import predict_generation_for_price_target
from src.price_forecast_weather import inject_archived_price_weather
from src.price_prediction_store import upsert_price_predictions


EVALUATION_MODE = "walk_forward_as_of_d1_1130"
TARGET = "price_de_lu_eur_mwh"


def _align_features(model, features: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_name_"):
        return features.reindex(columns=list(model.feature_name_))
    if hasattr(model, "feature_names_in_"):
        return features.reindex(columns=list(model.feature_names_in_))
    try:
        names = model.get_booster().feature_names
        if names:
            return features.reindex(columns=names)
    except Exception:
        pass
    return features.select_dtypes("number")


def predict_price_target_day(
    model,
    price_data: pd.DataFrame,
    demand_data: pd.DataFrame,
    target_date: str,
    model_name: str,
    demand_ml_forecasts: pd.DataFrame | None = None,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> pd.DataFrame:
    """Predict D from the information available at D-1 11:30 Berlin.

    The raw inputs must already contain externally published demand/PV/wind
    forecasts.  Actuals after the protocol boundary are masked before all lags
    are generated.
    """
    if not protocol.is_evaluation_target(target_date):
        raise ValueError("target date precedes the frozen-model evaluation cutoff")
    target_start = protocol.target_start(target_date)
    target_end = target_start + pd.DateOffset(days=1)
    actual = price_data.copy()
    actual["time"] = pd.to_datetime(actual["time"], utc=True).dt.tz_convert(protocol.timezone)
    actual_target = actual.loc[
        (actual["time"] >= target_start) & (actual["time"] < target_end),
        ["time", TARGET],
    ].rename(columns={TARGET: "actual_eur_mwh"})

    demand_inputs = demand_data.copy()
    demand_inputs["time"] = pd.to_datetime(
        demand_inputs["time"], utc=True
    ).dt.tz_convert(protocol.timezone)
    if demand_ml_forecasts is not None:
        required = {"time", "demand_forecast_ml_mwh"}
        if missing := required - set(demand_ml_forecasts.columns):
            raise ValueError(f"demand_ml_forecasts lacks columns: {sorted(missing)}")
        replacement = demand_ml_forecasts.rename(
            columns={"demand_forecast_ml_mwh": "_ml_forecast"}
        ).copy()
        replacement["time"] = pd.to_datetime(replacement["time"], utc=True).dt.tz_convert(protocol.timezone)
        known_times = pd.to_datetime(demand_inputs["time"], utc=True).dt.tz_convert(protocol.timezone)
        missing_times = replacement.loc[~replacement["time"].isin(known_times), ["time"]]
        if not missing_times.empty:
            additions = missing_times.copy()
            for column in demand_inputs.columns:
                if column != "time":
                    additions[column] = float("nan")
            demand_inputs = pd.concat([demand_inputs, additions[demand_inputs.columns]], ignore_index=True)
        demand_inputs = demand_inputs.merge(replacement, on="time", how="left")
        demand_inputs["smard_forecast_mwh"] = demand_inputs["_ml_forecast"].fillna(
            demand_inputs["smard_forecast_mwh"]
        )
        demand_inputs = demand_inputs.drop(columns="_ml_forecast")

    base = build_price_feature_base(
        price_data.copy(), demand_inputs,
        physical_actual_until_exclusive=protocol.physical_actual_known_until_exclusive(target_date),
        price_known_until_exclusive=protocol.price_known_until_exclusive(target_date),
    )
    target = base.loc[(base["time"] >= target_start) & (base["time"] < target_end)].copy()
    if target.empty:
        raise ValueError(f"No feature rows for target date {target_date}")
    features = _align_features(model, target.drop(columns=["time", TARGET], errors="ignore"))
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Unavailable walk-forward features: {missing}")
    result = pd.DataFrame({
        "target_time": target["time"],
        "as_of_time": protocol.as_of(target_date),
        "model_name": model_name,
        "evaluation_mode": EVALUATION_MODE,
        "prediction_eur_mwh": model.predict(features),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    for column in ("demand_input_mwh", "gen_wind_input_mwh", "gen_pv_input_mwh"):
        if column in target:
            result[column] = target[column].to_numpy()
    return result.merge(actual_target, left_on="target_time", right_on="time", how="left").drop(columns="time")


def predict_and_store_price_target_day(connection, *args, **kwargs) -> pd.DataFrame:
    """Run one target day and atomically persist its hourly predictions."""
    prediction = predict_price_target_day(*args, **kwargs)
    upsert_price_predictions(connection, prediction)
    return prediction


def predict_price_target_day_from_db(
    price_model,
    demand_model,
    connection,
    target_date: str,
    price_model_name: str,
    protocol: PriceWalkForwardProtocol = PRICE_WALK_FORWARD_PROTOCOL,
) -> pd.DataFrame:
    """Run the complete frozen-model price forecast from persisted as-of inputs."""
    price_raw = load_time_series_data_from_db().reset_index()
    price_raw = inject_archived_price_weather(price_raw, connection, target_date, protocol)
    generation_forecast = predict_generation_for_price_target(price_raw, target_date, protocol)
    generation_forecast["time"] = pd.to_datetime(generation_forecast["time"], utc=True).dt.tz_convert(protocol.timezone)
    price_raw["time"] = pd.to_datetime(price_raw["time"], utc=True).dt.tz_convert(protocol.timezone)
    price_raw = price_raw.merge(generation_forecast, on="time", how="left")
    generation_mask = price_raw["generation_ml_pv_mwh"].notna()
    # The frozen price model has a direct PV feature.  Replace D-1/D actuals
    # with the upstream forecast before feature construction, in both modes.
    price_raw.loc[generation_mask, "gen_pv_mwh"] = price_raw.loc[
        generation_mask, "generation_ml_pv_mwh"
    ]
    demand_raw = load_energy_demand_table()
    demand_forecast = predict_demand_for_price_target_from_db(
        demand_model, connection, target_date, protocol
    )
    return predict_price_target_day(
        price_model, price_raw, demand_raw, target_date, price_model_name,
        demand_ml_forecasts=demand_forecast, protocol=protocol,
    )
