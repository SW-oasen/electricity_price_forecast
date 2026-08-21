"""Reproducible price-model training entry point.

The notebook is only a thin manual front end; this module is the CI/CD entry
point and writes versioned candidates plus an MLflow run.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROJECT_ROOT
from src.fetch_price_data import prepare_price_model_dataset
from src.train_predict_model import (
    save_model_to_pickle,
    train_test_split_by_date,
    tune_model_bayesian,
)


def train_price_model(
    model_family: str = "lightgbm",
    train_end: str = "2026-03-31",
    validation_end: str = "2026-04-30",
    experiment_name: str = "electricity-price",
    train_window_years: int | None = None,
):
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    data = prepare_price_model_dataset()
    data["time"] = pd.to_datetime(data["time"], utc=True)
    train, target, features, target_test = train_test_split_by_date(
        data, "time", "price_de_lu_eur_mwh", train_end
    )
    if model_family == "lightgbm":
        model = LGBMRegressor(random_state=42, force_col_wise=True, verbosity=-1)
        search = {"n_estimators": (50, 500), "learning_rate": (0.01, 0.3), "max_depth": (3, 15)}
    elif model_family == "xgboost":
        model = XGBRegressor(random_state=42)
        search = {"n_estimators": (50, 1000), "max_depth": (3, 15), "learning_rate": (0.01, 0.3)}
    else:
        raise ValueError("model_family must be lightgbm or xgboost")

    if train_window_years is not None:
        raise NotImplementedError(
            "Rolling training windows are not enabled yet; use None for expanding training."
        )
    fitted, best_params = tune_model_bayesian(
        model, search, train, target,
        mlflow_experiment=experiment_name,
        mlflow_run_name=f"{model_family}-train-{train_end}",
        mlflow_tags={
            "train_end": train_end,
            "validation_end": validation_end,
            "train_window": "expanding",
        },
    )
    output_dir = PROJECT_ROOT / "models" / "production"
    archive_dir = PROJECT_ROOT / "models" / "archive"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    version = date.today().isoformat()
    versioned = archive_dir / f"price_{model_family}_train_end-{train_end}_created-{version}.pkl"
    production = output_dir / f"price_{model_family}.pkl"
    save_model_to_pickle(fitted, versioned)
    save_model_to_pickle(fitted, production)
    return fitted, best_params, production


def evaluate_price_model_walk_forward(model, connection, demand_model, start: str, end: str, *,
                                      model_family: str = "unknown",
                                      experiment_name: str = "electricity-price",
                                      prepare_weather: bool = True) -> dict[str, float]:
    """Evaluate through the existing leakage-safe day-ahead walk-forward path."""
    from src.price_walk_forward import predict_price_target_day_from_db
    days = pd.date_range(start, end, freq="D", inclusive="left")
    if not len(days):
        raise ValueError(f"No evaluation days in [{start}, {end})")

    evaluation_id = str(uuid4())
    if prepare_weather:
        # The walk-forward path deliberately reads archived as-of weather only.
        # Populate those inputs before prediction; storage is idempotent.
        from src.historical_price_weather import (
            fetch_and_store_demand_weather_for_target,
            fetch_and_store_weather_for_target,
        )

        for day in days:
            target_date = day.date().isoformat()
            fetch_and_store_weather_for_target(
                connection, target_date, evaluation_id=evaluation_id
            )
            fetch_and_store_demand_weather_for_target(
                connection, target_date, evaluation_id=evaluation_id
            )

    predictions = [predict_price_target_day_from_db(
        model, demand_model, connection, day.date().isoformat(), model_family,
        evaluation_id=evaluation_id,
    ) for day in days]
    results = pd.concat(predictions, ignore_index=True).dropna(
        subset=["prediction_eur_mwh", "actual_eur_mwh"]
    )
    if results.empty:
        raise ValueError(f"No scored walk-forward rows in [{start}, {end})")
    actual = results["actual_eur_mwh"]
    predicted = results["prediction_eur_mwh"]
    scores = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "n_test": float(len(results)),
    }
    from src.mlflow_tracking import log_model_evaluation
    from src.price_input_lineage import lineage_summary
    lineage_metrics = lineage_summary(connection, evaluation_id)
    scores.update(lineage_metrics)
    log_model_evaluation(
        experiment_name=experiment_name,
        run_name=f"{model_family}-walk-forward-{start}",
        model_name=model_family,
        params={"evaluation_start": start, "evaluation_end": end},
        metrics=scores,
        tags={
            "evaluation_mode": "walk_forward_as_of_d1_1130",
            "input_lineage_evaluation_id": evaluation_id,
        },
    )
    return scores


if __name__ == "__main__":
    train_price_model()
