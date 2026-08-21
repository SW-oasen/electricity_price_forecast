"""Small MLflow adapter used by the model training/evaluation code."""

from __future__ import annotations

from typing import Any


def log_model_evaluation(
    *,
    experiment_name: str,
    run_name: str,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    fold_scores=None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log one reproducible evaluation run and return its MLflow run id.

    MLflow is imported lazily so the numerical helpers remain usable in small
    environments and unit tests that do not enable tracking.
    """
    import mlflow

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_name", model_name)
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        if fold_scores is not None and not fold_scores.empty:
            for row in fold_scores.itertuples(index=False):
                mlflow.log_metrics(
                    {"mae": float(row.mae), "rmse": float(row.rmse), "r2": float(row.r2)},
                    step=int(row.fold),
                )
        return run.info.run_id
