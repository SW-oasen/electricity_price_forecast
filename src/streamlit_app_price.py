"""
Streamlit web app — Germany hourly electricity price forecast.

Run from project root:
    streamlit run src/streamlit_app_price.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import streamlit as st

# make direct execution from src/ work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.etl_price import update_price_database
    from src.etl_price import create_price_tables, seed_series_catalog
    from src.etl_demand import update_demand_database
    from src.config import DATABASE_PATH, DEMAND_UPSTREAM_MODEL_PATH
    from src.historical_price_weather import fetch_and_store_demand_weather_for_target, fetch_and_store_weather_for_target
    from src.price_walk_forward import predict_price_target_day_from_db
    from src.fetch_price_data import (
        build_price_feature_base,
        load_energy_demand_table,
        load_time_series_data_from_db,
        prepare_data_for_price_prediction_operational,
        prepare_price_model_dataset,
    )
    from src.train_predict_model import load_model_from_pickle
except ImportError:
    from etl_price import update_price_database
    from etl_price import create_price_tables, seed_series_catalog
    from etl_demand import update_demand_database
    from config import DATABASE_PATH, DEMAND_UPSTREAM_MODEL_PATH
    from historical_price_weather import fetch_and_store_demand_weather_for_target, fetch_and_store_weather_for_target
    from price_walk_forward import predict_price_target_day_from_db
    from fetch_price_data import (
        build_price_feature_base,
        load_energy_demand_table,
        load_time_series_data_from_db,
        prepare_data_for_price_prediction_operational,
        prepare_price_model_dataset,
    )
    from train_predict_model import load_model_from_pickle

MAX_RANGE_DAYS = 365
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
PRICE_MODEL_PATH = MODEL_DIR / "production" / "price_xgboost.pkl"
if not PRICE_MODEL_PATH.exists():
    PRICE_MODEL_PATH = MODEL_DIR / "price_xgb_model.pkl"

PRICE_TARGET_COL = "price_de_lu_eur_mwh"
PRICE_DISPLAY_COL = "Price (EUR/MWh)"
MODEL_NAME = "XGBoost"


st.set_page_config(
    page_title="Strompreisprognose DE",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource(show_spinner="Preis-Datenbank wird aktualisiert …")
def init_db() -> bool:
    update_price_database()
    update_demand_database()
    return True


@st.cache_resource
def load_price_model() -> object:
    if not PRICE_MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost-Preismodell nicht gefunden: {PRICE_MODEL_PATH}")
    return load_model_from_pickle(PRICE_MODEL_PATH)


@st.cache_resource
def load_frozen_demand_model() -> object:
    return load_model_from_pickle(DEMAND_UPSTREAM_MODEL_PATH)


def to_berlin_naive(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True)
    return ts.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)


def align_features(model: object, df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["time", PRICE_TARGET_COL], errors="ignore")
    if hasattr(model, "feature_name_"):
        return X.reindex(columns=list(model.feature_name_))
    if hasattr(model, "feature_names_in_"):
        return X.reindex(columns=list(model.feature_names_in_))
    try:
        booster_features = model.get_booster().feature_names
        if booster_features:
            return X.reindex(columns=list(booster_features))
    except Exception:
        pass
    return X.select_dtypes("number")


def predict_df(model: object, df_features: pd.DataFrame, pred_col: str = "Prediction by XGBoost") -> pd.DataFrame:
    out = df_features[["time"]].copy()
    X = align_features(model, df_features)
    out[pred_col] = model.predict(X)
    return out


def load_actual_context(start_date: date, end_date: date) -> pd.DataFrame:
    """Load historical context from DB for the historical tab."""
    df_ts = load_time_series_data_from_db().reset_index()
    df_ts["time"] = pd.to_datetime(df_ts["time"], utc=True)

    df_dem = load_energy_demand_table()
    df_dem["time"] = pd.to_datetime(df_dem["time"], utc=True)
    df_dem = df_dem.rename(columns={"smard_forecast_mwh": "demand_forecast_mwh"})

    df_base = build_price_feature_base(df_ts, df_dem)
    start_ts = pd.Timestamp(start_date, tz="Europe/Berlin")
    end_ts = pd.Timestamp(end_date + timedelta(days=1), tz="Europe/Berlin")
    mask = (df_base["time"] >= start_ts) & (df_base["time"] < end_ts)
    return df_base.loc[mask].copy().reset_index(drop=True)


def load_price_history_berlin() -> pd.DataFrame:
    df_price = load_time_series_data_from_db().reset_index()
    df_price["time_berlin"] = pd.to_datetime(df_price["time"], utc=True).dt.tz_convert("Europe/Berlin")
    return df_price


def plot_tomorrow_forecast_notebook_style(
    pred_price_xgb,
    df_price_daybeforeyesterday: pd.DataFrame,
    df_price_last7days: pd.Series,
    residual_load_forecast: pd.Series,
    tomorrow: date,
) -> None:
    """Match the notebook plot: price forecast + day-before-yesterday + 7-day average + residual load."""
    time_range = range(24)

    y_candidates = [
        pd.Series(pred_price_xgb),
        pd.to_numeric(df_price_daybeforeyesterday.get(PRICE_TARGET_COL), errors="coerce"),
        pd.to_numeric(df_price_last7days, errors="coerce"),
    ]
    predicted_max = max(float(s.max()) for s in y_candidates if s is not None and s.notna().any())
    predicted_min = min(float(s.min()) for s in y_candidates if s is not None and s.notna().any())
    predicted_min = min(predicted_min, 0)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time_range, pred_price_xgb, linestyle="--", color="orange", label="Prediction by XGBoost")

    if not df_price_daybeforeyesterday.empty and PRICE_TARGET_COL in df_price_daybeforeyesterday.columns:
        y_day_before = df_price_daybeforeyesterday[PRICE_TARGET_COL].to_numpy()
        ax.plot(time_range, y_day_before, color="steelblue", label="Day Before Yesterday")

    ax.plot(
        time_range,
        df_price_last7days.reindex(time_range).to_numpy(),
        color="skyblue",
        label="Last 7 days average",
    )

    ax_twin = ax.twinx()
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (EUR/MWh)", color="darkgreen")
    ax.set_ylim(predicted_min * 1.1, predicted_max * 1.1)
    ax.legend(loc="upper center")
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.grid(True, alpha=0.3)

    ax_twin.plot(
        time_range,
        residual_load_forecast.to_numpy(),
        color="silver",
        label="Residual Load Forecast MWH",
    )
    ax_twin.set_ylabel("Residual Load Forecast (MWh)", color="black")
    ax_twin.legend(loc="lower left")

    plt.suptitle(f"Predicted Electricity Price for {tomorrow}")
    plt.tight_layout()
    st.pyplot(fig)


def render_metrics(df: pd.DataFrame, actual_col: str, pred_col: str) -> None:
    if actual_col not in df.columns or pred_col not in df.columns:
        return
    cmp_df = df[[actual_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if cmp_df.empty:
        return

    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(cmp_df[actual_col], cmp_df[pred_col])
    rmse = np.sqrt(mean_squared_error(cmp_df[actual_col], cmp_df[pred_col]))
    r2 = r2_score(cmp_df[actual_col], cmp_df[pred_col])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f} €/MWh")
    col2.metric("RMSE", f"{rmse:.2f} €/MWh")
    col3.metric("R²", f"{r2:.2f}")
    col4.metric("Datenpunkte", f"{len(cmp_df):,}")


init_db()
price_model = load_price_model()

st.title("⚡ Strompreisprognose Deutschland")
st.markdown(
    "Die App verwendet das **XGBoost-Preismodell**. Die Morgenansicht zeigt die Prognose, "
    "den Preis von vorgestern, den 7-Tage-Durchschnitt und die prognostizierte Residuallast."
)

tab_future, tab_hist = st.tabs(["Vorhersage für morgen", "Historische Vorhersage"])

with tab_future:
    today = pd.Timestamp.now(tz="Europe/Berlin").date()
    tomorrow = today + timedelta(days=1)
    daybeforeyesterday = today - timedelta(days=2)

    st.markdown(f"**Modell:** {MODEL_NAME}")
    st.markdown(f"**Vorgestern:** {daybeforeyesterday.isoformat()}")
    st.markdown(f"**Heute:** {today.isoformat()}")
    st.markdown(f"**Vorhersagetag:** {tomorrow.isoformat()}")

    if st.button("Vorhersage berechnen", type="primary", key="btn_future_price"):
        with st.spinner("Features für morgen werden vorbereitet …"):
            conn = create_price_tables(DATABASE_PATH)
            try:
                seed_series_catalog(conn)
                target_str = tomorrow.isoformat()
                fetch_and_store_weather_for_target(conn, target_str)
                fetch_and_store_demand_weather_for_target(conn, target_str)
                df_tomorrow_feat = predict_price_target_day_from_db(
                    price_model,
                    load_frozen_demand_model(),
                    conn,
                    target_str,
                    MODEL_NAME,
                ).rename(columns={"target_time": "time"})
            finally:
                conn.close()
        if df_tomorrow_feat.empty:
            st.error("Keine Features für morgen erzeugt.")
            st.stop()

        df_tomorrow_feat["time"] = pd.to_datetime(df_tomorrow_feat["time"], utc=True).dt.tz_convert("Europe/Berlin")
        df_tomorrow_feat["residual_load_forecast"] = (
            df_tomorrow_feat["gen_pv_input_mwh"]
            + df_tomorrow_feat["gen_wind_input_mwh"]
            - df_tomorrow_feat["demand_input_mwh"]
        )

        pred_price_xgb = df_tomorrow_feat["prediction_eur_mwh"].to_numpy()

        df_price = load_price_history_berlin()
        df_price_daybeforeyesterday = (
            df_price
            .loc[df_price["time_berlin"].dt.date == daybeforeyesterday]
            .sort_values("time_berlin")
        )
        df_price_last7days = (
            df_price
            .loc[
                (df_price["time_berlin"].dt.date >= (today - timedelta(days=7)))
                & (df_price["time_berlin"].dt.date < today)
            ]
            .groupby(df_price["time_berlin"].dt.hour)[PRICE_TARGET_COL]
            .mean()
        )

        st.success("Vorhersage abgeschlossen (XGBoost).")
        plot_tomorrow_forecast_notebook_style(
            pred_price_xgb=pred_price_xgb,
            df_price_daybeforeyesterday=df_price_daybeforeyesterday,
            df_price_last7days=df_price_last7days,
            residual_load_forecast=df_tomorrow_feat["residual_load_forecast"],
            tomorrow=tomorrow,
        )

        table = pd.DataFrame({
            "Stunde": list(range(24)),
            "Prediction by XGBoost": pred_price_xgb,
            "Day Before Yesterday": df_price_daybeforeyesterday[PRICE_TARGET_COL].to_numpy()
            if not df_price_daybeforeyesterday.empty and len(df_price_daybeforeyesterday) == 24 else pd.NA,
            "Last 7 days average": df_price_last7days.reindex(range(24)).to_numpy(),
            "Residual Load Forecast MWH": df_tomorrow_feat["residual_load_forecast"].to_numpy(),
        })
        st.dataframe(table, use_container_width=True)

with tab_hist:
    _default_to = date.today() - timedelta(days=1)
    _default_from = _default_to - timedelta(days=6)
    _min_date = date(2019, 1, 8)
    _max_date = date.today() - timedelta(days=1)

    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Von:", value=_default_from, min_value=_min_date, max_value=_max_date, key="hist_from_price")
    with col2:
        date_to = st.date_input("Bis:", value=_default_to, min_value=_min_date, max_value=_max_date, key="hist_to_price")

    st.markdown(f"**Modell:** {MODEL_NAME}")

    delta_days = (date_to - date_from).days
    if delta_days < 0:
        st.error('⚠ „Bis"-Datum muss nach dem „Von"-Datum liegen.')
    elif delta_days > MAX_RANGE_DAYS:
        st.warning(f"⚠ Gewählter Zeitraum: {delta_days} Tage — Maximum sind {MAX_RANGE_DAYS} Tage.")
    else:
        st.success(f"Zeitraum: {delta_days + 1} Tag(e) ✓")

    if st.button("Historische Vorhersage berechnen", type="primary", key="btn_hist_price"):
        from_str, to_str = str(date_from), str(date_to)
        if delta_days > 31 or delta_days < 0:
            st.error("Bitte einen gueltigen Zeitraum von maximal 31 Tagen waehlen.")
            st.stop()

        frozen_demand_model = load_frozen_demand_model()
        days = pd.date_range(date_from, date_to, freq="D")
        predictions = []
        progress = st.progress(0, text="Walk-forward wird vorbereitet ...")
        conn = None
        try:
            conn = create_price_tables(DATABASE_PATH)
            seed_series_catalog(conn)
            for position, target_day in enumerate(days, start=1):
                day = target_day.strftime("%Y-%m-%d")
                progress.progress((position - 1) / len(days), text=f"Walk-forward fuer {day} ...")
                fetch_and_store_weather_for_target(conn, day)
                fetch_and_store_demand_weather_for_target(conn, day)
                predictions.append(predict_price_target_day_from_db(
                    price_model, frozen_demand_model, conn, day, MODEL_NAME
                ))
            progress.progress(1.0, text="Walk-forward abgeschlossen")
        finally:
            if conn is not None:
                conn.close()
        df_plot = pd.concat(predictions, ignore_index=True).rename(columns={
            "target_time": "time", "actual_eur_mwh": PRICE_TARGET_COL,
            "prediction_eur_mwh": "ML Prediction",
        })
        st.success("Historische Walk-forward-Vorhersage abgeschlossen (eingefrorene Modelle).")
        fig, ax = plt.subplots(figsize=(14, 5))
        x = to_berlin_naive(df_plot["time"])
        ax.plot(x, df_plot[PRICE_TARGET_COL], linewidth=1.5, color="steelblue", label="Echter Strompreis")
        ax.plot(x, df_plot["ML Prediction"], linewidth=1.5, linestyle="--", color="orange", label="ML-Vorhersage (XGBoost)")
        ax.set_title(f"Historische Strompreisvorhersage — {from_str} bis {to_str}")
        ax.set_xlabel("Zeit (Europe/Berlin)")
        ax.set_ylabel(PRICE_DISPLAY_COL)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)

        render_metrics(df_plot, PRICE_TARGET_COL, "ML Prediction")

        table = df_plot.copy()
        table["Zeit (Berlin)"] = to_berlin_naive(table["time"]).dt.strftime("%Y-%m-%d %H:%M")
        display_cols = ["Zeit (Berlin)", PRICE_TARGET_COL, "ML Prediction"]
        st.dataframe(table[[c for c in display_cols if c in table.columns]], use_container_width=True)

st.markdown("---")
st.caption("Strompreisprognose App • Modell: XGBoost • Datenquellen: SQLite DB, SMARD, Open-Meteo")
