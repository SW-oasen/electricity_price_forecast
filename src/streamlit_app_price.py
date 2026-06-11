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
import pandas as pd
import streamlit as st

#from config import DATABASE_URL

# make direct execution from src/ work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.etl_price import update_price_database
    from src.etl_demand import update_demand_database
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
    from etl_demand import update_demand_database
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

PRICE_TARGET_COL = "price_de_lu_eur_mwh"
PRICE_DISPLAY_COL = "Strompreis (€/MWh)"


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
def load_models() -> dict[str, object]:
    candidates = {
        "LGBM": MODEL_DIR / "price_lgbm_model.pkl",
        "XGBoost": MODEL_DIR / "price_xgb_model.pkl",
    }
    models: dict[str, object] = {}
    for name, path in candidates.items():
        if path.exists():
            models[name] = load_model_from_pickle(path)
    if not models:
        raise FileNotFoundError(
            "Kein Preismodell gefunden. Erwartet z. B. models/price_lgbm_model.pkl"
        )
    return models


def to_berlin_naive(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True)
    return ts.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)


def align_features(model: object, df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["time", PRICE_TARGET_COL], errors="ignore")
    if hasattr(model, "feature_name_"):
        return X.reindex(columns=list(model.feature_name_))
    if hasattr(model, "feature_names_in_"):
        return X.reindex(columns=list(model.feature_names_in_))
    return X.select_dtypes("number")


def predict_df(model: object, df_features: pd.DataFrame, pred_col: str = "ML Prediction") -> pd.DataFrame:
    out = df_features[["time"]].copy()
    X = align_features(model, df_features)
    out[pred_col] = model.predict(X)
    return out


def load_actual_context(start_date: date, end_date: date) -> pd.DataFrame:
    """Load price, PV, wind and demand actual/forecast context from DB."""
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


def add_mean_last_7_days(df_plot: pd.DataFrame, ref_date: date) -> pd.DataFrame:
    last7_start = ref_date - timedelta(days=7)
    last7_end = ref_date - timedelta(days=1)
    df_last7 = load_actual_context(last7_start, last7_end)
    if df_last7.empty or PRICE_TARGET_COL not in df_last7.columns:
        df_plot["7-Tage-Mittel"] = pd.NA
        return df_plot

    tmp = df_last7.copy()
    tmp["hour"] = pd.to_datetime(tmp["time"], utc=True).dt.tz_convert("Europe/Berlin").dt.hour
    hourly_mean = tmp.groupby("hour")[PRICE_TARGET_COL].mean()
    df_plot["hour"] = pd.to_datetime(df_plot["time"], utc=True).dt.tz_convert("Europe/Berlin").dt.hour
    df_plot["7-Tage-Mittel"] = df_plot["hour"].map(hourly_mean)
    return df_plot.drop(columns=["hour"])


def plot_price_forecast(df_plot: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = to_berlin_naive(df_plot["time"])

    plot_cols = [
        ("Heute ML", "Heute: ML-Vorhersage"),
        ("Morgen ML", "Morgen: ML-Vorhersage"),
        (PRICE_TARGET_COL, "Echter / veröffentlichter Preis"),
        ("7-Tage-Mittel", "Mittelwert letzte 7 Tage"),
    ]
    for col, label in plot_cols:
        if col in df_plot.columns and pd.to_numeric(df_plot[col], errors="coerce").notna().any():
            ax.plot(x, df_plot[col], linewidth=1.6, label=label)

    ax.set_title(title)
    ax.set_xlabel("Zeit (Europe/Berlin)")
    ax.set_ylabel(PRICE_DISPLAY_COL)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)


def plot_energy_context(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = to_berlin_naive(df["time"])

    cols = [
        ("gen_pv_total_mwh", "PV Erzeugung"),
        #("gen_pv_input_mwh", "PV Vorhersage/Input"),
        ("gen_wind_total_mwh", "Wind Erzeugung"),
        #("gen_wind_input_mwh", "Wind Vorhersage/Input"),
        ("energy_demand_mwh", "Stromverbrauch"),
        #("demand_input_mwh", "Verbrauch Vorhersage/Input"),
    ]
    colors_dict = {"gen_pv_total_mwh": "orange", "gen_wind_total_mwh": "green", "energy_demand_mwh": "blue"}

    for col, label in cols:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            ax.plot(x, df[col], linewidth=1.4, label=label, color=colors_dict.get(col, None))

    ax.set_title(title)
    ax.set_xlabel("Zeit (Europe/Berlin)")
    ax.set_ylabel("MWh")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.autofmt_xdate()
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

    #err = cmp_df[actual_col] - cmp_df[pred_col]
    col1, col2, col3, col4 = st.columns(4)
    #col1.metric("MAE", f"{err.abs().mean():.2f} €/MWh")
    #col2.metric("RMSE", f"{(err.pow(2).mean() ** 0.5):.2f} €/MWh")
    col1.metric("MAE", f"{mae:.2f} €/MWh")
    col2.metric("RMSE", f"{rmse:.2f} €/MWh")
    col3.metric("R²", f"{r2:.2f}")
    col4.metric("Datenpunkte", f"{len(cmp_df):,}")


init_db()
models = load_models()

st.title("⚡ Strompreisprognose Deutschland")
st.markdown(
    "Zwei Ansichten: **Vorhersage für morgen** inklusive Vorgestern- und 7-Tage-Vergleich "
    "sowie **historische Vorhersage** mit echten Preisen."
)

tab_future, tab_hist = st.tabs(["Vorhersage für morgen", "Historische Vorhersage"])

with tab_future:
    today = pd.Timestamp.now(tz="Europe/Berlin").date()
    tomorrow = today + timedelta(days=1)

    col_info, col_ctrl = st.columns([2, 1])
    with col_info:
        st.markdown(f"**Vorgestern:** {(today - timedelta(days=2)).isoformat()}")
        st.markdown(f"**Heute:** {today.isoformat()}")
        st.markdown(f"**Vorhersagetag:** {tomorrow.isoformat()}")
    with col_ctrl:
        future_model_name = st.selectbox("Modell", list(models.keys()), key="future_model")

    if st.button("Vorhersage berechnen", type="primary", key="btn_future_price"):
        model = models[future_model_name]

        with st.spinner("Features für morgen werden vorbereitet …"):
            df_tomorrow_feat = prepare_data_for_price_prediction_operational()
        if df_tomorrow_feat.empty:
            st.error("Keine Features für morgen erzeugt.")
            st.stop()

        with st.spinner("Heute-Vergleich wird aus dem Modell berechnet …"):
            df_all_model = prepare_price_model_dataset()
            df_all_model["time"] = pd.to_datetime(df_all_model["time"], utc=True)
            today_start = pd.Timestamp(today, tz="Europe/Berlin")
            today_end = today_start + pd.Timedelta(days=1)
            df_today_feat = df_all_model[
                (df_all_model["time"] >= today_start) & (df_all_model["time"] < today_end)
            ].copy()

        df_tom_pred = predict_df(model, df_tomorrow_feat, "Morgen ML")
        df_today_pred = (
            predict_df(model, df_today_feat, "Heute ML") if not df_today_feat.empty else pd.DataFrame(columns=["time", "Heute ML"])
        )

        df_context = load_actual_context(today - timedelta(days=7), tomorrow)
        keep_cols = [
            "time", PRICE_TARGET_COL, "gen_pv_total_mwh", "gen_pv_input_mwh",
            "gen_wind_total_mwh", "gen_wind_input_mwh", "energy_demand_mwh", "demand_input_mwh",
        ]
        df_context = df_context[[c for c in keep_cols if c in df_context.columns]]

        df_plot = pd.concat([df_today_pred, df_tom_pred], ignore_index=True)
        df_plot = df_plot.merge(df_context, on="time", how="left")
        df_plot = add_mean_last_7_days(df_plot, today)

        st.success(f"Vorhersage abgeschlossen ({future_model_name}).")
        plot_price_forecast(
            df_plot,
            f"Strompreis: Heute, Morgen und 7-Tage-Mittel — {today} bis {tomorrow}",
        )
        plot_energy_context(
            df_plot,
            f"PV- und Wind-Erzeugung und Stromverbrauch - {today} bis {tomorrow}",
        )

        table = df_plot.copy()
        table["Zeit (Berlin)"] = to_berlin_naive(table["time"]).dt.strftime("%Y-%m-%d %H:%M")
        display_cols = [
            "Zeit (Berlin)", "Heute ML", "Morgen ML", "7-Tage-Mittel",
            "gen_pv_input_mwh", "gen_wind_input_mwh", "demand_input_mwh",
        ]
        st.dataframe(table[[c for c in display_cols if c in table.columns]], use_container_width=True)

with tab_hist:
    _default_to = date.today() - timedelta(days=1)
    _default_from = _default_to - timedelta(days=6)
    _min_date = date(2019, 1, 8)
    _max_date = date.today() - timedelta(days=1)

    col1, col2, col3 = st.columns(3)
    with col1:
        date_from = st.date_input("Von:", value=_default_from, min_value=_min_date, max_value=_max_date, key="hist_from_price")
    with col2:
        date_to = st.date_input("Bis:", value=_default_to, min_value=_min_date, max_value=_max_date, key="hist_to_price")
    with col3:
        hist_model_name = st.selectbox("Modell", list(models.keys()), key="hist_model")

    delta_days = (date_to - date_from).days
    if delta_days < 0:
        st.error('⚠ „Bis"-Datum muss nach dem „Von"-Datum liegen.')
    elif delta_days > MAX_RANGE_DAYS:
        st.warning(f"⚠ Gewählter Zeitraum: {delta_days} Tage — Maximum sind {MAX_RANGE_DAYS} Tage.")
    else:
        st.success(f"Zeitraum: {delta_days + 1} Tag(e) ✓")

    if st.button("Historische Vorhersage berechnen", type="primary", key="btn_hist_price"):
        from_str, to_str = str(date_from), str(date_to)
        model = models[hist_model_name]

        with st.spinner(f"Historische Features werden geladen: {from_str} → {to_str} …"):
            df_model = prepare_price_model_dataset()
            df_model["time"] = pd.to_datetime(df_model["time"], utc=True)
            start_ts = pd.Timestamp(date_from, tz="Europe/Berlin")
            end_ts = pd.Timestamp(date_to + timedelta(days=1), tz="Europe/Berlin")
            df_hist_feat = df_model[(df_model["time"] >= start_ts) & (df_model["time"] < end_ts)].copy()

        if df_hist_feat.empty:
            st.error(f"Keine Modelldaten für {from_str} → {to_str} gefunden.")
            st.stop()

        df_pred = predict_df(model, df_hist_feat, "ML Prediction")
        df_plot = df_hist_feat[["time", PRICE_TARGET_COL]].merge(df_pred, on="time", how="left")

        df_context = load_actual_context(date_from, date_to)
        ctx_cols = [
            "time", "gen_pv_total_mwh", "gen_wind_total_mwh", "energy_demand_mwh"
        ]
        df_plot = df_plot.merge(df_context[[c for c in ctx_cols if c in df_context.columns]], on="time", how="left")

        st.success(f"Historische Vorhersage abgeschlossen ({hist_model_name}).")

        fig, ax = plt.subplots(figsize=(14, 5))
        x = to_berlin_naive(df_plot["time"])
        ax.plot(x, df_plot[PRICE_TARGET_COL], linewidth=1.5, label="Echter Strompreis")
        ax.plot(x, df_plot["ML Prediction"], linewidth=1.5, linestyle="--", label=f"ML-Vorhersage ({hist_model_name})")
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
        plot_energy_context(
            df_plot,
            f"PV- und Wind-Erzeugung und Stromverbrauch - {from_str} bis {to_str}",
        )

        table = df_plot.copy()
        table["Zeit (Berlin)"] = to_berlin_naive(table["time"]).dt.strftime("%Y-%m-%d %H:%M")
        display_cols = [
            "Zeit (Berlin)", PRICE_TARGET_COL, "ML Prediction",
            "gen_pv_total_mwh", "gen_wind_total_mwh", "energy_demand_mwh",
        ]
        st.dataframe(table[[c for c in display_cols if c in table.columns]], use_container_width=True)

st.markdown("---")
st.caption("Strompreisprognose App • Datenquellen: SQLite DB, SMARD, Open-Meteo")
