"""
Streamlit web app — Germany hourly energy demand forecast.

Two sections:
  1. Vorhersage (morgen)  — predict the full next day (00:00–23:00 UTC)
  2. Historischer Vergleich — compare predictions vs actual SMARD demand
     over a user-selected date range (max 1 year)

Run with (from workspace root):
    streamlit run src/streamlit_app.py
"""

import sys
import os
# Allow importing sibling modules (fetch_prepare_data, train_model_predict)
# Works whether the app is run from the workspace root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta, datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

FILTER_SMARD_FORECAST = 411  # Prognostizierter Stromverbrauch: Netzlast

from fetch_prepare_data import (
    #prepare_data_for_next_day_prediction,
    prepare_for_prediction_tomorrow,
    fetch_smard_netzlast,
    create_energy_features,
    create_time_based_features,
    prepare_weather_data,
    combine_energy_weather_dataset,
)
from train_model_predict import load_model_from_pickle

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stromverbrauchsprognose Deutschland",
    page_icon="⚡",
    layout="wide",
)

# ── load models once (cached across sessions) ──────────────────────────────────
@st.cache_resource
def load_models():
    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    return {
        "LGBM":          load_model_from_pickle(os.path.join(_base, "best_lgbm_model_bayesian.pkl")),
        'LGBM_conservative': load_model_from_pickle(os.path.join(_base, "best_lgbm_model_bayesian_conservative.pkl")),
        "XGBoost":       load_model_from_pickle(os.path.join(_base, "best_xgb_model_bayesian.pkl")),
        'XGBoost_conservative': load_model_from_pickle(os.path.join(_base, "best_xgb_model_bayesian_conservative.pkl")),
    }


models = load_models()


def _set_padded_ylim(ax: plt.Axes, df_plot: pd.DataFrame) -> None:
    plotted_values = pd.Series(
        df_plot[["Actual", "ML Prediction", "SMARD Forecast"]].to_numpy().ravel()
    ).dropna()
    if plotted_values.empty:
        return

    y_min = float(plotted_values.min())
    y_max = float(plotted_values.max())
    padding = (y_max - y_min) * 0.10
    if padding == 0:
        padding = max(abs(y_max) * 0.10, 1.0)
    ax.set_ylim(y_min - padding, y_max + padding)


def _render_metric_comparison(
    model_name: str,
    mae_ml: float,
    rmse_ml: float,
    ml_points: int,
    mae_smard: float | None = None,
    rmse_smard: float | None = None,
    smard_points: int | None = None,
) -> None:
    rows = [
        {
            "Series": f"ML Prediction ({model_name})",
            "MAE (MWh)": f"{mae_ml:,.0f}",
            "RMSE (MWh)": f"{rmse_ml:,.0f}",
            "Points": f"{ml_points:,}",
        }
    ]
    if mae_smard is not None and rmse_smard is not None and smard_points is not None:
        rows.append(
            {
                "Series": "SMARD official forecast",
                "MAE (MWh)": f"{mae_smard:,.0f}",
                "RMSE (MWh)": f"{rmse_smard:,.0f}",
                "Points": f"{smard_points:,}",
            }
        )

    st.markdown("### **Metrikvergleich**")
    df_metrics = pd.DataFrame(rows)
    html = df_metrics.to_html(index=False, border=0)
    st.markdown(
        """
        <style>
        .metric-comparison-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
        }
        .metric-comparison-table th,
        .metric-comparison-table td {
            padding: 10px 12px;
            text-align: left;
            border: 1px solid rgba(49, 51, 63, 0.2);
        }
        .metric-comparison-table th {
            background: rgba(240, 242, 246, 0.9);
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="metric-comparison-table">{html}</div>', unsafe_allow_html=True)

# ── page header ────────────────────────────────────────────────────────────────
st.title("⚡ Stromverbrauchsprognose Deutschland")
st.markdown("Stündliche Vorhersage und Vergleich der deutschen Netzlast (SMARD).")

tab_future, tab_hist = st.tabs(["🔮 Vorhersage (morgen)", "📊 Historischer Vergleich"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Future Prediction (tomorrow, full day)
# ══════════════════════════════════════════════════════════════════════════════
with tab_future:
    st.markdown("Vorhersage des Stromverbrauchs für den **nächsten Tag** (00:00–23:00 UTC).")

    now_utc  = datetime.now(timezone.utc)
    tomorrow = date.today() + timedelta(days=1)

    col_info, col_ctrl = st.columns([2, 1])
    with col_info:
        st.markdown(f"**Aktuelle Uhrzeit (UTC):** {now_utc.strftime('%Y-%m-%d %H:%M')}")
        st.markdown(f"**Vorhersagetag:** {tomorrow.isoformat()}")
    with col_ctrl:
        future_model = st.selectbox("Modell", options=list(models.keys()), key="future_model")

    if st.button("Predict for Tomorrow", type="primary", key="btn_future"):
        tomorrow_str = tomorrow.isoformat()

        # 1. SMARD official consumption forecast (filter 411) ─────────────────
        with st.spinner("SMARD-Prognose wird abgerufen …"):
            try:
                df_smard_fc = fetch_smard_netzlast(
                    tomorrow_str, tomorrow_str, filter_id=FILTER_SMARD_FORECAST
                )
            except Exception:
                df_smard_fc = pd.DataFrame(columns=["time", "EnergyDemand"])

        # 2. ML features + prediction ─────────────────────────────────────────
        with st.spinner(f"Features werden vorbereitet für {tomorrow_str} …"):
            try:
                df_future = prepare_for_prediction_tomorrow(prediction_date=tomorrow_str)
            except Exception as exc:
                st.error(f"Feature-Vorbereitung fehlgeschlagen: {exc}")
                st.stop()

        if df_future.empty:
            st.error("Keine Features zurückgegeben — API-Verbindung prüfen.")
            st.stop()

        with st.spinner(f"{future_model} wird ausgeführt …"):
            X     = df_future.drop(columns=["time", "EnergyDemand"], errors="ignore")
            model = models[future_model]
            if hasattr(model, "feature_names_in_"):
                X = X.reindex(columns=model.feature_names_in_)
            preds = model.predict(X)

        st.success(f"Vorhersage abgeschlossen — {tomorrow_str} ({future_model})")

        col_chart, col_table = st.columns([2.5, 1])

        with col_chart:
            fig, ax = plt.subplots(figsize=(10, 4))
            if not df_smard_fc.empty:
                ax.plot(df_smard_fc["time"], df_smard_fc["EnergyDemand"],
                        color="mediumseagreen", linewidth=1.5, linestyle="-.",
                        label="SMARD offizielle Prognose")
            ax.plot(df_future["time"], preds, linewidth=2, color="darkorange", linestyle="--",
                    label=f"{future_model} Vorhersage")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.set_xlabel("Stunde (UTC)")
            ax.set_ylabel("Netzlast (MWh)")
            ax.set_title(f"Stromverbrauchsprognose — {tomorrow_str}  [{future_model}]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)

        with col_table:
            df_result = df_future[["time"]].copy()
            df_result["ML (MWh)"] = preds.round(0).astype(int)
            if not df_smard_fc.empty:
                smard_idx = df_smard_fc.set_index("time")["EnergyDemand"]
                df_result["SMARD (MWh)"] = (
                    df_result["time"].map(smard_idx).round(0).astype("Int64")
                )
            df_result["Stunde (UTC)"] = df_result["time"].dt.strftime("%H:%M")
            display_cols = ["Stunde (UTC)", "ML (MWh)"]
            if "SMARD (MWh)" in df_result.columns:
                display_cols.append("SMARD (MWh)")
            st.dataframe(
                df_result[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=600,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Historical Comparison
# ══════════════════════════════════════════════════════════════════════════════
MAX_RANGE_DAYS = 365

with tab_hist:
    st.markdown(
        "Vorhersage und tatsächlicher Verbrauch (SMARD) im Vergleich. "
        "Maximaler Zeitraum: **1 Jahr**."
    )

    _default_to   = date.today() - timedelta(days=1)
    _default_from = _default_to - timedelta(days=6)
    _max_date     = date.today() - timedelta(days=1)

    col1, col2, col3 = st.columns(3)
    with col1:
        date_from = st.date_input(
            "Von:",
            value=_default_from,
            min_value=date(2019, 1, 1),
            max_value=_max_date,
            key="hist_from",
        )
    with col2:
        date_to = st.date_input(
            "Bis:",
            value=_default_to,
            min_value=date(2019, 1, 1),
            max_value=_max_date,
            key="hist_to",
        )
    with col3:
        hist_model = st.selectbox("Modell", options=list(models.keys()), key="hist_model")

    # ── Range validation ───────────────────────────────────────────────────────
    delta_days = (date_to - date_from).days

    if delta_days < 0:
        st.error('⚠ „Bis"-Datum muss nach dem „Von"-Datum liegen.')
    elif delta_days > MAX_RANGE_DAYS:
        st.warning(
            f"⚠ Gewählter Zeitraum: **{delta_days} Tage** — "
            f"Maximum sind **{MAX_RANGE_DAYS} Tage** (1 Jahr). "
            "Bitte Auswahl einschränken."
        )
    else:
        st.success(f"Zeitraum: {delta_days + 1} Tag(e)  ✓")

        if st.button("Compare Prediction vs Actual", type="primary", key="btn_compare"):
            from_str = str(date_from)
            to_str   = str(date_to)

            # 1. Fetch actual SMARD data (filter 410) ──────────────────────────────
            with st.spinner(f"SMARD-Verbrauchsdaten werden abgerufen für {from_str} → {to_str} …"):
                try:
                    df_actual = fetch_smard_netzlast(from_str, to_str)
                except Exception as exc:
                    st.error(f"SMARD-Abruf fehlgeschlagen: {exc}")
                    st.stop()

            if df_actual.empty:
                st.error(f"Keine SMARD-Daten verfügbar für {from_str} → {to_str}.")
                st.stop()

            # 2. Fetch SMARD official forecast (filter 411) ───────────────────────
            with st.spinner("SMARD-Prognose (Filter 411) wird abgerufen …"):
                try:
                    df_smard_fc = fetch_smard_netzlast(
                        from_str, to_str, filter_id=FILTER_SMARD_FORECAST
                    )
                except Exception:
                    df_smard_fc = pd.DataFrame(columns=["time", "EnergyDemand"])

            # 3. Build feature matrix ──────────────────────────────────────────
            with st.spinner("Modellfeatures werden berechnet (Energie + Wetter) …"):
                HISTORY_DAYS = 15
                try:
                    hist_start = (
                        pd.to_datetime(from_str) - pd.Timedelta(days=HISTORY_DAYS)
                    ).strftime("%Y-%m-%d")

                    df_energy = fetch_smard_netzlast(hist_start, to_str)
                    df_energy = create_energy_features(df_energy)
                    df_energy = create_time_based_features(
                        df_energy, in_year=pd.to_datetime(to_str).year
                    )
                    df_weather = prepare_weather_data(
                        in_start_date=hist_start, in_end_date=to_str
                    )
                    df_feat = combine_energy_weather_dataset(df_energy, df_weather)
                    df_feat = df_feat.sort_values("time").reset_index(drop=True)

                    from_ts = pd.to_datetime(from_str, utc=True)
                    to_ts   = pd.to_datetime(to_str,   utc=True) + pd.Timedelta(hours=23)
                    df_feat = df_feat[
                        (df_feat["time"] >= from_ts) & (df_feat["time"] <= to_ts)
                    ].reset_index(drop=True)

                except Exception as exc:
                    st.error(f"Feature-Vorbereitung fehlgeschlagen: {exc}")
                    st.stop()

            if df_feat.empty:
                st.error("Keine Feature-Daten für den gewählten Zeitraum.")
                st.stop()

            # 4. Predict ───────────────────────────────────────────────────────
            with st.spinner(f"{hist_model} wird ausgeführt …"):
                X     = df_feat.drop(columns=["time", "EnergyDemand"], errors="ignore")
                model = models[hist_model]
                if hasattr(model, "feature_names_in_"):
                    X = X.reindex(columns=model.feature_names_in_)
                preds = model.predict(X)

            # 5. Align all three series on shared timestamps ───────────────────
            s_pred   = pd.Series(preds, index=df_feat["time"], name="ML Prediction")
            s_actual = df_actual.set_index("time")["EnergyDemand"].rename("Actual")
            s_smard  = (
                df_smard_fc.set_index("time")["EnergyDemand"].rename("SMARD Forecast")
                if not df_smard_fc.empty
                else pd.Series(dtype=float, name="SMARD Forecast")
            )
            df_plot = pd.concat([s_actual, s_smard, s_pred], axis=1)

            st.success(f"Vergleich abgeschlossen — {from_str} → {to_str} ({hist_model})")

            # 6. Plot ──────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df_plot.index, df_plot["Actual"],
                    color="steelblue", linewidth=1.5,
                    label="Tatsächlicher Verbrauch (SMARD)")
            if not df_smard_fc.empty and df_plot["SMARD Forecast"].notna().any():
                ax.plot(df_plot.index, df_plot["SMARD Forecast"],
                        color="mediumseagreen", linewidth=1.5, linestyle="-.",
                        label="SMARD offizielle Prognose")
            ax.plot(df_plot.index, df_plot["ML Prediction"],
                    color="darkorange", linewidth=1.5, linestyle="--",
                    label=f"ML Vorhersage ({hist_model})")
            locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            _set_padded_ylim(ax, df_plot)
            ax.set_xlabel("Datum / Uhrzeit (UTC)")
            ax.set_ylabel("Netzlast (MWh)")
            ax.set_title(
                f"Tatsächlicher vs. vorhergesagter Verbrauch — "
                f"{from_str} bis {to_str}  [{hist_model}]"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)

            # 7. Metrics ───────────────────────────────────────────────────────
            df_ml_cmp = df_plot[["Actual", "ML Prediction"]].dropna()
            mae_ml    = (df_ml_cmp["Actual"] - df_ml_cmp["ML Prediction"]).abs().mean()
            rmse_ml   = ((df_ml_cmp["Actual"] - df_ml_cmp["ML Prediction"]) ** 2).mean() ** 0.5

            if not df_smard_fc.empty and df_plot["SMARD Forecast"].notna().any():
                df_sm_cmp  = df_plot[["Actual", "SMARD Forecast"]].dropna()
                mae_smard  = (df_sm_cmp["Actual"] - df_sm_cmp["SMARD Forecast"]).abs().mean()
                rmse_smard = ((df_sm_cmp["Actual"] - df_sm_cmp["SMARD Forecast"]) ** 2).mean() ** 0.5
                _render_metric_comparison(
                    model_name=hist_model,
                    mae_ml=mae_ml,
                    rmse_ml=rmse_ml,
                    ml_points=len(df_ml_cmp),
                    mae_smard=mae_smard,
                    rmse_smard=rmse_smard,
                    smard_points=len(df_sm_cmp),
                )
            else:
                _render_metric_comparison(
                    model_name=hist_model,
                    mae_ml=mae_ml,
                    rmse_ml=rmse_ml,
                    ml_points=len(df_ml_cmp),
                )
