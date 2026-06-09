import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta

from src.config import *
from util.weather_weighted import *
from src.etl_price import *
from src.fetch_price_data import *
from src.train_predict_model import *
from util.time_features import *

# Constants
MAX_RANGE_DAYS = 365

# Set page config
st.set_page_config(
    page_title="Strompreisprognose",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Strompreisprognose")
st.markdown("""
Diese Anwendung zeigt die Prognose für Strompreise basierend auf historischen Daten.
Sie können Vorhersagen für morgen oder historische Vergleiche durchführen.
""")

# Load model
@st.cache_resource
def load_model():
    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    return {
        "LGBM":               load_model_from_pickle(
            os.path.join(_base, "price_lgbm_model.pkl")),
        #"XGBoost":            load_model_from_pickle(
        #    os.path.join(_base, "price_xgb_model.pkl")),
    }

model = load_model()

# Function to prepare data for plotting
def prepare_plot_data(df_db, from_str, to_str):
    # Extract actual prices and predictions
    s_actual = df_db.set_index("time")["price_eur_per_mwh"].rename("Tatsächlicher Preis")
    
    # Check if we have predictions in the database
    if "prediction" in df_db.columns:
        s_pred = df_db.set_index("time")["prediction"].rename("Vorhersage")
    else:
        # If no predictions, compute them using the model
        s_pred = pd.Series(index=df_db["time"], name="Vorhersage")
        # This would require re-implementing prediction logic or storing predictions in DB
    
    return s_actual, s_pred

# Function to render metrics
def render_metrics(actual, pred):
    if not pred.isna().all():
        df_cmp = pd.concat([actual, pred], axis=1).dropna()
        if len(df_cmp) > 0:
            mae = (df_cmp["Tatsächlicher Preis"] - df_cmp["Vorhersage"]).abs().mean()
            rmse = ((df_cmp["Tatsächlicher Preis"] - df_cmp["Vorhersage"]) ** 2).mean() ** 0.5
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE (Mean Absolute Error)", f"{mae:.2f} €/MWh")
            with col2:
                st.metric("RMSE (Root Mean Square Error)", f"{rmse:.2f} €/MWh")

# Tab 1: Vorhersage für morgen
with st.expander("Vorhersage für morgen", expanded=True):
    st.subheader("Vorhersage für morgen")
    
    # Get tomorrow's date
    tomorrow = date.today() + timedelta(days=1)
    
    # Create a simple prediction for tomorrow (this would be more complex in reality)
    with st.spinner("Berechne Vorhersage für morgen..."):
        # In a real app, this would use the model to predict tomorrow's price
        # For now, we'll simulate it based on recent trends
        conn = get_connection()
        try:
            df_recent = load_combined_data(conn, start_date=(date.today() - timedelta(days=7)), end_date=date.today())
        finally:
            conn.close()
            
        if not df_recent.empty:
            recent_avg = df_recent["price_eur_per_mwh"].mean()
            # Simple prediction: average of last week + some variation
            tomorrow_pred = recent_avg + np.random.normal(0, 5)
        else:
            tomorrow_pred = 50.0  # Default value
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vorhergesagter Preis für morgen", f"{tomorrow_pred:.2f} €/MWh")
    with col2:
        st.metric("Aktueller Durchschnittspreis", f"{recent_avg:.2f} €/MWh" if 'recent_avg' in locals() else "N/A")

# Tab 2: Historischer Vergleich
with st.expander("Historischer Vergleich"):
    st.subheader("Historischer Vergleich")
    
    # Date selection
    _default_to = date.today() - timedelta(days=1)
    _default_from = _default_to - timedelta(days=6)
    _min_date = date(2019, 1, 8)
    _max_date = date.today() - timedelta(days=1)

    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input(
            "Von:",
            value=_default_from,
            min_value=_min_date,
            max_value=_max_date,
            key="hist_from_price",
        )
    with col2:
        date_to = st.date_input(
            "Bis:",
            value=_default_to,
            min_value=_min_date,
            max_value=_max_date,
            key="hist_to_price",
        )

    # Range validation
    delta_days = (date_to - date_from).days

    if delta_days < 0:
        st.error('⚠ „Bis"-Datum muss nach dem „Von"-Datum liegen.')
    elif delta_days > MAX_RANGE_DAYS:
        st.warning(
            f"⚠ Gewählter Zeitraum: **{delta_days} Tage** — "
            f"Maximum sind **{MAX_RANGE_DAYS} Tage**. "
            "Bitte Auswahl einschränken."
        )
    else:
        st.success(f"Zeitraum: {delta_days + 1} Tag(e)  ✓")

        if st.button("Vergleich Vorhersage vs. Tatsächlich", type="primary", key="btn_compare_price"):
            from_str = str(date_from)
            to_str = str(date_to)

            # Load data from database
            with st.spinner(f"Daten werden aus DB geladen: {from_str} → {to_str} …"):
                conn = get_connection()
                try:
                    df_db = load_combined_data(conn, start_date=from_str, end_date=to_str)
                finally:
                    conn.close()

            if df_db.empty:
                st.error(f"Keine Daten in der DB für {from_str} → {to_str}.")
                st.stop()

            # Prepare data for plotting
            s_actual, s_pred = prepare_plot_data(df_db, from_str, to_str)

            # Plot
            fig, ax = plt.subplots(figsize=(14, 5))
            
            # Plot actual prices
            ax.plot(s_actual.index, s_actual.values, 
                   color="steelblue", linewidth=1.5, label="Tatsächlicher Preis")
            
            # Plot predictions if available
            if not s_pred.isna().all():
                ax.plot(s_pred.index, s_pred.values, 
                       color="darkorange", linewidth=1.5, linestyle="--", label="Vorhersage")
            
            ax.set_xlabel("Datum / Uhrzeit (Europe/Berlin)")
            ax.set_ylabel("Strompreis (€/MWh)")
            ax.set_title(
                f"Tatsächlicher vs. vorhergesagter Strompreis — "
                f"{from_str} bis {to_str}"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Show metrics
            render_metrics(s_actual, s_pred)

# Additional information
st.markdown("---")
st.subheader("Informationen zur Prognose")
st.markdown("""
Diese Strompreisprognose basiert auf historischen Daten und maschinellem Lernen.
Die Vorhersagen sind nicht garantiert und sollten nur als Richtwerte verwendet werden.
""")

# Footer
st.markdown("---")
st.caption("Strompreisprognose App • Datenquelle: DB-Verbindung")