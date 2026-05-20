# Electricity Demand Forecasting — Deutschland

Portfolio-Projekt zur stündlichen Vorhersage des deutschen Stromverbrauchs auf Basis von Wetter-, Kalender- und historischen Verbrauchsdaten.

> **Technische Details** (Feature-Engineering, Implementierung, Modellparameter, Projektstatus): [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)

---

## Was macht dieses Projekt?

- **Tagesvorhersage**: Stündliche ML-Prognose für den nächsten Tag (00:00–23:00 UTC), verglichen mit der offiziellen SMARD-Prognose
- **Historischer Vergleich**: Tatsächlicher Verbrauch vs. SMARD-Prognose vs. ML-Vorhersage für einen frei wählbaren Zeitraum (bis 1 Jahr) — inkl. MAE und RMSE
- **Modelle**: LightGBM, Random Forest, XGBoost — trainiert auf 2019–2024, evaluiert auf 2025

---

## Quick Start – Streamlit App

Voraussetzung: virtuelle Umgebung aktiviert, trainierte Modelle liegen unter `models/`.

```powershell
# 1. Ins Projektverzeichnis wechseln
cd d:\Projects\DataScience\Portfolio\electricity_demand_forecast\workspace_energy_demand

# 2. Virtuelle Umgebung aktivieren (einmalig pro Terminal-Session)
.venv\Scripts\Activate.ps1

# 3. Streamlit starten
streamlit run src/streamlit_app.py
```

Der Browser öffnet sich automatisch unter **http://localhost:8501**.

| Tab | Funktion |
|---|---|
| Vorhersage (morgen) | ML-Prognose für den nächsten Tag inkl. SMARD-Vergleichslinie |
| Historischer Vergleich | Actual / SMARD / ML — frei wählbarer Zeitraum bis 1 Jahr, mit MAE + RMSE |

---

## Datenquellen

| Quelle | Inhalt | Lizenz |
|---|---|---|
| [Kaggle / ENTSO-E](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025) | Stündlicher Stromverbrauch Europa 2019–2025 | CC BY-SA 4.0 |
| [SMARD (Bundesnetzagentur)](https://www.smard.de/home) | Realisierter + prognostizierter Verbrauch (Filter 410 / 411) | — |
| [Open-Meteo](https://open-meteo.com/en/docs) | Stündliche Wetterdaten 5 Städte DE (Archiv + Forecast) | CC BY 4.0 |
| [python-holidays](https://holidays.readthedocs.io/) | Deutsche Feiertage, alle 16 Bundesländer | — |

---

---

## Erkenntnisse

- **Demand-Lag-Features** (`lag_168h`, `lag_24h`) sind die stärksten Prädiktoren — deutlich wirksamer als Kalender-Integer-Features allein
- Baumbasierte Modelle (LightGBM, XGBoost, Random Forest) übertreffen lineare Modelle klar
- Industrieller Verbrauch (~40% der Netzlast) wird durch Wetterdaten nicht abgebildet — größte verbleibende Fehlerquelle
- Feiertags- und Brückentag-Features (`holiday_ratio`, `is_bridge_day`, `holiday_weight`) verbessern die Vorhersage an Ausnahmetagen spürbar
- SMARD offizielle Prognose (Filter 411) dient als starker Benchmark; das ML-Modell kommt ihr nah ohne Zugang zu internen Netzbetreiber-Daten

---

## Potenzielle Erweiterungen

- ENTSO-E Day-Ahead-Preise als Feature
- Industrieproduktionsindex (Destatis, monatlich)
- Schulferienratio
- Mehrere Länder wegen besonderem Klima (FI – Finnland, ES – Spanien)
- 7-Tage-Forecast (iterative/rekursive Vorhersage)
- Quantilregression (α > 0.5) für konservative Planungsszenarien analog SMARD

### Folgeprojekt

- Residuallast-Vorhersage: `Residuallast = Netzlast − PV − Wind Onshore − konventionelle Erzeugung`

---

## Links

- [Europe Electricity Load (Hourly, 2019–2025) – Kaggle](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025)
- [SMARD Marktdaten - Bundesnetzagentur](https://www.smard.de/page/home/marktdaten/)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [python-holidays](https://holidays.readthedocs.io/)
- [Deutsche Schulferien API](https://ferien-api.maxleistner.de/)

## GitHub

- https://github.com/SW-oasen/electricity_demand_forecast