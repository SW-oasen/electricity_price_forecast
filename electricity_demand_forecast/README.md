# Electricity Demand Forecasting — Deutschland

Portfolio-Projekt zur stündlichen Vorhersage des deutschen Stromverbrauchs auf Basis von Wetter-, Kalender- und historischen Verbrauchsdaten.

> **Technische Details** (Feature-Engineering, Implementierung, Modellparameter, Projektstatus): [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)

---

## Was macht dieses Projekt?

- **Tagesvorhersage**: Stündliche ML-Prognose für den nächsten Tag (00:00–23:00 Europe/Berlin), verglichen mit der offiziellen SMARD-Prognose
- **Historischer Vergleich**: Tatsächlicher Verbrauch vs. SMARD-Prognose vs. ML-Vorhersage für einen frei wählbaren Zeitraum (bis 1 Jahr) — inkl. MAE und RMSE
- **ETL-Pipeline**: Inkrementelles SQLite-Datenbank-Update (`etl.py`), das Kaggle-CSV, SMARD-API und Open-Meteo-API kombiniert und alle Features vorberechnet — Basis für schnelle historische Abfragen ohne Live-API-Aufrufe
- **Modelle**: LightGBM, LightGBM_conservative, XGBoost und XGBoost_conservative — je zwei Varianten: legacy (CSV/API) und ETL (DB-basiert), trainiert auf 2019–2024, evaluiert auf 2025
- LightGBM_conservative und XGBoost_conservative streben bis zu 5 % Unterschätzung an, führen jedoch zu mehr Überschätzung — analog zur asymmetrischen SMARD-Prognosestrategie.

---

## Quick Start – Streamlit Apps

Voraussetzung: virtuelle Umgebung aktiviert, trainierte Modelle liegen unter `models/`.

### Legacy App (API-basiert)

```powershell
cd d:\Projects\DataScience\Portfolio\electricity_demand_forecast\workspace_energy_demand
.venv\Scripts\Activate.ps1
streamlit run src/streamlit_app.py
```

Browser öffnet sich unter **http://localhost:8501**.

### ETL App (SQLite-DB-basiert, empfohlen)

```powershell
cd d:\Projects\DataScience\Portfolio\electricity_demand_forecast\workspace_energy_demand
.venv\Scripts\Activate.ps1
streamlit run src/streamlit_app_etl.py
```

Browser öffnet sich unter **http://localhost:8501** (oder Port 8502 bei gleichzeitigem Betrieb).

Beim ersten Start führt die App `update_database()` aus und befüllt/aktualisiert die SQLite-DB — danach idempotent (Sekunden).

| Tab | Funktion |
|---|---|
| 🔮 Vorhersage (morgen) | Energie-Lag-Kontext aus DB + Open-Meteo Wetter-Forecast → ML-Prognose für morgen inkl. SMARD-Vergleichslinie |
| 📊 Historischer Vergleich | Actual / SMARD / ML aus **einem einzigen DB-Query** — kein Live-API-Abruf, max. 1 Jahr, mit MAE + RMSE |

### Vergleich: Legacy vs. ETL

| Aspekt | Legacy (`streamlit_app.py`) | ETL (`streamlit_app_etl.py`) |
|---|---|---|
| Modelle | `*_bayesian.pkl` | `*_bayesian_etl.pkl` |
| Morgen-Energie-Lag | SMARD API (re-fetch) | SQLite DB (letzte 168 Zeilen) |
| Historische Daten | 3 API-Aufrufe + Feature-Berechnung | 1 SQL-Query |
| Spaltenbenennung | `EnergyDemand_lag_*` | `energy_demand_lag_*` (DB-Schema) |

---

## Interaktive Notebooks

Neben den Streamlit-Apps stehen zwei interaktive Jupyter-Notebooks bereit (in `notebook/`):

| Notebook | Art | Beschreibung |
|---|---|---|
| `08_interactive_prediction.ipynb` | Legacy (API-basiert) | Tagesvorhersage + historischer 3-Kurven-Vergleich; alle Daten werden live von SMARD / Open-Meteo abgerufen |
| `11_interactive_prediction_etl.ipynb` | ETL (DB-basiert, empfohlen) | Energie-Lag-Kontext aus SQLite-DB; historischer Vergleich per Single-SQL-Query — kein Live-API-Abruf |

### Notebook 11 — Aufbau

**Teil 1 — Tagesvorhersage (morgen)**
- Energie-Lag-Kontext wird aus der SQLite-DB geladen (letzte 168 Zeilen, kein SMARD-API-Aufruf)
- Wetter-Forecast wird live von der Open-Meteo API abgerufen
- Spaltennamen entsprechen bereits dem ETL-DB-Schema — kein Umbenennen nötig
- Ergebnis: Liniengrafik + stündliche Wertetabelle nebeneinander; SMARD-Tagesprognose als Vergleichslinie (sofern veröffentlicht)

**Teil 2 — Historischer Vergleich (Actual vs. SMARD vs. ML)**
- Einzelner DB-Query lädt Features, Istwert (`energy_demand_mwh`) und SMARD-Prognose (`smard_forecast_mwh`) in einem Schritt
- Kein erneuter API-Abruf, kein erneutes Feature-Engineering — deutlich schneller als die Legacy-Version
- Auswählbarer Zeitraum bis maximal 1 Jahr; Live-Validierung verhindert ungültige Auswahl
- Metriktabelle (MAE, RMSE, Datenpunkte) für ML-Prognose **und** SMARD-Prognose im Vergleich

### Vergleich: Notebook 08 vs. Notebook 11

| Aspekt | 08 (Legacy) | 11 (ETL) |
|---|---|---|
| Modelle | `*_bayesian.pkl` | `*_bayesian_etl.pkl` |
| Historische Features | Re-fetch + Re-Berechnung | SQLite DB (vorberechnet) |
| Historischer Istwert | SMARD API (Filter 410) | DB `energy_demand_mwh` |
| SMARD-Prognose (hist.) | SMARD API (Filter 411) | DB `smard_forecast_mwh` |
| Energie-Lag (morgen) | SMARD API (re-fetch) | SQLite DB (letzte 168 Zeilen) |
| Spaltenbenennung | `EnergyDemand_lag_*` | `energy_demand_lag_*` (DB-Schema) |


---

## Datenquellen

| Quelle | Inhalt | Lizenz |
|---|---|---|
| [Kaggle / ENTSO-E](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025) | Stündlicher Stromverbrauch Europa 2019–2025 | CC BY-SA 4.0 |
| [SMARD (Bundesnetzagentur)](https://www.smard.de/home) | Realisierter + prognostizierter Verbrauch (Filter 410 / 411) | — |
| [Open-Meteo](https://open-meteo.com/en/docs) | Stündliche Wetterdaten 5 Städte DE (Archiv + Forecast) | CC BY 4.0 |
| [python-holidays](https://holidays.readthedocs.io/) | Deutsche Feiertage, alle 16 Bundesländer | — |


### Electricity Market Data
Quelle: Bundesnetzagentur | SMARD.de  
https://www.smard.de/

SMARD Daten ist lizensiert unter CC BY 4.0.

### Weather Data
Quelle: Open-Meteo  
https://open-meteo.com/

Wetterdaten ist lizensiert unter CC BY 4.0.

Die orginal Daten sind bereinigt, aggregiert und transformiert für Machine Learning und Visualisierung. 

---

## Erkenntnisse

- **Demand-Lag-Features** (`lag_168h`, `lag_24h`) sind die stärksten Prädiktoren — deutlich wirksamer als Kalender-Integer-Features allein
- Baumbasierte Modelle (LightGBM, XGBoost) übertreffen lineare Modelle klar
- Industrieller Verbrauch (~40% der Netzlast) wird durch Wetterdaten nicht abgebildet — größte verbleibende Fehlerquelle
- Feiertags- und Brückentag-Features (`holiday_ratio`, `is_bridge_day`, `holiday_weight`) verbessern die Vorhersage an Ausnahmetagen spürbar
- SMARD offizielle Prognose (Filter 411) dient als starker Benchmark; das ML-Modell kommt ihr nah ohne Zugang zu internen Netzbetreiber-Daten. Diese benutzt vermutlich Asymmetrische Verlustfunktionen und Quantilregression um die Unterschätzung zu mininieren. Diese werden auch in den LightGBM_conservative und XGBoost_conservative umgesetzt. 

---

## Potenzielle Erweiterungen

- ENTSO-E Day-Ahead-Preise als Feature
- Industrieproduktionsindex (Destatis, monatlich)
- Schulferienratio
- Mehrere Länder wegen besonderem Klima (FI – Finnland, ES – Spanien)
- 7-Tage-Forecast (iterative/rekursive Vorhersage)

### Folgeprojekt

- Strompreis-Vorhersage: `Abhängig von PV und Wind-Energie Produktion, Gas- und Kohlepreise (für konventionelle Erzeugung als Ergänzung zu erneuerbarer Energie, und Stromverbrauch. Parallelle und stacked Prognose.`

---

## Links

- [Europe Electricity Load (Hourly, 2019–2025) – Kaggle](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025)
- [SMARD Marktdaten - Bundesnetzagentur](https://www.smard.de/page/home/marktdaten/)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [python-holidays](https://holidays.readthedocs.io/)
- [Deutsche Schulferien API](https://ferien-api.maxleistner.de/)

## GitHub

- https://github.com/SW-oasen/electricity_demand_forecast