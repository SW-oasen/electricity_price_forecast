# Projekt: European Electricity Demand Forecasting

## Ziel

Ein Data-Science-/Data-Analyst-Portfolio-Projekt zur Vorhersage des stündlichen Stromverbrauchs in Deutschland.

**Energy Analytics + Time Series + Wetter + Kalenderfeatures + Power BI Storytelling**

Primäres Ziel: **Electricity Load Forecasting für Deutschland (stündlich, 2020–2025)**.

---

## Projektstatus

### Abgeschlossen

- [x] EDA Stromverbrauch Deutschland (Notebook 01)
- [x] EDA Wetterdaten (Notebook 02)
- [x] Feature Engineering & EDA kombinierter Datensatz (Notebook 03)
- [x] Baseline- und ML-Modell-Evaluation (Notebook 04)
- [x] Feature Importances Analyse (Notebook 05)
- [x] Web scraping SMARD die Stromverbrauch-Daten seit 2025-10-01 für die aktuelle Vorhersage (Notebook 06)
- [x] Python Source Refactoring, /src (`fetch_prepare_data.py`, `train_model_predict.py`)
- [x] Interaktive Vorhersage mit Datenpipeline (Notebook 07/08)
- [x] Bayesian Optimization (Optuna) auf AI PC
- [x] interactive Notebook, Streamlit für Stromverbrauch-Vorhersage bzw. Vergleich zwischen Vorhersage und realem Verbrauch
- [x] Notebook 08 GUI-Trennung: separates Interface für (1) Tages-Vorhersage (morgen) und (2) historischen Vergleich mit Datumsbereichsauswahl (max. 1 Jahr), europäischer Kalender (Wochenstart Montag)

### Offen

- [ ] ETL Pipeline
- [ ] Residuallast=Netzlast−PV−Wind Onshore - separates Projekt

---

## Datenquellen

### 1. Stromverbrauch

**Europe Electricity Load (Hourly, 2019–2025)**  
Quelle: Kaggle, basierend auf ENTSO-E Transparency Platform.  
([Kaggle](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025))

Verwendete Spalten:
- `DateUTC`
- `CountryCode` (gefiltert auf `DE`)
- `Value` → umbenannt in `EnergyDemand`

Lizenzhinweis:
- ENTSO-E attribution
- CC BY-SA 4.0

---

### 2. Aktuelle Stromverbrauchsdaten (ab 2025-10-01)

**SMARD Chart Data API** (Bundesnetzagentur)  
([SMARD](https://www.smard.de/home))

Filter-ID 410: Realisierter Stromverbrauch – Netzlast  
Filter-ID 411: Prognostizierter Stromverbrauch – Netzlast (offizielle SMARD-Tagesvorhersage)  
Programmatisch abgerufen über `fetch_smard_netzlast(filter_id=...)` in `src/fetch_prepare_data.py`.  
Filter 411 wird in Notebook 08 für den 3-Kurven-Vergleich und als Referenz-Benchmark verwendet.

---

### 3. Historische Wetterdaten

**Open-Meteo Historical Weather API**  
([Open Meteo](https://open-meteo.com/en/docs/historical-weather-api))

API-Endpunkt:
https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2019-01-01&end_date=2025-09-30&hourly={variables}&timezone=auto



Verwendete Variablen:
- `apparent_temperature`
- `rain`
- `snowfall`
- `wind_speed_10m`
- `shortwave_radiation`

Aggregation über Top-5-Städte Deutschland (gewichtet nach Stadtbevölkerung):

| Stadt | Einwohner |
|---|---|
| Berlin | 3,69 Mio |
| Hamburg | 1,86 Mio |
| München | 1,51 Mio |
| Köln | 1,02 Mio |
| Frankfurt a.M. | 0,76 Mio |

---

### 4. Feiertage

**python-holidays**  
([holidays.readthedocs.io](https://holidays.readthedocs.io/))

Features:
- `is_holiday` — nationaler/regionaler Feiertag (0/1)
- `holiday_ratio` — Anteil der Bundesländer mit Feiertag (0–1)

---

## Feature Engineering

### Zeitfeatures
| Feature | Beschreibung |
|---|---|
| `hour` | Stunde des Tages (0–23) |
| `weekday` | Wochentag (0=Mo, 6=So) |
| `month` | Monat (1–12) |
| `is_weekend` | 1 wenn Sa/So |

> Empfehlung für Weiterentwicklung: zyklische Kodierung (`sin_hour`, `cos_hour`, `sin_month`, `cos_month`) statt Integer, um Periodizität korrekt abzubilden.

### Kalenderfeatures
| Feature | Beschreibung |
|---|---|
| `is_holiday` | Feiertag ja/nein |
| `holiday_ratio` | Anteil der Bundesländer mit Feiertag (0–1) |
| `is_workday` | 1 wenn Werktag und kein Feiertag (direktes Signal für Hochlasttage) |
| `is_bridge_day` | 1 wenn Werktag eingeklemmt zwischen Feiertag und Wochenende |
| `holiday_weight` | kombiniertes Signal: `max(holiday_ratio, is_weekend × 0.5)` |
| `is_pandemic_time` | 2020-03-01 bis 2021-12-31 |

### Wetterfeatures
| Feature | Beschreibung |
|---|---|
| `apparent_temperature` | gefühlte Temperatur |
| `rain`, `snowfall` | Niederschlag |
| `wind_speed_10m` | Windgeschwindigkeit |
| `shortwave_radiation` | Solarstrahlung |
| `apparent_temperature_lag_24h` | Temperatur vor 24h |
| `apparent_temperature_rolling_mean_24h` | 24h-Rollmittel Temperatur |
| `shortwave_radiation_0m_lag_24h` | Solarstrahlung vor 24h |
| `shortwave_radiation_0m_rolling_mean_24h` | 24h-Rollmittel Solarstrahlung |
| `heating_degree` | `max(0, 18 - apparent_temperature)` |
| `cooling_degree` | `max(0, apparent_temperature - 25)` |

* Gewichtete Wetteraggregation nach Stadtbevölkerung

### Lag-Features Stromverbrauch (entscheidend für Saisonalität)
| Feature | Beschreibung |
|---|---|
| `EnergyDemand_lag_24h` | Verbrauch vor 24h (selbe Stunde gestern) |
| `EnergyDemand_lag_168h` | Verbrauch vor 168h (selbe Stunde letzte Woche) |
| `EnergyDemand_rolling_mean_24h` | 24h-Rollmittel Verbrauch (shift(1)) |
| `EnergyDemand_rolling_mean_168h` | 168h-Rollmittel Verbrauch (shift(1)) |

> `EnergyDemand_lag_8760h` und `EnergyDemand_rolling_mean_8760h` wurden nach Feature-Importance-Analyse entfernt (geringer Beitrag, erzwang Wegfall von 2019).

---

## Train/Test Split

| Split | Zeitraum | Verwendung |
|---|---|---|
| Training | 2019–2024 | Modelltraining |
| Test | 2025 | Finale Evaluation |

Zeitbasierter Split — kein zufälliges Mischen. Cross-Validation mit `TimeSeriesSplit` (kein Standard-k-Fold, da Datenleck durch Lag-Features).

---

## Modelle

### Preprocessing
Für distanzbasierte Modelle (Linear Regression, SVR): `StandardScaler` + `OneHotEncoder` über `ColumnTransformer`.  
Für baumbasierte Modelle (Random Forest, XGBoost, LightGBM): kein Preprocessing nötig.

### Evaluierte Modelle

| Modell | Preprocessing | Anmerkung |
|---|---|---|
| Linear Regression | StandardScaler + OHE | Schwache Baseline |
| Random Forest | keines | Beste Performance mit Lag-Features |
| SVR (rbf) | StandardScaler + OHE | Nicht geeignet für ~46k Zeilen; nur auf 10k-Subset getestet |
| XGBoost | keines | Gute Performance |
| LightGBM | keines | Vergleichbar mit XGBoost, schneller |
| SARIMAX | — | Auf täglicher Frequenz getestet (zu langsam auf Stundenbasis) |

### Hyperparameter-Tuning

`RandomizedSearchCV` mit `TimeSeriesSplit(n_splits=5)` — respektiert zeitliche Reihenfolge.  
Scoring: `neg_mean_absolute_error` (MAE praxisrelevanter als R² für Lastvorhersage).

### Bewertungsmetriken

- **MAE** — mittlerer absoluter Fehler (primäre Metrik)
- **MSE** — mittlerer quadratischer Fehler
- **R²** — Erklärte Varianz

---

## Erkenntnisse

- **Demand-Lag-Features** (v.a. `lag_168h`, `lag_8760h`) sind die wichtigsten Features für Saisonal-Erkennung — deutlich wirksamer als `month` oder `hour` als Integer
- Baumbasierte Modelle übertreffen lineare Modelle deutlich
- **SVR** skaliert schlecht auf große Datensätze ($O(n^2)$ bis $O(n^3)$)
- Standard-k-Fold CV führt bei Lag-Features zu Datenleck → `TimeSeriesSplit` verwenden
- Zyklische Kodierung (`sin`/`cos`) für `hour` und `month` empfohlen, da Integer keine Periodizität abbilden
- Industrieller Verbrauch (~40%) nicht durch Wetterdaten abgedeckt — potenzielle Verbesserung durch Industrieproduktionsindex (Destatis) oder ENTSO-E Day-Ahead-Preise

---

## Potenzielle Erweiterungen

- ENTSO-E Day-Ahead-Preise als Feature
- Industrieproduktionsindex (Destatis, monatlich)
- Schulferienratio
- Mehrere Länder wegen besonderem Klima (FI – Finnland, ES – Spanien)
- 7-Tage-Forecast (iterative/rekursive Vorhersage)
- Quantilregression (α > 0.5) für konservativere Vorhersagen analog SMARD

---

## Notebook-Übersicht

| Notebook | Inhalt |
|---|---|
| `01_eda_energy.ipynb` | EDA Stromverbrauch, Zeitreihenzerlegung, Saisonalität |
| `02_eda_weather.ipynb` | EDA Wetterdaten je Stadt |
| `03_eda_energy_weather.ipynb` | Feature Engineering, kombinierter Datensatz, Korrelationsanalyse |
| `04_base_models_eval.ipynb` | Modelltraining, Tuning, Lernkurven, Prediction vs. Actual |
| `05_feature_importances.ipynb` | Feature Importance Analyse, Entfernung schwacher Features |
| `06_scrape_smard.ipynb` | SMARD API Scraping für Stromverbrauch ab 2025-10-01 |
| `07_complete_ml_pipeline.ipynb` | Vollständige ML-Pipeline: Training, Tuning, Persistenz |
| `08_interactive_prediction.ipynb` | Interaktive Vorhersage: (1) Tagesvorhersage morgen mit SMARD-Prognoselinie; (2) historischer 3-Kurven-Vergleich Actual / SMARD / ML mit MAE+RMSE-Tabelle, max. 1 Jahr, europäischer Kalender |
| `09_asymmetric_loss.ipynb` | Exploration asymmetrischer Verlustfunktionen und Quantilregression für konservativere Vorhersagen |

---

## Source Code (`/src`)

| Datei | Inhalt |
|---|---|
| `fetch_prepare_data.py` | Kaggle/SMARD (Filter 410 + 411)/Open-Meteo Datenabruf, Feature Engineering; `prepare_data_for_next_day_prediction()` für die Tagesvorhersage |
| `train_model_predict.py` | Modelltraining, Hyperparameter-Tuning, Modell-Persistenz |

---

## Links

- [Europe Electricity Load (Hourly, 2019–2025) – Kaggle](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025)
- [SMARD Marktdaten - Bundesnetzagentur](https://www.smard.de/page/home/marktdaten/)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [python-holidays](https://holidays.readthedocs.io/)
- [Deutsche Schulferien API](https://ferien-api.maxleistner.de/)

## GitHub

- https://github.com/SW-oasen/electricity_demand_forecast