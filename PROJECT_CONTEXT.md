# PROJECT_CONTEXT — Electricity Demand Forecasting

> Projektübersicht, App-Nutzung und Business-Erkenntnisse: [README.md](README.md)

**Energy Analytics + Time Series + Wetter + Kalenderfeatures**

---

## Projektstatus

### Abgeschlossen

- [x] EDA Stromverbrauch Deutschland (Notebook 01)
- [x] EDA Wetterdaten (Notebook 02)
- [x] Feature Engineering & EDA kombinierter Datensatz (Notebook 03)
- [x] Baseline- und ML-Modell-Evaluation (Notebook 04)
- [x] Feature Importances Analyse (Notebook 05)
- [x] Web Scraping SMARD Stromverbrauch-Daten ab 2025-10-01 (Notebook 06)
- [x] Python Source Refactoring /src (`fetch_prepare_data.py`, `train_model_predict.py`)
- [x] Vollständige ML-Pipeline: Training, Tuning, Persistenz (Notebook 07)
- [x] Bayesian Hyperparameter-Optimierung mit Optuna auf AI PC
- [x] Rolling-Features auf `shift(24)` umgestellt (kein Datenleck durch unmittelbar vorangehende Stunden)
- [x] Interaktives Notebook und Streamlit-App (Notebook 08): Tages-Vorhersage (morgen) + historischer Vergleich (Actual / SMARD / ML)
- [x] Notebook 08 GUI-Trennung: Tab 1 = Tages-Vorhersage, Tab 2 = historischer Vergleich (max. 1 Jahr, europäischer Kalender)
- [x] Bug behoben: `prepare_for_prediction_tomorrow` — Lag-Features via direktem Zeitstempel-Lookup statt `tail(24)`-Ansatz (behebt ~12h Musterverzug durch SMARD-Teiltag)

### Offen

- [ ] Asymmetrische Verlustfunktionen und Quantilregression (Notebook 09)
- [ ] ETL Pipeline
- [ ] Timezone-Fix: Open-Meteo gibt Wetterdaten mit `timezone=auto` zurück (lokale Zeit CEST/CET), während SMARD-Daten in UTC vorliegen — mögliche 1h-Verschiebung im Sommer oder Winter
- [ ] Residuallast-Vorhersage — separates Folgeprojekt

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

> **Hinweis Timezone**: SMARD liefert Timestamps in CET/CEST. Die Kaggle-Quelldaten enthalten ebenfalls UTC-Zeitstempel (`DateUTC`). Open-Meteo gibt mit `&timezone=auto` lokale Zeit zurück (CEST im Sommer +2h, CET im Winter +1h) — potenzielle 1h-Verschiebung zwischen Wetter- und Verbrauchsdaten im Sommer.


> SMARD JSON API Dokumentation: /documents/smard_api.md

---

### 3. Historische Wetterdaten

**Open-Meteo Historical Weather API**  
([Open Meteo](https://open-meteo.com/en/docs/historical-weather-api))

API-Endpunkt:
```
https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2019-01-01&end_date=2025-09-30&hourly={variables}&timezone=auto
```

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

> open-meteo API Dokumentation: /documents/open-meteo_api.md

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
| `EnergyDemand_rolling_mean_24h` | 24h-Rollmittel Verbrauch (shift(24)) |
| `EnergyDemand_rolling_mean_168h` | 168h-Rollmittel Verbrauch (shift(24)) |

> `EnergyDemand_lag_8760h` und `EnergyDemand_rolling_mean_8760h` wurden nach Feature-Importance-Analyse entfernt (geringer Beitrag, erzwang Wegfall des Jahres 2019).

> **Hinweis Rolling-Features**: `shift(24)` stellt sicher, dass das Fenster auf gestern ausgerichtet ist (T-24h bis T-(24+n-1)h). Kein Datenleck durch unmittelbar vorangehende Stunden; konsistente temporale Ausrichtung mit `lag_24h`.

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

Schritt 1: `RandomizedSearchCV` mit `TimeSeriesSplit(n_splits=5)` — respektiert zeitliche Reihenfolge.  
Schritt 2: Bayesian Optimization mit **Optuna** (`TPESampler`, 100 Trials) — bestes LightGBM-Modell gespeichert als `best_lgbm_model_bayesian_changed_rolling.pkl`.  
Scoring: `neg_mean_absolute_error` (MAE praxisrelevanter als R² für Lastvorhersage).

### Bewertungsmetriken

- **MAE** — mittlerer absoluter Fehler (primäre Metrik)
- **RMSE** — Wurzel mittlerer quadratischer Fehler (gleiche Skala wie MAE, stärker gewichtete Ausreißer)
- **R²** — Erklärte Varianz

---

## Technische Erkenntnisse & Limitierungen

- **Demand-Lag-Features** (`lag_168h`, `lag_24h`) sind die stärksten Prädiktoren — deutlich wirksamer als Kalender-Integer-Features allein
- Baumbasierte Modelle übertreffen lineare Modelle deutlich; **SVR** skaliert schlecht ($O(n^2)$–$O(n^3)$) auf den ~46k-Zeilen-Datensatz
- Standard-k-Fold CV führt bei Lag-Features zu Datenleck → `TimeSeriesSplit` verwenden
- Zyklische Kodierung (`sin`/`cos`) für `hour` und `month` empfohlen (Integer bilden keine Periodizität ab)
- Industrieller Verbrauch (~40% der Netzlast) wird durch Wetterdaten nicht abgebildet — größte verbleibende Fehlerquelle
- **Timezone-Problem (offen)**: Open-Meteo mit `&timezone=auto` liefert CEST im Sommer (+2h UTC), CET im Winter (+1h UTC); SMARD/Kaggle in UTC → potenzielle 1h-Abweichung im Sommer; Fix: `&timezone=UTC` + UTC-Parsing in `fetch_weather_data_for_cities()` / `fetch_weather_forecast_for_cities()`
- Bekannter SMARD-Zeitversatz in Notebook 08: ML-Prognose vs. SMARD-Prognose zeigt ~3h-Shift; Ursache: SMARD liefert Prognosedaten mit anderer Zeitauflösung/Offset im CSV-Export
- **Bug (behoben): ~12h Musterverzug in Vorhersage-Lag-Features** — der frühere `tail(24)`-Ansatz in `prepare_for_prediction_tomorrow` hatte zwei Fehler: (1) *Grundlegender 24h-Offset*: `create_energy_features(df_history).tail(24)` berechnet `lag_24h` via `shift(24)` relativ zur Listenposition; die letzten 24 Zeilen (z.B. `2026-05-20 00–23 Uhr`) haben dadurch `lag_24h = Verbrauch 2026-05-19`, nicht `2026-05-20`. Nach Umetikettierung auf Vorhersagezeiten `2026-05-21` zeigt das Modell Lag-Werte von 48h statt 24h vor dem Vorhersagezeitpunkt. (2) *SMARD-Teiltag*: wird der Code tagsüber ausgeführt (z.B. 13:00), liefert SMARD den aktuellen Tag nur halb (Publikationsverzug ~1–2h). `tail(24)` greift dann auf 12h von gestern + 12h von heute zurück — zwei halbvolle Tage, die als morgiger Tag umetikettiert werden. Das Ergebnis ist ein ~12h invertiertes Tagesmuster (Mittagshoch erscheint bei Mitternacht). **Fix**: Lag-Features werden jetzt per direktem Zeitstempel-Lookup berechnet (`energy_idx.get(t - 24h)`); für noch nicht von SMARD veröffentlichte Stunden des heutigen Tages greift ein Fallback auf dieselbe Stunde der Vorwoche (`t - 168h`).

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
| `streamlit_app.py` | Interaktive Web-App (2 Tabs: Morgen-Prognose + Historischer Vergleich) |

---

## Links

- [Europe Electricity Load (Hourly, 2019–2025) – Kaggle](https://www.kaggle.com/datasets/dsersun/europe-electricity-load-hourly-20192025)
- [SMARD Marktdaten - Bundesnetzagentur](https://www.smard.de/page/home/marktdaten/)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs)
- [python-holidays](https://holidays.readthedocs.io/)
- [Optuna – Hyperparameter Optimization Framework](https://optuna.readthedocs.io/)
- [Deutsche Schulferien API](https://ferien-api.maxleistner.de/)

## GitHub

- https://github.com/SW-oasen/electricity_demand_forecast


## Implementierungshinweie

### Zeitdiskrepanzen aus verschiedenen Quellen

- SMARD ts_ms
pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")

- Open-Meteo archive mit timezone=UTC
pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")

- Open-Meteo forecast mit timezone=Europe/Berlin
pd.to_datetime(df["time"]).dt.tz_localize("Europe/Berlin")

### Vorhersage-Lag-Features: Zeitstempel-Lookup statt `tail(24)`

In `prepare_for_prediction_tomorrow` werden die Energie-Lag-Features nicht mehr über `create_energy_features().tail(24)` + Zeitstempel-Überschreibung berechnet, sondern per direktem Lookup in der SMARD-History:

```python
energy_idx = df_history.set_index('time')['EnergyDemand']

# lag_24h: Verbrauch genau 24h vor dem Vorhersagezeitpunkt
lookup = energy_idx.get(t - pd.Timedelta(hours=24), np.nan)
# Fallback auf selbe Stunde Vorwoche, wenn SMARD noch nicht veröffentlicht hat
if pd.isna(lookup):
    lookup = energy_idx.get(t - pd.Timedelta(hours=168), np.nan)

# rolling_mean_24h: entspricht dem Trainings-Feature EnergyDemand.shift(24).rolling(24).mean()
# = Mittelwert von EnergyDemand aus [T-47h, T-24h] (24 Werte)
window = energy_idx.loc[
    (energy_idx.index >= t - pd.Timedelta(hours=47)) &
    (energy_idx.index <= t - pd.Timedelta(hours=24))
]
```

Dies stellt sicher, dass `lag_24h` für den Vorhersagezeitpunkt T immer auf den Verbrauch von T−24h zeigt — identisch zur Trainingsdaten-Definition — unabhängig davon, wieviele Stunden SMARD für den aktuellen Tag bereits veröffentlicht hat.