# Strompreisprognose Deutschland

Dieses Projekt prognostiziert stündliche Day-Ahead-Strompreise für Deutschland/Luxemburg (DE/LU). Dazu werden Markt-, Erzeugungs-, Nachfrage-, Wetter- und Kalenderdaten in einer reproduzierbaren Machine-Learning-Pipeline kombiniert.

## Zielsetzung

Der Ausbau wetterabhängiger erneuerbarer Energien erhöht die Volatilität des Strommarkts und führt unter anderem zu negativen Preisen und stark schwankender Residuallast. Das Projekt untersucht diese Zusammenhänge und stellt historische Auswertungen sowie eine Prognose für die nächsten 24 Stunden bereit.

## Datenquellen

- **SMARD / Bundesnetzagentur:** Day-Ahead-Preise, Nachfrage, Wind Onshore/Offshore, Photovoltaik, konventionelle Erzeugung sowie veröffentlichte Prognosen.
- **Open-Meteo:** Global-, Direkt- und Diffusstrahlung, Bewölkung sowie Windgeschwindigkeit und -richtung in 100 m Höhe.
- **Marktstammdatenregister (MaStR):** Anlagenstandorte, installierte Leistungen und Inbetriebnahmedaten zur regional gewichteten Wetteraggregation.

## Pipeline

```text
SMARD + MaStR + Open-Meteo
            ↓
   SQLite-Datenbank / ETL
            ↓
     Feature Engineering
            ↓
      ML-Modelle (XGBoost / LightGBM)
            ↓
 Historische Auswertung / 24h-Prognose
            ↓
       Streamlit-Anwendung
```

Die Daten werden in `db/energy_demand.db` gehalten. Das normalisierte Preisschema umfasst unter anderem `series_catalog`, `timeseries_values`, `ingestion_runs`, `data_quality_log`, historische Wetter-Forecast-Läufe sowie die Lineage- und Ergebnistabellen des Walk-forward-Verfahrens.

## Leakage-sichere Bewertung

Historische Prognosen laufen über `src/price_walk_forward.py` und verwenden dasselbe Protokoll wie die operative Prognose:

- eingefrorener Trainings-Cutoff: **01.10.2025** (exklusiv),
- Informationsstichtag: Vortag des Liefertags um **11:30 Europe/Berlin**,
- Day-Ahead-Preise sind bis einschließlich `D-1` zulässig,
- physische Istwerte von Last, PV und Wind werden konservativ nur bis `D-2` verwendet,
- für `D-1` und `D` werden veröffentlichte Markt- und Wetterprognosen genutzt.

Die Regeln sind in `src/forecast_protocol.py` zentralisiert und durch Tests abgesichert. Das eingefrorene Nachfrage-Upstream-Modell liegt unter `models/demand_lgbm_cutoff_2025-10-01.pkl`; das zugehörige Manifest dokumentiert Herkunft und SHA-256-Hash. Das operative Preismodell ist XGBoost, das Nachfrage-Upstream-Modell LightGBM.

Für D-1 und D werden historische Open-Meteo-ECMWF-IFS-Single-Runs verwendet. Die Läufe werden unter `data/cache/openmeteo_single_runs/` zwischengespeichert, auf vollständige D-1/D-Abdeckung geprüft und bei fehlenden oder unvollständigen historischen Zyklen durch ältere Läufe ersetzt. SMARD liefert davon unabhängig die Markt-, Last- und Erzeugungszeitreihen.

Walk-forward-Ergebnisse werden versioniert in SQLite gespeichert. Jeder Lauf erhält eine `run_id` mit Hashes des Preis- und Nachfrage-Modells sowie der Feature- und Protokollversion; dadurch bleiben frühere Prognosen auch bei späteren Modell- oder Codeänderungen nachvollziehbar. Die verwendeten Eingaben und verworfenen Wetterlauf-Kandidaten werden zusätzlich protokolliert.

## Installation

Voraussetzungen: Python **3.14** und `uv`.

```powershell
uv sync
```

Alternativ kann die vorhandene virtuelle Umgebung aktiviert werden:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Verwendung

Streamlit-Anwendung starten:

```powershell
uv run streamlit run src/streamlit_app_price.py
```

Die Anwendung bietet die Analyse historischer Prognosen, den Vergleich von Prognose und Ist-Wert sowie die Vorhersage der nächsten 24 Stunden. Die Preis-App nutzt dabei den gemeinsamen Walk-forward-Pfad (`src/price_walk_forward.py`).

ETL-Funktionen können aus Python aufgerufen werden:

```python
from src.etl_demand import update_demand_database
from src.etl_price import update_price_database

update_demand_database()
update_price_database()
```

Eine historische Bewertung kann über die Trainingshilfen gestartet werden. Die
Wetter-Inputs werden dabei vor der Prognose als historische, zeitpunktgetreue
Forecast-Läufe in SQLite abgelegt:

```python
import sqlite3
from src.config import DATABASE_PATH, DEMAND_UPSTREAM_MODEL_PATH
from src.train_predict_model import load_model_from_pickle
from src.train_price_model import evaluate_price_model_walk_forward

price_model = load_model_from_pickle("models/production/price_xgboost.pkl")
demand_model = load_model_from_pickle(DEMAND_UPSTREAM_MODEL_PATH)
with sqlite3.connect(DATABASE_PATH) as connection:
    scores = evaluate_price_model_walk_forward(
        price_model, connection, demand_model, "2025-10-01", "2025-11-01",
        model_family="xgboost",
    )
```

## MLflow-Tracking

Trainings-, Rolling-Origin- und Preis-Walk-forward-Läufe können mit MLflow
verglichen werden. Geloggt werden Modellname, Split-Parameter, MAE, RMSE, R²,
die Metriken einzelner Folds sowie bei Walk-forward-Läufen Kennzahlen zur
Input-Lineage und zu Wetter-Fallbacks. Beispiel:

```python
fold_scores, summary = rolling_origin_backtest(
    model_pipeline, df, "time", "price_de_lu_eur_mwh", "2025-10-01",
    mlflow_experiment="electricity-price",
    mlflow_run_name="lightgbm-walk-forward",
    mlflow_tags={"evaluation_mode": "walk_forward"},
)
```

Standardmäßig verwendet MLflow den lokalen Tracking-Speicher. Ein anderer
Tracking-Server kann über `MLFLOW_TRACKING_URI` gesetzt werden.

## Tests

```powershell
uv run pytest
```

Die Tests decken unter anderem das Forecast-Protokoll, Feature-Verfügbarkeit, Schema-Verträge, Walk-forward-Prognosen und die Speicherung von Prognosen ab.

## Projektstruktur

```text
data/          Roh- und Eingangsdaten
db/            SQLite-Datenbank
documents/     technische Projektdokumentation
models/        trainierte Modelle und Manifeste
notebook/      explorative Analysen
reports/       Auswertungen
src/           ETL, Features, Modelle und Streamlit-App
tests/         automatisierte Tests
util/          Hilfsskripte
```

Weitere technische Details zu Architektur, ETL, Datenmodell und Feature Engineering stehen in [documents/PROJECT_CONTEXT.md](documents/PROJECT_CONTEXT.md).

## Technologien

Python, Pandas, NumPy, scikit-learn, LightGBM, XGBoost, Optuna, MLflow, SQLite, Streamlit, SMARD API und Open-Meteo API.

## Ausblick

- Modellvergleich und Ensemble-Ansätze
- automatisierte Modellaktualisierung
- erweiterte Marktmerkmale und Backtesting-Auswertungen
- automatisierte Aktualisierung der MaStR-Erzeugerdaten und Wettergewichtungen
- Power-BI-Dashboard

## Autor

Yuchuan Liu — persönliches Data-Science-Projekt im Bereich Energieanalytik und Machine Learning.

*Letzte Aktualisierung: 2026-08-20*
