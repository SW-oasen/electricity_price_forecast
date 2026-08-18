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
      ML-Modelle (LightGBM)
            ↓
 Historische Auswertung / 24h-Prognose
            ↓
       Streamlit-Anwendung
```

Die Daten werden in `db/energy_demand.db` gehalten. Das normalisierte Preisschema umfasst unter anderem `series_catalog`, `timeseries_values`, `ingestion_runs`, `data_quality_log` sowie Tabellen für historische Wetter-Forecast-Läufe und externe Forecast-Snapshots.

## Leakage-sichere Bewertung

Historische Prognosen laufen über `src/price_walk_forward.py` und verwenden dasselbe Protokoll wie die operative Prognose:

- eingefrorener Trainings-Cutoff: **01.10.2025** (exklusiv),
- Informationsstichtag: Vortag des Liefertags um **11:30 Europe/Berlin**,
- Day-Ahead-Preise sind bis einschließlich `D-1` zulässig,
- physische Istwerte von Last, PV und Wind werden konservativ nur bis `D-2` verwendet,
- für `D-1` und `D` werden veröffentlichte Markt- und Wetterprognosen genutzt.

Die Regeln sind in `src/forecast_protocol.py` zentralisiert und durch Tests abgesichert. Das eingefrorene Nachfrage-Upstream-Modell liegt unter `models/demand_lgbm_cutoff_2025-10-01.pkl`; das zugehörige Manifest dokumentiert Herkunft und SHA-256-Hash.

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

Die Anwendung bietet die Analyse historischer Prognosen, den Vergleich von Prognose und Ist-Wert sowie die Vorhersage der nächsten 24 Stunden.

ETL-Funktionen können aus Python aufgerufen werden:

```python
from src.etl_demand import update_demand_database
from src.etl_price import update_price_database

update_demand_database()
update_price_database()
```

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

Python, Pandas, NumPy, scikit-learn, LightGBM, XGBoost, Optuna, SQLite, Streamlit, SMARD API und Open-Meteo API.

## Ausblick

- Modellvergleich und Ensemble-Ansätze
- automatisierte Modellaktualisierung
- erweiterte Marktmerkmale und Backtesting-Auswertungen
- automatisierte Aktualisierung der MaStR-Erzeugerdaten und Wettergewichtungen
- Power-BI-Dashboard

## Autor

Yuchuan Liu — persönliches Data-Science-Projekt im Bereich Energieanalytik und Machine Learning.

*Letzte Aktualisierung: 2026-08-18*
