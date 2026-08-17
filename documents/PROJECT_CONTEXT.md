# Projektkontext – Strompreisprognose Deutschland

## Zweck dieses Dokuments

Dieses Dokument dient als technische Arbeitsdokumentation des Projekts.

Während die README einen fachlichen Überblick über Zielsetzung, Datenquellen und Ergebnisse liefert, beschreibt dieses Dokument die technische Architektur, den aktuellen Implementierungsstand sowie bekannte offene Punkte.

Die fachliche Einführung mit Motivation, Datenquellen und Anwendungsziel findet sich in der [README.md](../README.md).

---

# Projektziel

Vorhersage der stündlichen Day-Ahead-Strompreise für Deutschland (DE/LU) auf Basis von:

* Historischen Strompreisen
* Stromnachfrage
* Stromerzeugung
* Wetterdaten
* Kalendermerkmalen

Der Fokus liegt auf einer reproduzierbaren End-to-End-Pipeline vom Datenabruf bis zur Vorhersage.

## Walk-forward-Protokoll (beschlossen)

Der eingefrorene Trainingsstand endet exklusiv am **01.10.2025**. Das bedeutet:

* Das Preis-, Verbrauchs-, PV- und Windmodell werden zunächst nicht täglich neu trainiert.
* Jede historische Preisprognose bewertet einen Liefertag `D >= 01.10.2025`.
* Der Informationsstichtag liegt am Vortag `D-1` um **11:30 Europe/Berlin**, also vor dem Day-Ahead-Auktionsschluss.
* Vollständige Day-Ahead-Preise bis einschließlich `D-1` sind erlaubt; sie wurden jeweils am Vortag des Liefertags veröffentlicht.
* Physische Istwerte (Last, PV, Wind) werden bis zur belastbaren Speicherung echter Veröffentlichungszeitpunkte konservativ nur bis einschließlich `D-2` verwendet.
* Für `D-1` und `D` werden Last-, PV-/Wind- und Wetterprognosen verwendet. Für Wetter muss der historische Forecast-Modelllauf zum Informationsstichtag abrufbar gewesen sein.

Die zugehörigen Regeln liegen zentral in `src/forecast_protocol.py` und werden vor der Implementierung der Feature-Pipeline unit-getestet.

Als Upstream-Artefakt für die Preisprognose liegt das eingefrorene, angepasste
LightGBM-Verbrauchsmodell unter `models/demand_lgbm_cutoff_2025-10-01.pkl`.
Seine Herkunft, sein Trainings-Cutoff und sein SHA-256-Hash stehen im zugehörigen
versionierten Manifest. Das Modell wird innerhalb eines Walk-forward-Laufs nicht
neu trainiert.

Die operative Morgenprognose und die historische Auswertung verwenden
denselben zentralen Pfad (`src/price_walk_forward.py`). Damit ist die
historische Bewertung unmittelbar fuer den Betrieb relevant.

---

# Systemarchitektur

## Datenquellen

### SMARD

Geladene Zeitreihen:

* Day-Ahead-Preis
* Stromnachfrage
* Wind Onshore
* Wind Offshore
* Photovoltaik
* Sonstige konventionelle Erzeugung

### Open-Meteo

PV-Wetterdaten:

* shortwave_radiation
* direct_radiation
* diffuse_radiation
* cloud_cover

Wind-Wetterdaten:

* wind_speed_100m
* wind_direction_100m

### Marktstammdatenregister (MaStR)

Erzeugerdaten:

* Standorte und Koordinaten
* Brutto- und Nettoleistung
* Inbetriebsdatum

Verwendung:

* Anlagenstandorte
* Clusterbildung
* Jährliche Leistung
* Wettergewichtung

---

# Datenhaltung

Primäre Datenbank:

```text
db/energy_demand.db
```

## Kernschema

### series_catalog

Enthält Metadaten aller Zeitreihen.

Beispiele:

* day_ahead_price
* demand_actual
* wind_generation
* pv_generation
* Wetterserien

### timeseries_values

Normalisierte Speicherung aller Zeitreihenwerte.

Wichtige Felder:

* series_id
* time
* value

### ingestion_runs

Protokollierung aller ETL-Läufe.

### data_quality_log

Erfassung von Qualitätsprüfungen und Auffälligkeiten.

### energy_demand

Historie und verfügbare Prognosen des Stromverbrauchs (aus dem Vorgängerprojekt).

---

# ETL-Pipeline

## Einstiegspunkt

```python
src/etl_price.py
```

Zentrale Orchestrierung:

```python
update_demand_database()
update_price_database()
```

Aufgaben:

* Datenabruf
* Delta-Erkennung
* Datenvalidierung
* Speicherung in SQLite

---

# Ingestion-Verhalten

## SMARD

* Prüfung des aktuellen Datenstandes
* Nachladen fehlender Zeiträume
* Überspringen bereits vollständiger Daten

## Open-Meteo

* Eigenständiger Ingestion-Zweig
* Nicht von SMARD abhängig
* Tagesbasierte Delta-Erkennung

Dadurch kann Wetterdatenaktualisierung unabhängig von SMARD erfolgen.

---

# Wetteraggregation

Implementierung:

```python
util/weather_weighted.py
```

## Ziel

Erzeugung deutschlandweiter, technologiegewichteter Wetterindikatoren.

Anstatt Wetterdaten einzelner Stationen zu verwenden, werden Wetterwerte anhand der installierten Leistung der Anlagen-Geo-Cluster gewichtet.

---

## Hauptfunktionen

```Notebook 
02_energy_gen_locations.ipynb
```

* Identifizieren die Cluster der Anlagen durch KMeans Clustering
* Aggregieren die jährliche Kapazitäten jeweiliges Clusters
* Speichern die jährliche Cluster-Kapazitäten in config

```python
build_yearly_weights(...)
```

Erzeugt jahresabhängige Gewichtungen auf Basis der installierten Leistung.

```python
fetch_weighted_weather_for_technology(...)
```

Lädt und aggregiert Wetterdaten für eine Technologie.

```python
aggregate_weighted_wind_vector_features(...)
```

Aggregiert Windrichtungen über Vektorkomponenten (u/v) statt über Winkelmittelwerte.

Optional:

* Windgeschwindigkeit²
* Windgeschwindigkeit³

## Visualisierung der Erzeuger-Cluster

* Mit open-street-map die Erzeuger-Cluster mit Kapazitäten dargestellt
* Wind (Onshore und Offshore)
    - ../reports/wind_clusters_capacity_map.html
* Solar / PV 
    - ../reports/solar_clusters_capacity_map.html

---

# Feature Engineering

## Preismerkmale

Aktuell verwendet:

* price_lag_24
* price_lag_168

---

## Nachfragemerkmale

* demand_lag_24
* demand_lag_168

---

## Erzeugungsmerkmale

* PV-Erzeugung
* Wind-Erzeugung
* Residuallast

Residuallast:

```text
Nachfrage - (PV + Wind)
```

---

## Kalendermerkmale

* Stunde
* Wochentag
* Monat
* Wochenende
* Feiertag

Zusätzlich zyklische Kodierung:

* hour_sin
* hour_cos

---

## Wettermerkmale

PV:

* shortwave_radiation
* direct_radiation
* diffuse_radiation
* cloud_cover

Wind:

* wind_speed_100m
* wind_direction_sin
* wind_direction_cos

Zusätzlich:

* Wetter-Lags

---

# Modellierung

Aktuelles Hauptmodell:

```text
XGBoost Regressor
```

Die komplette eingefrorene Modellkette lautet:

* Preis: `price_xgb_model.pkl` (XGBoost)
* Verbrauch: `demand_lgbm_cutoff_2025-10-01.pkl` (LightGBM)
* PV-Erzeugung: `pv_lgbm_model.pkl` (LightGBM)
* Wind-Erzeugung: `wind_lgbm_model.pkl` (LightGBM)

`price_lgbm_model.pkl` bleibt als Vergleichsartefakt erhalten und wird nicht
von der GUI oder dem Walk-forward-Pfad verwendet.

Zielvariable:

```text
Day-Ahead-Strompreis Deutschland (EUR/MWh)
```

---

# Vorhersagepipeline

## Historische Vorhersagen

Verwendung:

* Modellbewertung
* Vergleich Prognose gegen Ist-Werte
* Analyse von Fehlern

---

## Tagesprognose für morgen

Ziel:

Vorhersage der nächsten 24 Stunden.

Aktuelle Pipeline:

```python
predict_price_target_day_from_db(...)
```

Dies ist dieselbe Funktion, die auch die historische Walk-forward-Auswertung
ausfuehrt. Fuer den Folgetag werden aktuelle verfuegbare Wettermodelllaeufe
gespeichert; im Backtest werden deren historische archivierte Gegenstuecke
verwendet.

Aufgaben:

* Zusammenführen aller Eingangsdaten
* Erzeugung der benötigten Merkmale
* Vorbereitung des Vorhersagedatensatzes

Ausgabe:

```text
24 stündliche Preisprognosen
```

---

# Streamlit-Anwendung

Aktueller Funktionsumfang:

## Historische Analyse

* Prognose vs. Ist
* Fehleranalyse
* Zeitreihenvisualisierung

## Morgenprognose

* Vorhersage der nächsten 24 Stunden
* Darstellung als Tabelle und Diagramm

---

# Bekannte technische Herausforderungen

## Datenlücken

* Die Preise, die Wind- und PV-Erzeugung von heute und gestern sind oft nicht vollständig
* Diese sind jedoch wichtige Prädikatoren
* Lösung - gestapelte Vorhersagemodell
    - Die fehlende Daten werden durch Modell bereitgestellt für die Vorhersage für morgen

## Zeitzonen

Projektstandard:

```text
Europe/Berlin
```

Besondere Aufmerksamkeit erforderlich bei:

* UTC-Konvertierung
* Sommerzeitumstellung
* 23-Stunden-Tagen
* 25-Stunden-Tagen

Betroffene Komponenten:

* ETL
* Feature Engineering
* Tomorrow Prediction

---

# Validierte Entscheidungen

Bereits umgesetzt und getestet:

* Entkopplung von SMARD- und Open-Meteo-Ingestion
* Tagesbasierte Delta-Logik
* Gewichtete Wetteraggregation
* Windvektoraggregation über u/v-Komponenten
* Gemeinsame Walk-forward-Pipeline fuer historische und operative Vorhersagen
* Persistierung historischer Preisvorhersagen

---

# Nächste Entwicklungsschritte

## Hohe Priorität

* Dokumentation der Modellmetriken

## Mittlere Priorität

* Modellvergleich
* Ensemble-Ansätze
* Erweiterte Marktmerkmale

## Niedrige Priorität

* Automatisierte Modellaktualisierung
* Deployment
* Cloud-Betrieb

---

# Zugehörige Dokumente

```text
documents/smard_api.md
documents/open-meteo_api.md
documents/umsetzung_preisdaten_smard.md

log/DECISIONS.md
log/NEXT_STEPS.md
log/SESSION_LOG.md
```

---

Letzte Aktualisierung: 2026-06-15
