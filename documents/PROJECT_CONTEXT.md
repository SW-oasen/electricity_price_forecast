# Projektkontext – Strompreisprognose Deutschland

## Zweck dieses Dokuments

Dieses Dokument dient als technische Arbeitsdokumentation des Projekts.

Während die README einen fachlichen Überblick über Zielsetzung, Datenquellen und Ergebnisse liefert, beschreibt dieses Dokument die technische Architektur, den aktuellen Implementierungsstand sowie bekannte offene Punkte.

---

# Projektziel

Vorhersage der stündlichen Day-Ahead-Strompreise für Deutschland (DE/LU) auf Basis von:

* historischen Strompreisen
* Stromnachfrage
* Stromerzeugung
* Wetterdaten
* Kalendermerkmalen

Der Fokus liegt auf einer reproduzierbaren End-to-End-Pipeline vom Datenabruf bis zur Vorhersage.

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
update_database(...)
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

---

# Feature Engineering

## Preismerkmale

Aktuell verwendet:

* price_lag_24
* price_lag_48
* price_lag_168

---

## Nachfragemerkmale

* demand_lag_24
* demand_lag_48
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
* Interaktion mit Residuallast

---

# Modellierung

Aktuelles Hauptmodell:

```text
LightGBM Regressor
```

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
prepare_data_for_price_prediction_operational()
```

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
* Historische Vorhersagepipeline
* Operative Vorhersagepipeline

---

# Nächste Entwicklungsschritte

## Hohe Priorität

* Rolling-Origin-Backtesting
* Persistierung historischer Vorhersagen
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

Letzte Aktualisierung: 2026-06-12
