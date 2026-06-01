# Projektuebersicht: Strompreisprognose Deutschland

## Zielsetzung
Dieses Projekt prognostiziert stuendliche Strompreise in Deutschland mit Machine Learning. Grundlage sind historische Zeitreihen zu Marktpreis, Erzeugung, Nachfrage und Wetter. Der Fokus liegt auf robuster Datenpipeline, nachvollziehbarer Feature-Bildung und reproduzierbarer Modellierung.

## Projektstruktur
- data/: Roh- und verarbeitete Datensaetze
- db/: SQLite-Datenbanken fuer ETL und Analyse
- documents/: Projektdokumentation, API-Notizen, Kontext
- models/: Modellartefakte
- notebook/: EDA-, ETL- und Analyse-Notebooks
- presentation/: Praesentationsmaterial
- reports/: Karten und Ergebnis-Reports
- src/: ETL-Orchestrierung, Training, Vorhersage, App
- util/: API-Clients und Hilfsfunktionen
- log/: lokale Entscheidungs- und Aenderungslogs

## Aktueller ETL-Stand (Preis + Wetter)
- Zielschema: normalisierte Tabellen mit series_catalog, timeseries_values, ingestion_runs und data_quality_log.
- Datenquellen:
	- SMARD: Preis und Erzeugungsserien
	- Open-Meteo: gewichtete Wetterserien fuer PV und Wind
- Ingestion-Verhalten:
	- SMARD und Open-Meteo werden in update_database unabhaengig voneinander bewertet und geladen.
	- Wenn SMARD bereits aktuell ist, wird nur SMARD uebersprungen. Open-Meteo laeuft weiterhin separat.
	- Open-Meteo nutzt Delta-Logik auf Tagesbasis (kein unnoetiger Re-Fetch desselben Tages bei bereits abgeschlossenem Tagesstand).
- Zeitgrenzen Open-Meteo:
	- Archive-Daten werden in Europe/Berlin auf [start_date 00:00, end_date+1 Tag 00:00) geclippt.
	- Dadurch ist die Tagesausrichtung stabil (24/h auf Normaltagen).

## Wettervariablen
- PV:
	- shortwave_radiation
	- direct_radiation
	- diffuse_radiation
	- cloud_cover
- Wind:
	- wind_speed_100m
	- wind_direction_100m

## Hilfsfunktionen (neu)
- util/weather_weighted.py enthaelt wiederverwendbare Logik fuer:
	- build_yearly_weights(...)
	- fetch_weighted_weather_for_technology(...)
	- aggregate_weighted_wind_vector_features(...)
- aggregate_weighted_wind_vector_features(...) aggregiert Windrichtung ueber u/v-Komponenten und kann Windgeschwindigkeits-Potenzmerkmale (pow2, pow3) erzeugen.

## Schnellstart
1. Abhaengigkeiten installieren
	 - pip install -r requirements_gpu.txt
2. ETL ausfuehren
	 - Python: src/etl_price.py (oder update_database im Notebook)
3. Notebook pruefen
	 - notebook/03_fetch_aggregate_weather.ipynb
4. Optional App starten
	 - streamlit run src/streamlit_app_demand.py

## Hinweise
- Die zentrale Arbeitsdokumentation zu Entscheidungen und groesseren Aenderungen liegt im Ordner log/.
- Fuer den Verlauf und technische Leitlinien siehe:
	- log/DECISIONS.md
	- log/NEXT_STEPS.md
	- log/SESSION_LOG.md

---
Letzte Aktualisierung: 2026-06-01