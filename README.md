# Strompreisprognose Deutschland

## Zielsetzung
Als Folgeprojekt zu Stromverbrauch Prognose ist das Ziel dieses Projektes die stündliche Strompreise in Deutschland mit Machine Learning vorherzusagen. Grundlage sind historische Zeitreihen zu Marktpreis, Erzeugung, Nachfrage und Wetter. Der Fokus liegt auf robuster Datenpipeline, nachvollziehbarer Feature-Bildung und reproduzierbarer Modellierung.

## Datenquellen
- Bundesnetzagentur / SMARD
    - https://www.smard.de/home
    - Lizenz:
- Marktstammdatenregister (MaStR.)
    - https://www.marktstammdatenregister.de/MaStR
    - Lizenz:
- OPEN-METEO
    - https://open-meteo.com/
    - Lizenz: 

## Applikation - Streamlit GUI


## Projektstruktur
- data/: Roh- und verarbeitete Datensaetze
- db/: SQLite-Datenbanken fuer ETL und Analyse
- documents/: Projektdokumentation, API-Notizen, Kontext
- models/: Modellartefakte
- notebook/: EDA-, ETL-, Analyse-, Train- und Predict-Notebooks
	- 01_eda_price
	- 02_energy_gen_locations
	- 03_fetch_aggregate_weather
	- 04_train_pv_gen_model
	- 05_train_win_gen_model
	- 06_predict_price
	- 07_interactive_predicton
- presentation/: Präsentationsmaterial
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

	## Aktueller Modell-Stand (Preisprognose)
	- Notebook-Implementierung gestartet: `notebook/05_predict_price.ipynb`
	- Merkmalsgruppen (aktuell):
		- Nachfrage (`demand_input_mwh`)
		- PV-/Wind-Erzeugung (`gen_wind_input_mwh`, `gen_pv_input_mwh`)
		- Residuallast (`residual_load_input_mwh`)
		- Preis-/Nachfrage-Lags (24h, 48h, 168h)
		- Kalendermerkmale (inkl. zyklischer Stunde)
		- optionaler Wetterblock (direkt + Lags + Interaktion)
	- Bisherige Beobachtung:
		- Basis-LightGBM mit wenigen Features: R² ~ 0.49
		- Zusätzliche Feature Set: R² ~ 0.53
		- Zusätzliche Feature + Wetterblock: deutliche Verbesserung (R² ~ 0.78, MAE deutlich reduziert)
	- Methodischer Hinweis:
		- Wettermerkmale werden als optionaler Residual-Signalblock behandelt und per Backtest validiert (nicht blind erzwungen).
	- Tagesprognose für morgen
		- Datenbereitstellung für Vorhersage für morgen

	## Nächster Umsetzungsschritt (Preis)
	- Danach produktionsnaher Ablauf fuer:
		- Streamlit GUI für die Vorhersagen 
			- In der Vergangenheit (seit 2025-10-01 bis gestern) und den Vergleich mit echten Preise (wie der Test)
			- des nächsten Tag (24h) 
	- Aufbau eines robusten Rolling-Origin Backtests fuer Stabilitaetspruefung der Score-Verbesserung.

## Wetterdaten

### Wettervariablen
- PV:
	- shortwave_radiation
	- direct_radiation
	- diffuse_radiation
	- cloud_cover
- Wind:
	- wind_speed_100m
	- wind_direction_100m

### Wetterdaten Aggregation
- PV, Wind Onshore und Wind Offshore Anlage Daten als csv aus Marktstammdatenregister herunterladen
	- zur Zeit Wind Onshore und Offshore aggregiert
	- Später prüfen, ob das separate Berechnen die Modellgenauigkeit verbessert 
- Anlage Clusters erkennen
- Jährliche Kapazitäten der Clusters und der Koordinaten der Cluster-Centroids als Gewichtung für Wetterdaten Aggregation  


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
	 - notebook/07_predict_price.ipynb
4. GUI App starten
	 - streamlit run src/streamlit_app_price.py
	 oder
	 - python -m streamlit run src/streamlit_app_price.p

## Hinweise
- Die zentrale Arbeitsdokumentation zu Entscheidungen und groesseren Aenderungen liegt im Ordner log/.
- Fuer den Verlauf und technische Leitlinien siehe:
	- log/DECISIONS.md
	- log/NEXT_STEPS.md
	- log/SESSION_LOG.md

---
Letzte Aktualisierung: 2026-06-02