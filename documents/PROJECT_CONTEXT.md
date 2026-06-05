# Projektkontext: Strompreisprognose Deutschland

## Projektziel
Vorhersage des stuendlichen Day-Ahead-Strompreises (DE/LU) auf Basis historischer Markt-, Erzeugungs-, Nachfrage- und Wetterdaten.

## Aktueller Stand (2026-06-02)
- Die Preis-ETL ist produktiv nutzbar und laeuft ueber src/etl_price.py.
- Wetterdaten werden technologiegetrennt (PV/Wind) verarbeitet, gewichtet und als eigene Serien gespeichert.
- Notebook 03 nutzt modulare Helferfunktionen aus util/weather_weighted.py statt lokaler Kernlogik.
- Preisprognose-Implementierung in Notebook 05 ist gestartet und lauffaehig.

## Datenquellen
- SMARD
    - Preis DE/LU
    - Erzeugung: Wind Onshore, Wind Offshore, PV, Sonstige Konventionelle
- Open-Meteo Archive
    - PV: shortwave_radiation, direct_radiation, diffuse_radiation, cloud_cover
    - Wind: wind_speed_100m, wind_direction_100m
- Struktur- und Standortdaten
    - verarbeitete Cluster-/Kapazitaetsdateien unter data/processed

## Datenhaltung
- Primäre ETL-Datenbank: db/energy_demand.db
- Kernschema:
    - series_catalog
    - timeseries_values
    - ingestion_runs
    - data_quality_log

## ETL-Logik (relevant)
- update_database(...) steuert den Ablauf.
- SMARD- und Open-Meteo-Ingestion sind entkoppelt.
    - Ein "SMARD up to date" blockiert Open-Meteo nicht.
- Open-Meteo Delta-Verhalten:
    - Zielstand ist standardmaessig gestern.
    - Delta wird tagbasiert bestimmt, um unnoetigen Re-Fetch desselben Kalendertags zu vermeiden.
- Open-Meteo Zeitgrenzen:
    - Clipping in Europe/Berlin auf [start_date 00:00, end_date+1d 00:00).

## Feature-Engineering-Richtung
- Windrichtung wird als zirkulare Groesse behandelt.
- Neuer Helper in util/weather_weighted.py:
    - aggregate_weighted_wind_vector_features(...)
    - Aggregation ueber u/v-Komponenten statt arithmetischem Winkelmittel
    - Rueckrechnung von Vektor-Windgeschwindigkeit/-richtung
    - Optionale Potenzmerkmale der Windgeschwindigkeit (pow2, pow3)

## Preisprognose (Notebook 05) - aktueller Zwischenstand
- Datenbasis:
    - `timeseries_values` (Preis, Erzeugung, Wetter)
    - `energy_demand` (Nachfrage + SMARD Forecast)
- Baseline-Merkmale:
    - Nachfrage-Proxy, PV-/Wind-Proxy (day-ahead via `shift(24)`), Residuallast
    - Preis-/Nachfrage-Lags (24h/168h)
    - Kalendermerkmale
- Enriched-Merkmale:
    - erweiterte Lag-Struktur (24h/48h/168h) fuer zentrale Signale
    - zusaetzliche Regime- und Kalendermerkmale
- Optionaler Wetterblock (Ablation):
    - direkte Wetterkanal-Merkmale
    - Wetter-Lags (24h/168h)
    - Windrichtung zyklisch (sin/cos)
    - Interaktion mit Residuallast
- Ergebnisbild (ein Split, 2025-10-01):
    - Baseline niedriger
    - Enriched besser
    - Enriched + Wetterblock aktuell klar besser
  -> naechster Pflichtschritt: Rolling-Origin Validierung fuer Robustheit.

## Validierte Punkte
- Import-Haertung fuer ETL (package-safe Imports mit Fallback)
- Zeitstempel-Angleichung SMARD/Open-Meteo fuer Tagesfenster
- Entkopplung der Ingestion-Zweige
- Open-Meteo-Tagesdelta statt Re-Fetch desselben Tages
- Notebook-Fehler behoben (Abfrage auf timeseries_values statt nicht vorhandener Tabellen)

## Offene naechste Schritte
- Rolling-Origin Backtest fuer Preisprognose aufsetzen und dokumentieren.
- Historische Preisprognosen als persistente Serien in DB schreiben.
- Tagesprognose fuer morgen (prodnaher Ablauf mit vorhergesagter Nachfrage/PV/Wind) implementieren.
- DST-Randfaelle (23/25h-Tage) gezielt testen und dokumentieren.
- Open-Meteo-Delta auf unvollstaendige letzte Tage pruefen (ueber reine Stunden-Heuristik hinaus).

## Verweise
- Technische Umsetzung: umsetzung_preisdaten_smard.md
- API-Referenz: smard_api.md
- Kontextverlauf und Entscheidungen:
    - ../log/DECISIONS.md
    - ../log/NEXT_STEPS.md
    - ../log/SESSION_LOG.md


