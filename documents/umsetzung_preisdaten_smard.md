# Umsetzung Preisdaten (SMARD) - Startphase

## Ziel
Dieses Dokument beschreibt die erste, kleine Ausbaustufe fuer die Preisprognose.
Wir starten mit 5 SMARD-Reihen und erweitern spaeter Schritt fuer Schritt.

## Startumfang (Phase 1)

### Zielvariable
- Strompreis DE/LU: Filter `4169` (Day-Ahead Marktpreis)

### Erste Prädiktoren aus SMARD
- Wind Onshore: Filter `4067`
- Wind Offshore: Filter `1225`
- Photovoltaik: Filter `4068`
- Sonstige Konventionelle: Filter `1227`

## Wichtige API-Regel
- Pro Anfrage ist nur **eine** Filter-ID moeglich.
- Daher: pro Filter eine eigene Anfrage, danach Zusammenfuehrung ueber `time`.

## Datenhaltung in SQLite

### Empfehlung
- Bestehende Tabellen (Legacy): behalten.
- Neue gescrapte Reihen: in neuen, normalisierten Tabellen speichern.

### Neue Tabellen (Start)
1. `series_catalog`
Inhalt: `series_id`, `source`, `filter_id`, `region`, `resolution`, `unit`, `active`

2. `timeseries_values`
Inhalt: `time_utc`, `series_id`, `value`, `data_source`, `fetched_at`, `version`

3. `ingestion_runs`
Inhalt: `run_id`, `start_ts`, `end_ts`, `status`, `rows_loaded`, `error_text`

4. `data_quality_log`
Inhalt: `run_id`, `series_id`, `check_name`, `result`, `details`

## Namenskonvention (einfach)
- `price_de_lu_eur_mwh`
- `gen_wind_onshore_mwh`
- `gen_wind_offshore_mwh`
- `gen_pv_mwh`
- `gen_other_conventional_mwh`

## Nächste konkrete Schritte
1. Tabellen `series_catalog` und `timeseries_values` anlegen.
2. Die 5 Filter in `series_catalog` eintragen.
3. Historische Daten pro Filter laden (sequentiell, `sleep=0.3`).
4. Daten in `timeseries_values` schreiben.
5. Erst danach `ingestion_runs` und `data_quality_log` aktiv nutzen.

## Abgrenzung zu anderen Dokus
- Projekt-Kontext: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
- API-Details: [smard_api.md](smard_api.md)