# Vorhersage des Strompreises in Deutschland

Die Erweiterung des Projekts Stromverbrauch-Vorhersage.

## Dokumentation
- Technische Umsetzung (Startphase): [umsetzung_preisdaten_smard.md](umsetzung_preisdaten_smard.md)
- API-Referenz: [smard_api.md](smard_api.md)

## Daten

### Der Zielvariabeln:
- Marktpreis: Deutschland/Luxemburg 

### Referenz:
SMARD Prognose 
- Gesamt Stromverbrauch: Netzlast
- Gesamt Stromerzeugung
    - Wind Onshore
    - Wind Offshore
    - Photovoltaik
    - Sonstige Konventionelle

### Datenquellen
- SMARD
- OPEN-METEO
- Wikipedia

### Die mögliche Prädikatoren:
- SMARD - Historische Daten
    - Strompreis - Filter 4169 day-ahead market clearing price (auction result)
    - Stromerzeugung: 
        - Wind Onshore
        - Wind Offshore
        - Photovoltaik
        - Sonstige Konventionelle
    - Stromverbrauch: 
        - Gesamt
        - evtl Residuallast
        - evtl Pumpspeicher
- OPEN-METEO - Historische Daten und Vorhersagen
    - Windgeschwindigkeit
    - Sonneneinstralung
    - Niederschlag
- evtl. extern Braukohle- und Gas-preise
- evtl. Import und Export

### Datenbereitstellung
- Aus der Datenbank energy_demand.db
- SMARD - seit 2019
- OPENT-Meteo - seit 2019 und Vorhersage
- Wikipedia: Windpark Onshore, Offshore, PV Anlagen
    - liste_der_deutschen_offshore_windparks
    - liste_der_deutschen_onshore_windparks
    - liste_Von_solarwerken_in_deutschland

## Vorgehen

### Parallelle und gestapelte Vorhersage
- Winderzeugung: Onshore, Offshore
- Solarerzeugung
- Residuallast
-> Strompreis

### Wetterdaten nach Region gewichten
- Die größte Windpark: Koordination und Kapazität - Offshore und Onshore
- Die größte PV-Anlagen: Koordination und Kapazität

## EDA

### Datenbereitstellung - Web Scraping > SQLite DB



## Baseline Modelle



## Evaluierung und Modellselektion



# Pipeline



# Applikation GUI


