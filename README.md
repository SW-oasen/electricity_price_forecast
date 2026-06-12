# Strompreisprognose Deutschland

## Projektüberblick

Dieses Projekt untersucht die Vorhersage stündlicher Day-Ahead-Strompreise für Deutschland mithilfe von Machine-Learning-Verfahren.

Als Folgeprojekt der Stromverbrauchsprognose steht hier die Analyse der Zusammenhänge zwischen Stromnachfrage, erneuerbarer Erzeugung, Wetterbedingungen und Marktpreisen im Mittelpunkt.

Neben einer möglichst genauen Vorhersage wird besonderer Wert auf eine nachvollziehbare Datenpipeline, reproduzierbare Modellierung und eine praxisnahe Prognoseumgebung gelegt.

---

## Zielsetzung

Der deutsche Strommarkt wird zunehmend durch wetterabhängige erneuerbare Energien geprägt. Insbesondere die starke Einspeisung von Wind- und Solarenergie führt zu:

* hoher Preisvolatilität,
* negativen Strompreisen,
* kurzfristigen Marktveränderungen,
* schwankender Residuallast.

Ziel des Projektes ist die Vorhersage der stündlichen Day-Ahead-Strompreise auf Basis historischer Markt-, Erzeugungs-, Nachfrage- und Wetterdaten.

---

## Datenquellen

### SMARD (Bundesnetzagentur)

Verwendete Daten:

* Day-Ahead-Strompreise
* Stromnachfrage
* Windstromerzeugung (Offshore + Onshore)
* Solarstromerzeugung
* Konventionelle Erzeugung

SMARD stellt offizielle Zeitreihen des deutschen Strommarktes bereit.

### Open-Meteo

Verwendete Wetterdaten:

#### Photovoltaik

* Globalstrahlung
* Direktstrahlung
* Diffusstrahlung
* Bewölkung

#### Windenergie

* Windgeschwindigkeit (100 m)
* Windrichtung (100 m)

### Marktstammdatenregister (MaStR)

Verwendet für:

* Anlagenstandorte
* installierte Leistungen
* gewichtete Wetteraggregation

---

## Fachlicher Ansatz

Die Strompreise werden durch das Zusammenspiel von Angebot und Nachfrage bestimmt.

Daher werden Informationen aus mehreren Bereichen kombiniert:

### Marktinformationen

* historische Strompreise
* Stromnachfrage
* Preis- und Nachfragelags

### Erzeugungsdaten

* Windstromerzeugung
* Solarstromerzeugung
* Residuallast

### Kalenderinformationen

* Stunde
* Wochentag
* Monat
* Feiertage
* Wochenenden

### Wetterinformationen

* Strahlungsdaten
* Bewölkung
* Windgeschwindigkeit
* Windrichtung

---

## Projektarchitektur

```text
SMARD
 ├─ Strompreise
 ├─ Stromnachfrage
 └─ Stromerzeugung
          │
          ▼

Marktstammdatenregister
          │
          ▼

Open-Meteo Wetterdaten
          │
          ▼

Feature Engineering
          │
          ▼

Machine-Learning-Modell
          │
          ▼

Historische Prognosen
und Tagesprognosen
          │
          ▼

Streamlit-Anwendung
```

---

## Aktueller Funktionsumfang

Bereits umgesetzt:

* automatisierte Datenbeschaffung
* SQLite-Datenbank
* Wetterdatenaggregation
* Feature Engineering
* Modelltraining
* historische Preisprognosen
* Tagesprognosen für den Folgetag
* interaktive Streamlit-Anwendung

---

## Streamlit-Anwendung

Die Anwendung ermöglicht:

* Analyse historischer Vorhersagen
* Vergleich von Prognose und Ist-Wert
* Vorhersage der nächsten 24 Stunden
* Interaktive Visualisierung der Ergebnisse

    ### start .venv
    * .\.venv\Scripts\Activate.ps1  

    ### GUI App starten
    * streamlit run src/streamlit_app_price.py
    * python -m streamlit run src/streamlit_app_price.p

---

## Verwendete Technologien

* Python
* Pandas
* Scikit-Learn
* LightGBM
* SQLite
* Streamlit
* Open-Meteo API
* SMARD API

---

## Projektstruktur

```text
data/
db/
documents/
models/
notebook/
presentation/
reports/
src/
util/
log/
```

---

## Geplante Erweiterungen

* Rolling-Origin-Backtesting
* Modellvergleich
* Ensemble-Ansätze
* Automatisierte Modellaktualisierung
* Erweiterte Marktmerkmale

---

## Autor

Yuchuan Liu

Persönliches Data-Science-Projekt im Bereich Energieanalytik und Machine Learning.

---

Letzte Aktualisierung: 2026-06-12
