# Projektübersicht: Strompreisprognose Deutschland

## Zielsetzung
Dieses Projekt hat das Ziel, stündliche Strompreise für Deutschland mithilfe von Machine Learning vorherzusagen. Die Prognosen basieren auf historischen Daten zu Strompreisen, Stromerzeugung, Energiebedarf und Wetterdaten. Die Ergebnisse sollen helfen, Preisschwankungen besser zu verstehen und fundierte Entscheidungen für Energieversorger, Unternehmen und Verbraucher zu ermöglichen.

## Projektstruktur
- **data/**: Roh- und verarbeitete Datensätze
- **db/**: SQLite-Datenbanken für ETL und Analyse
- **documents/**: Projektdokumentation, API-Beschreibungen, Pläne
- **electricity_demand_forecast/**: Teilprojekt zur Prognose des Strombedarfs
- **models/**: Trainierte Modelle und Modellartefakte
- **notebook/**: Jupyter Notebooks für EDA, Modellierung und Visualisierung
- **presentation/**: Präsentationsmaterialien
- **src/**: Quellcode für ETL, Modelltraining, Vorhersage und Streamlit-App
- **util/**: Hilfsfunktionen und API-Clients

## Hauptfunktionen
- **Datenbeschaffung**: Automatisches Laden und Aktualisieren von Strompreis-, Erzeugungs-, Nachfrage- und Wetterdaten (SMARD, Open-Meteo)
- **ETL-Pipeline**: Verarbeitung und Speicherung der Daten in einer SQLite-Datenbank
- **Explorative Datenanalyse (EDA)**: Untersuchung von Zusammenhängen, Ausreißererkennung, Zeitreihenanalyse
- **Feature Engineering**: Erstellung von Zeit- und Wettermerkmalen
- **Modellierung**: Training und Evaluierung von ML-Modellen zur Preisprognose
- **Interaktive Vorhersage**: Streamlit-App zur Visualisierung und Prognose

## Verwendete Technologien
- Python (pandas, scikit-learn, statsmodels, SQLAlchemy, Streamlit, matplotlib, seaborn)
- SQLite
- Jupyter Notebooks

## Hinweise zur Nutzung
1. Abhängigkeiten installieren: `pip install -r requirements_gpu.txt`
2. Datenbank aktualisieren: ETL-Skripte oder Notebooks ausführen
3. Notebooks für EDA und Modellierung nutzen
4. Streamlit-App starten: `streamlit run src/streamlit_app_demand.py`

## Autoren & Kontakt
- Projektleitung: Yuchuan Liu
- Kontakt: [Ihre E-Mail-Adresse]

---
*Letzte Aktualisierung: Mai 2026*