# Ausreißern Erkennung

---

# 1️⃣ Die negativen Preise folgen einem klaren Muster

Schauen wir uns die Zeitpunkte an:

| Datum      | Uhrzeit UTC |
| ---------- | ----------- |
| 28.05.2023 | 11–12 Uhr   |
| 02.07.2023 | 10–14 Uhr   |
| 01.05.2024 | 11–12 Uhr   |
| 12.05.2024 | 10–12 Uhr   |
| 06.04.2025 | 11–12 Uhr   |
| 27.04.2025 | 10–12 Uhr   |
| 11.05.2025 | 10–13 Uhr   |
| 26.04.2026 | 09–13 Uhr   |
| 01.05.2026 | 09–13 Uhr   |


* Frühling (April/Mai)
* Frühsommer (Mai/Juli)
* Mittagszeit

hohe Solarproduktion
+
geringer Verbrauch
=
negative Preise

---

# 2️⃣ Feiertage und Wochenenden

| Datum      | Wochentag  | Möglicher Anlass         |
| ---------- | ---------- | ------------------------ |
| 28.05.2023 | Sonntag    | Pfingstsonntag           |
| 29.05.2023 | Montag     | Pfingstmontag (Feiertag) |
| 02.07.2023 | Sonntag    | Wochenende               |
| 01.05.2024 | Mittwoch   | Tag der Arbeit           |
| 12.05.2024 | Sonntag    | Wochenende               |
| 06.04.2025 | Sonntag    | Wochenende               |
| 27.04.2025 | Sonntag    | Wochenende               |
| 01.05.2025 | Donnerstag | Tag der Arbeit           |
| 10.05.2025 | Samstag    | Wochenende               |
| 11.05.2025 | Sonntag    | Wochenende               |
| 05.04.2026 | Sonntag    | Ostersonntag             |
| 06.04.2026 | Montag     | Ostermontag              |
| 25.04.2026 | Samstag    | Wochenende               |
| 26.04.2026 | Sonntag    | Wochenende               |
| 01.05.2026 | Freitag    | Tag der Arbeit           |
| 02.05.2026 | Samstag    | Wochenende               |


---

Das bedeutet:

Industrieverbrauch ↓
Büroverbrauch ↓
PV-Erzeugung ↑

gleichzeitig.

---

# 3️⃣ Warum werden die Preise negativ?

Vereinfacht:

Deutschland kann Strom nicht beliebig speichern.

Wenn:

```text
Angebot > Nachfrage
```

muss Strom trotzdem ins Netz.

Dann zahlen Produzenten teilweise Geld dafür, dass ihnen jemand den Strom abnimmt.

Dadurch entstehen:

```text
-100 €/MWh
-200 €/MWh
-500 €/MWh
```

wie in deinen Daten. 

---

# 4️⃣ Die positiven Extrempreise zeigen das Gegenstück

Deine RANSAC-Ausreißer:

### August 2022

750–870 €/MWh

### November 2024

800 €/MWh

### Dezember 2024

936 €/MWh



---

## Warum August 2022?

Energiekrise
Gaskrise
Ukraine-Krieg
Nord Stream Probleme

---

Die Preise sind nicht zufällig:

Fast alle Ausreißer liegen Ende August 2022. 

Das ist ein historisches Marktregime.

---

# 5️⃣ Warum Dezember 2024?

Die Uhrzeiten:

```text
07 Uhr
14–17 Uhr
```



---

Typischer Fall:

```text
Winter
+
wenig Solar
+
hoher Verbrauch
+
Abendspitze
```

---

Das Gegenteil deiner negativen Preise.

---

# 6️⃣ Das liefert dir bereits Features für ML

Dein EDA zeigt eigentlich schon:

## Negative Preise

Features:

```text
Solarstrahlung
Monat
Wochenende
Feiertag
Stromverbrauch
```

---

## Positive Preise

Features:

```text
Gaspreis (wenn vorhanden)
Last
Winter
Wenig Solar
Windflaute
```

---

# 7️⃣ Das würde ich im Notebook untersuchen

## Negative Preise

Vergleich:

negative_prices
vs
normal_prices

für:

* Stromverbrauch
* Temperatur
* Solarstrahlung
* Wind

---

## Positive Preise

Vergleich:

```python
price > 500
vs
normal_prices
```

---

# 8️⃣ Meine Hypothese (vor der Modellierung)

Für Deutschland würde ich erwarten:

### Wichtigste Einflussgrößen

1. Stromverbrauch (Load)
2. Solarstrahlung
3. Windproduktion
4. Feiertag
5. Wochenende

---

Bei den negativen Preisen würde ich fast wetten:

```text
hohe Solarstrahlung
+
Wochenende/Feiertag
+
niedrige Last
```

erklärt einen großen Teil der Extremwerte.

---

# Für dein Projekt

Das ist Gold wert.

Denn du kannst später im README schreiben:

> "The EDA revealed that extreme negative prices were concentrated around spring and early summer weekends and holidays during midday hours, indicating a strong relationship between solar generation, reduced demand, and market oversupply."

Das ist genau die Art Business-Interpretation, die über reines Modelltraining hinausgeht.


The exploratory analysis revealed two distinct market conditions: scarcity-driven price spikes during the 2022 energy crisis and oversupply-driven negative prices during periods of high renewable generation and reduced demand. These findings guided the feature engineering process for the forecasting model.

Die extremen Preispeaks im August 2022 können nicht allein durch den Stromverbrauch erklärt werden. Der Verbrauch lag in ähnlichen Größenordnungen wie in anderen Jahren, die Preise waren jedoch um ein Vielfaches höher. Dies deutet auf externe Markteinflüsse wie die europäische Energiekrise, hohe Gaspreise und eingeschränkte Erzeugungskapazitäten hin.