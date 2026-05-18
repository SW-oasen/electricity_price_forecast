# Open-Meteo API

## Overview

**Open-Meteo** is a free, open-source weather API that provides historical weather data and forecasts without requiring an API key for non-commercial use.

- Website: [https://open-meteo.com](https://open-meteo.com)
- Documentation: [https://open-meteo.com/en/docs](https://open-meteo.com/en/docs)
- Authentication: None required (free tier, non-commercial)
- Rate limiting: Polite usage recommended (`1 s` sleep between city requests)

---

## API Endpoints Used

### 1. Historical Weather Archive

```
GET https://archive-api.open-meteo.com/v1/archive
```

Used to fetch hourly weather data from **2019-01-01 to 2025-09-30** to build the training dataset.

**Documentation:** [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

**Example request:**
```
https://archive-api.open-meteo.com/v1/archive
  ?latitude=52.5200
  &longitude=13.4050
  &start_date=2019-01-01
  &end_date=2025-09-30
  &hourly=apparent_temperature,rain,snowfall,wind_speed_10m,shortwave_radiation
  &timezone=auto
```

---

### 2. Weather Forecast

```
GET https://api.open-meteo.com/v1/forecast
```

Used to fetch future hourly weather forecasts for next-day electricity demand prediction.

**Documentation:** [Forecast API](https://open-meteo.com/en/docs)

**Example request:**
```
https://api.open-meteo.com/v1/forecast
  ?latitude=52.5200
  &longitude=13.4050
  &hourly=apparent_temperature,rain,snowfall,wind_speed_10m,shortwave_radiation
  &forecast_days=2
  &timezone=auto
```

- `forecast_days`: 1–16 (free tier maximum is 16 days ahead)
- Default in this project: `2` days (today + tomorrow)

---

## Query Parameters

| Parameter | Description |
|-----------|-------------|
| `latitude` | Location latitude (decimal degrees) |
| `longitude` | Location longitude (decimal degrees) |
| `start_date` | Archive start date (`YYYY-MM-DD`) |
| `end_date` | Archive end date (`YYYY-MM-DD`) |
| `hourly` | Comma-separated list of weather variables |
| `forecast_days` | Number of forecast days (forecast endpoint only) |
| `timezone` | Timezone for returned timestamps (`auto` = location-based) |

---

## Weather Variables Used

| Variable | Unit | Description |
|----------|------|-------------|
| `apparent_temperature` | °C | Perceived temperature (accounts for humidity and wind chill) |
| `rain` | mm | Rainfall per hour |
| `snowfall` | cm | Snowfall per hour |
| `wind_speed_10m` | km/h | Wind speed at 10 m above ground |
| `shortwave_radiation` | W/m² | Solar shortwave radiation at the surface |

> `temperature_2m` was dropped due to high correlation with `apparent_temperature` (see Notebook 02 EDA).

---

## API Response Format

```json
{
  "latitude": 52.52,
  "longitude": 13.405,
  "timezone": "Europe/Berlin",
  "hourly": {
    "time": ["2019-01-01T00:00", "2019-01-01T01:00", ...],
    "apparent_temperature": [-3.2, -3.5, ...],
    "rain": [0.0, 0.0, ...],
    "snowfall": [0.1, 0.0, ...],
    "wind_speed_10m": [12.3, 11.8, ...],
    "shortwave_radiation": [0.0, 0.0, ...]
  }
}
```

The `hourly` object is directly converted to a `pd.DataFrame` in the implementation.

---

## Implementation in This Project

### Cities Covered

Weather data is fetched for the **5 largest German cities** and then aggregated into a single Germany-wide representative value using population weights.

| City | Latitude | Longitude | Population |
|------|----------|-----------|------------|
| Berlin | 52.5200 | 13.4050 | 3,644,826 |
| Hamburg | 53.5511 | 9.9937 | 1,841,179 |
| München | 48.1351 | 11.5820 | 1,471,508 |
| Köln | 50.9375 | 6.9603 | 1,085,664 |
| Frankfurt | 50.1109 | 8.6821 | 753,056 |

Population-weighted aggregation ensures that cities with more inhabitants have greater influence on the national weather signal, better reflecting the distribution of electricity consumers across Germany.

### Key Functions — `src/fetch_prepare_data.py`

#### `fetch_weather_data_for_cities(in_selected_cities, in_start_date, in_end_date, in_weather_variables) → dict`

Fetches historical weather data for each city from the archive API. Returns a dictionary mapping city name to a `pd.DataFrame` with raw hourly weather data. Sleeps 1 second between city requests.

#### `fetch_weather_forecast_for_cities(in_selected_cities, in_weather_variables, forecast_days) → dict`

Fetches weather forecast data for each city from the forecast API. `forecast_days` defaults to `2`. Sleeps 1 second between city requests.

#### `merge_weather_data_with_city_weights(in_weather_city_dict, in_city_population, in_weather_variables) → pd.DataFrame`

Merges per-city DataFrames into a single Germany-wide DataFrame by computing a **population-weighted average** of each weather variable:

$$w_{\text{city}} = \frac{\text{population}_{\text{city}}}{\sum_i \text{population}_i}$$

$$\text{variable}_{\text{DE}} = \sum_{\text{city}} w_{\text{city}} \cdot \text{variable}_{\text{city}}$$

#### `create_weather_features(in_df) → pd.DataFrame`

Derives additional features from the raw weather variables:

| Derived Feature | Formula / Description |
|----------------|-----------------------|
| `apparent_temperature_lag_24h` | `apparent_temperature` shifted by 24 hours |
| `apparent_temperature_rolling_mean_24h` | 24-hour rolling mean of `apparent_temperature` (shifted by 1 to avoid leakage) |
| `shortwave_radiation_0m_lag_24h` | `shortwave_radiation` shifted by 24 hours |
| `shortwave_radiation_0m_rolling_mean_24h` | 24-hour rolling mean of `shortwave_radiation` (shifted by 1) |
| `heating_degree` | `max(0, 18 - apparent_temperature)` — proxy for heating energy demand |
| `cooling_degree` | `max(0, apparent_temperature - 25)` — proxy for cooling energy demand |

#### `prepare_weather_data(in_start_date, in_end_date, ...) → pd.DataFrame`

Full pipeline for historical weather: fetch → merge with weights → rename time column → sort → create features.

#### `prepare_weather_forecast(...) → pd.DataFrame`

Full pipeline for forecast weather: fetch forecast → merge with weights → rename time column.

---

## Usage Examples

### Fetch Historical Weather Data

```python
from src.fetch_prepare_data import prepare_weather_data

df_weather = prepare_weather_data(
    in_start_date="2019-01-01",
    in_end_date="2025-09-30"
)
```

### Fetch Weather Forecast (for next-day prediction)

```python
from src.fetch_prepare_data import prepare_weather_forecast

df_forecast = prepare_weather_forecast(forecast_days=2)
```

---

## Output Schema (after feature engineering)

| Column | Type | Description |
|--------|------|-------------|
| `time` | `datetime64` | Hourly timestamp (local timezone via `timezone=auto`) |
| `apparent_temperature` | `float64` | Weighted apparent temperature (°C) |
| `rain` | `float64` | Weighted rainfall (mm) |
| `snowfall` | `float64` | Weighted snowfall (cm) |
| `wind_speed_10m` | `float64` | Weighted wind speed (km/h) |
| `shortwave_radiation` | `float64` | Weighted solar radiation (W/m²) |
| `apparent_temperature_lag_24h` | `float64` | Temperature 24 h prior |
| `apparent_temperature_rolling_mean_24h` | `float64` | 24 h rolling mean temperature |
| `shortwave_radiation_0m_lag_24h` | `float64` | Solar radiation 24 h prior |
| `shortwave_radiation_0m_rolling_mean_24h` | `float64` | 24 h rolling mean solar radiation |
| `heating_degree` | `float64` | Heating degree value |
| `cooling_degree` | `float64` | Cooling degree value |

---

## Data Files

| File | Description |
|------|-------------|
| `data/processed/weather_data_de_2019_2025.csv` | Merged, population-weighted historical weather data |
| `data/processed/weather_data_mean_cities_2019_2025.csv` | Per-city mean aggregation (unweighted) |
| `data/raw/tmp/weather_data_Berlin.csv` | Raw per-city weather data (Berlin) |
| `data/raw/tmp/weather_data_Hamburg.csv` | Raw per-city weather data (Hamburg) |
| `data/raw/tmp/weather_data_München.csv` | Raw per-city weather data (München) |
| `data/raw/tmp/weather_data_Köln.csv` | Raw per-city weather data (Köln) |
| `data/raw/tmp/weather_data_Frankfurt.csv` | Raw per-city weather data (Frankfurt) |
| `data/processed/energy_weather_data_2026-05-07_for_prediction.csv` | Combined energy+weather data prepared for prediction |

---

## Notes

- The archive API returns timestamps in the **local timezone** of the requested location when `timezone=auto` is used. When merging with the energy data (UTC), ensure consistent timezone handling.
- The forecast API returns timestamps as UTC-aware datetime strings in the implementation (`utc=True` in `pd.to_datetime`).
- There is a known discrepancy between how the archive and forecast APIs handle timezones — this is handled consistently by `rename_time_column()` before merging.
- The free tier of Open-Meteo is sufficient for this project. No API key is required for non-commercial use.
