# Backend: live Bengaluru air quality

The API uses OpenAQ v3 as a transport for CPCB/KSPCB reference-station data. It does not use the official-monitor historical CSV or K-Means artifacts to serve those live readings. A separate experimental forecast endpoint uses an Open-Meteo-trained PM2.5 model.

## Run locally

From PowerShell:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your OpenAQ key to `backend/.env` as `OPENAQ_API_KEY=...`, then start the server:

```powershell
uvicorn app.main:app --reload
```

Get an API key from the [OpenAQ API documentation](https://docs.openaq.org/using-the-api/api-key). Keep the key in the backend environment; it is not sent to the browser.

## Data behavior and limitations

- Only non-mobile monitor locations attributed to CPCB or KSPCB are accepted. Measurements older than six hours and unsupported units are discarded.
- Named areas are estimated from up to four nearby stations using inverse-square distance weighting, and only when a station is within 30 km. Areas without nearby pollutant readings remain unavailable; the API does not fill gaps with sample data.
- The model fits K-Means zones from the current complete station observations on each refresh. Results are cached for five minutes.
- The map can show PM2.5 concentration or an **indicative AQI** calculated as the highest available pollutant sub-index from at least three recent readings, including PM2.5 or PM10. It uses CPCB concentration breakpoints, but only latest measurements are available here; prescribed averaging periods and data-completeness rules are not applied, so this is **not official CPCB AQI**.
- OpenAQ may not expose eligible CPCB/KSPCB stations for Bengaluru at a given time. In that case the API reports a data error rather than showing fabricated readings.

`train_model.py` is an optional offline trainer. It requires at least three complete, CPCB/KSPCB-attributed historical rows in `data/historical_bengaluru_aqi.csv` with columns `location_name,observed_at,source,pm25,pm10,no2`; `wind_speed` is optional. Its saved artifacts are not used by the live API.

## Download Open-Meteo historical data

The optional collector downloads Bengaluru hourly PM2.5, PM10, NO2, wind, temperature, humidity, and precipitation data into a separate file:

```powershell
py fetch_open_meteo_history.py
```

By default, it fetches from August 2022 through yesterday (Asia/Kolkata) and creates or updates `data/open_meteo_bengaluru_history.csv`. To fetch a smaller date range:

```powershell
py fetch_open_meteo_history.py --start-date 2024-01-01 --end-date 2024-01-31
```

The output is deliberately separate from `historical_bengaluru_aqi.csv`: Open-Meteo values are modelled grid-cell estimates, not CPCB/KSPCB monitor observations. The global CAMS air-quality product is roughly 45 km spatial and 3-hourly temporal resolution, so it is suitable for experimenting with city-level patterns and forecasts, not distinguishing adjacent neighborhoods. Wind components (`wind_u`, `wind_v`) are derived from wind speed and direction. Re-running the collector updates matching timestamps without duplicating rows.

Open-Meteo air-quality data is based on CAMS global atmospheric composition data; weather comes from Open-Meteo's Historical Forecast API. Cite [Open-Meteo](https://open-meteo.com/) and the [CAMS data provider](https://atmosphere.copernicus.eu/) when presenting results. See the [Air Quality API](https://open-meteo.com/en/docs/air-quality-api) and [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) for data details and limitations.

## Train the PM2.5 forecasting model

After collecting the separate Open-Meteo dataset, train and evaluate city-level, three-hour and twelve-hour PM2.5 forecasters:

```powershell
py train_forecast_model.py
```

The script fits a separate model per horizon, uses a chronological 80/20 holdout, compares mean absolute error against persistence and same-time-yesterday baselines, then saves both models to `artifacts/pm25_forecaster.joblib`. The dashboard lets the user choose a 3-hour or 12-hour horizon via `GET /api/forecast?horizon_hours=3` or `GET /api/forecast?horizon_hours=12`. The endpoint fetches recent Open-Meteo inputs and needs no API key. The forecast card shows the selected model's holdout error and an approximate residual-based interval; it is a city-grid estimate, not an official or neighborhood reading. Re-run the trainer after refreshing the data to update both models.

The current dataset produced holdout MAEs of 3.31 µg/m³ at 3 hours (versus 4.71 persistence and 6.60 same-time-yesterday) and 5.44 µg/m³ at 12 hours (versus 8.45 persistence and 6.60 same-time-yesterday). These are evaluations against Open-Meteo historical data, not independent CPCB/KSPCB ground truth.
