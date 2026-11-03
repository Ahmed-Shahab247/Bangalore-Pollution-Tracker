# Backend: live Bengaluru air quality

The API uses OpenAQ v3 as a transport for CPCB/KSPCB reference-station data. It does not use the official-monitor historical CSV or K-Means artifacts to serve those live readings. The experimental forecast is trained against historical CPCB/KSPCB-attributed station PM2.5 targets, using Open-Meteo weather variables as exogenous predictors.

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

- Only non-mobile monitor locations attributed to CPCB or KSPCB are accepted. Measurements older than 12 hours and unsupported units are discarded; area readings show their observation age.
- Named areas are estimated from up to four nearby stations using inverse-square distance weighting, and only when a station is within 30 km. Areas without nearby pollutant readings remain unavailable; the API does not fill gaps with sample data.
- The model fits K-Means zones from the current complete station observations on each refresh. Results are cached for five minutes.
- The map can show PM2.5 concentration or an **indicative AQI** calculated as the highest available pollutant sub-index from at least three recent readings, including PM2.5 or PM10. It uses CPCB concentration breakpoints, but only latest measurements are available here; prescribed averaging periods and data-completeness rules are not applied, so this is **not official CPCB AQI**.
- OpenAQ may not expose eligible CPCB/KSPCB stations for Bengaluru at a given time. In that case the API reports a data error rather than showing fabricated readings.
- `GET /api/areas` returns the static neighborhood directory even when live station measurements are unavailable, so the browser can still identify the closest listed neighborhood. It does not invent an air-quality score when there is no usable station estimate.

`train_model.py` is an optional offline trainer. It requires at least three complete, CPCB/KSPCB-attributed historical rows in `data/historical_bengaluru_aqi.csv` with columns `location_name,observed_at,source,pm25,pm10,no2`; `wind_speed` is optional. Its saved artifacts are not used by the live API.

## Download Open-Meteo historical data

The optional collector downloads Bengaluru hourly modelled pollutants and weather predictors (wind, temperature, surface pressure, boundary-layer height, humidity, and precipitation) into a separate file:

```powershell
py fetch_open_meteo_history.py
```

By default, it fetches from August 2022 through yesterday (Asia/Kolkata) and creates or updates `data/open_meteo_bengaluru_history.csv`. To fetch a smaller date range:

```powershell
py fetch_open_meteo_history.py --start-date 2024-01-01 --end-date 2024-01-31
```

The output is deliberately separate from `historical_bengaluru_aqi.csv`: Open-Meteo values are modelled grid-cell estimates, not CPCB/KSPCB monitor observations. The global CAMS air-quality product is roughly 45 km spatial and 3-hourly temporal resolution, so it is suitable for experimenting with city-level patterns and forecasts, not distinguishing adjacent neighborhoods. Wind components (`wind_u`, `wind_v`) are derived from wind speed and direction. Re-running the collector updates matching timestamps without duplicating rows.

Open-Meteo air-quality data is based on CAMS global atmospheric composition data; weather comes from Open-Meteo's Historical Forecast API. Cite [Open-Meteo](https://open-meteo.com/) and the [CAMS data provider](https://atmosphere.copernicus.eu/) when presenting results. See the [Air Quality API](https://open-meteo.com/en/docs/air-quality-api) and [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) for data details and limitations.

## Train and evaluate the PM2.5 forecasting model

The trainer keeps the two data sources separate:

- `data/open_meteo_bengaluru_history.csv` supplies weather-only features (`wind_speed`, wind components, temperature, surface pressure, boundary-layer height, humidity, precipitation, and future-known calendar features). CAMS/Open-Meteo pollutant concentrations are never model inputs or targets.
- `data/historical_bengaluru_aqi.csv` supplies the measured PM2.5 target. Rows must be attributed to CPCB/KSPCB (OpenAQ may be the transport), and the hourly target is the mean across available Bengaluru reference stations. This file must contain real ground-station observations; the checked-in header-only file is not training data.

The ground-station CSV requires `location_name,observed_at,source,pm25`. Historical archives may omit `retrieved_at`; for latest/snapshot data, include `retrieved_at`, and rows whose sensor observation was more than 24 hours old when retrieved are discarded. Historical archived readings are not discarded merely because their observation date is in the past.

The evaluation is an expanding-window chronological backtest over each available June–September monsoon and November–February winter/post-monsoon window. Every holdout is trained only on earlier targets; models are compared to persistence, the same-hour-previous-day baseline, and a Ridge regression. MAE and RMSE are reported per season and pooled across evaluated holdouts for both horizons. Holdouts are not used to fit the reported models; the deployed estimators are refit on all valid history after evaluation.

Run:

```powershell
py train_forecast_model.py
```

You can override the default data locations with `--weather`, `--ground-stations`, and `--model`. The trainer stops without saving a model if ground observations are absent, source attribution is invalid, data validation fails, or there is not enough chronological coverage for both seasonal regimes. Re-run `py fetch_open_meteo_history.py` to refresh the weather CSV and add attributed station observations to the ground-station CSV before training.

The trainer saves the HistGradientBoosting and Ridge estimators and the full seasonal evaluation to `artifacts/pm25_forecaster.joblib`. The forecast endpoint uses current/future Open-Meteo weather inputs (no CAMS pollutant values) and returns the selected model's seasonal holdout metrics. Choose 3- or 12-hour forecasts via `GET /api/forecast?horizon_hours=3` or `GET /api/forecast?horizon_hours=12`; no Open-Meteo API key is required. The result is a weather-driven city-level estimate for the mean across monitors, not an official AQI or neighborhood-specific reading.

The prior artifact was trained against modelled CAMS PM2.5 and is intentionally rejected by the new runtime schema. `GET /api/forecast/status` reports model readiness; the dashboard disables horizon selection and explains the missing ground-station training data when the model is unavailable. It will remain unavailable until the model is retrained with real CPCB/KSPCB observations. Do not report accuracy until the new seasonal ground-station evaluation has completed; Open-Meteo weather predictors do not make the target independent if modelled pollution concentrations are used as labels.

## Build a unified ground-station training dataset

With `OPENAQ_API_KEY` set in `backend/.env`, fetch up to 12 or 24 months of CPCB/KSPCB-attributed OpenAQ PM2.5 and PM10 observations, match each station-hour to the existing Open-Meteo weather CSV, and save the combined dataset:

```powershell
py build_ground_station_training_data.py --months 12 --stations Peenya "Silk Board"
```

The default output is `data/bengaluru_ground_station_training.csv`. Use `--months 24` for two years, omit `--stations` to request every eligible monitor, or provide other case-insensitive station-name filters. Override the input weather file or output path with `--weather` and `--output`. The script includes only fixed CPCB/KSPCB monitors, normalizes station observations to local hourly timestamps, and drops invalid concentrations, unsupported units, measurement periods longer than three hours, and rows without complete matching weather data. Historical readings inside the requested date range are not discarded simply because they are more than 24 hours old; they are the intended training observations.

If OpenAQ has no valid station measurements overlapping the selected period and weather data, the script reports that no output was written. It does not substitute CAMS values or create a synthetic dataset. The resulting CSV can be passed as the target data to `py train_forecast_model.py --ground-stations data/bengaluru_ground_station_training.csv`; the seasonal trainer still requires enough chronological ground-station coverage for both monsoon and winter/post-monsoon holdouts.
