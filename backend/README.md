# Backend: live Bengaluru air quality

The API uses OpenAQ v3 as a transport for CPCB/KSPCB reference-station data. It does not use the historical CSV or the saved offline model artifacts to serve live readings.

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
