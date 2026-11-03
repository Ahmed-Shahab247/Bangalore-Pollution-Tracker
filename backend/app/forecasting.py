"""City-level PM2.5 forecasting from Open-Meteo historical data."""

from datetime import datetime, timedelta
import math
from pathlib import Path
from threading import Lock
from time import monotonic
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "open_meteo_bengaluru_history.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "pm25_forecaster.joblib"
TIMEZONE = ZoneInfo("Asia/Kolkata")
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZONS = (3, 12)
AIR_QUALITY_FIELDS = ("pm2_5", "pm10", "nitrogen_dioxide")
WEATHER_FIELDS = (
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
)
POLLUTANT_LAGS = (0, 1, 3, 6, 12, 21, 24)
FEATURE_COLUMNS = tuple(
    [
        f"{pollutant}_lag_{lag}h"
        for pollutant, lags in (
            ("pm25", POLLUTANT_LAGS),
            ("pm10", (0, 3)),
            ("no2", (0, 3)),
        )
        for lag in lags
    ]
    + [
        "wind_u_at_issue",
        "wind_v_at_issue",
        "temperature_at_issue",
        "humidity_at_issue",
        "precipitation_at_issue",
        "target_hour_sin",
        "target_hour_cos",
        "target_day_sin",
        "target_day_cos",
    ]
)
MODEL_SOURCE = (
    "Open-Meteo CAMS global modelled air quality and Open-Meteo weather forecast"
)
_cache_lock = Lock()
_cached_forecast = None
_cache_updated_at = 0.0
FORECAST_CACHE_SECONDS = 600


class ForecastError(Exception):
    """A safe, user-facing model or forecast-data failure."""


def _hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "observed_at" in frame.columns:
        timestamps = pd.to_datetime(frame.pop("observed_at"), utc=True).dt.tz_convert(
            TIMEZONE
        )
        frame.index = timestamps
    else:
        frame.index = pd.DatetimeIndex(frame.index).tz_convert(TIMEZONE)
    return frame.sort_index().asfreq("h")


def build_feature_frame(data: pd.DataFrame, horizon_hours: int):
    """Build leakage-safe features and a horizon-specific PM2.5 target."""
    frame = _hourly_frame(data.copy())
    features = pd.DataFrame(index=frame.index)
    for pollutant, lags in (
        ("pm25", POLLUTANT_LAGS),
        ("pm10", (0, 3)),
        ("no2", (0, 3)),
    ):
        for lag in lags:
            features[f"{pollutant}_lag_{lag}h"] = frame[pollutant].shift(lag)

    target_index = frame.index + pd.Timedelta(hours=horizon_hours)
    wind_speed = frame["wind_speed"]
    wind_direction = np.deg2rad(frame["wind_direction"])
    features["wind_u_at_issue"] = -wind_speed * np.sin(wind_direction)
    features["wind_v_at_issue"] = -wind_speed * np.cos(wind_direction)
    features["temperature_at_issue"] = frame["temperature_2m"]
    features["humidity_at_issue"] = frame["relative_humidity_2m"]
    features["precipitation_at_issue"] = frame["precipitation"]

    target_hour = target_index.hour.to_numpy() * (2 * math.pi / 24)
    target_day = target_index.dayofyear.to_numpy() * (2 * math.pi / 365.25)
    features["target_hour_sin"] = np.sin(target_hour)
    features["target_hour_cos"] = np.cos(target_hour)
    features["target_day_sin"] = np.sin(target_day)
    features["target_day_cos"] = np.cos(target_day)

    target = frame["pm25"].shift(-horizon_hours).rename("target_pm25")
    target_time = pd.Series(target_index, index=frame.index, name="target_time")
    return features.loc[:, FEATURE_COLUMNS], target, target_time


def load_training_data(path: Path = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast training data not found: {path}. "
            "Run fetch_open_meteo_history.py first."
        )
    data = pd.read_csv(path)
    required = {
        "observed_at",
        "pm25",
        "pm10",
        "no2",
        "wind_speed",
        "wind_direction",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Forecast data is missing columns: {', '.join(sorted(missing))}.")
    for column in required - {"observed_at"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _metrics(actual, predicted):
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def _train_horizon(raw: pd.DataFrame, horizon_hours: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    features, target, target_time = build_feature_frame(raw, horizon_hours)
    samples = features.copy()
    samples["target_pm25"] = target
    samples["target_time"] = target_time
    samples = samples.replace([np.inf, -np.inf], np.nan).dropna()
    samples = samples[samples["target_pm25"] >= 0]
    if len(samples) < 1000:
        raise ValueError(
            f"Only {len(samples)} complete forecast examples are available; "
            "at least 1,000 are required."
        )

    cutoff_index = int(len(samples) * 0.8)
    cutoff = samples["target_time"].iloc[cutoff_index]
    train = samples[samples["target_time"] < cutoff]
    test = samples[samples["target_time"] >= cutoff]
    if len(train) < 500 or len(test) < 100:
        raise ValueError(
            f"Not enough time-ordered samples for a reliable {horizon_hours}-hour holdout test."
        )

    estimator = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    estimator.fit(train.loc[:, FEATURE_COLUMNS], train["target_pm25"])
    test_prediction = np.maximum(
        0, estimator.predict(test.loc[:, FEATURE_COLUMNS])
    )
    persistence = test["pm25_lag_0h"].to_numpy()
    seasonal_baseline = test[f"pm25_lag_{24 - horizon_hours}h"].to_numpy()
    model_metrics = _metrics(test["target_pm25"], test_prediction)
    persistence_metrics = _metrics(test["target_pm25"], persistence)
    seasonal_metrics = _metrics(test["target_pm25"], seasonal_baseline)
    best_baseline_mae = min(
        persistence_metrics["mae"], seasonal_metrics["mae"]
    )

    residuals = test["target_pm25"].to_numpy() - test_prediction
    interval_low, interval_high = np.quantile(residuals, [0.05, 0.95])
    beats_baselines = model_metrics["mae"] < best_baseline_mae

    final_model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    final_model.fit(samples.loc[:, FEATURE_COLUMNS], samples["target_pm25"])
    return {
        "model": final_model,
        "feature_columns": list(FEATURE_COLUMNS),
        "trained_through": samples["target_time"].max().isoformat(),
        "sample_count": len(samples),
        "horizon_hours": horizon_hours,
        "model_metrics": model_metrics,
        "persistence_metrics": persistence_metrics,
        "seasonal_metrics": seasonal_metrics,
        "beats_baselines": beats_baselines,
        "residual_interval": [float(interval_low), float(interval_high)],
        "source": MODEL_SOURCE,
    }


def train_forecaster(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH):
    raw = load_training_data(data_path)
    models = {
        horizon: _train_horizon(raw, horizon)
        for horizon in FORECAST_HORIZONS
    }
    bundle = {"models": models, "supported_horizons": list(FORECAST_HORIZONS)}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def _request_hourly(url: str, fields: tuple[str, ...]):
    params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "hourly": ",".join(fields),
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    if url == AIR_QUALITY_URL:
        params["domains"] = "cams_global"
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise ForecastError(
            "Could not retrieve current Bengaluru forecast inputs from Open-Meteo."
        ) from error
    except ValueError as error:
        raise ForecastError("Open-Meteo returned an invalid forecast response.") from error

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ForecastError("Open-Meteo returned an unexpected hourly data format.")
    for field in fields:
        values = hourly.get(field)
        if not isinstance(values, list) or len(values) != len(hourly["time"]):
            raise ForecastError(f"Open-Meteo did not return valid {field} data.")
    return hourly


def _load_bundle():
    if not MODEL_PATH.exists():
        raise ForecastError(
            "The PM2.5 forecasting model is not trained. "
            "Run `python train_forecast_model.py` from the backend folder."
        )
    try:
        return joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError) as error:
        raise ForecastError(
            "The PM2.5 forecasting model is incompatible with this Python environment. "
            "From the backend folder, retrain it using the same Python environment "
            "that runs the API: `python train_forecast_model.py`."
        ) from error


def get_pm25_forecast(horizon_hours: int = 3, force_refresh: bool = False):
    global _cached_forecast, _cache_updated_at
    if horizon_hours not in FORECAST_HORIZONS:
        raise ForecastError(
            f"Unsupported forecast horizon. Choose one of: {', '.join(map(str, FORECAST_HORIZONS))} hours."
        )
    with _cache_lock:
        if (
            not force_refresh
            and _cached_forecast is not None
            and horizon_hours in _cached_forecast
            and monotonic() - _cache_updated_at < FORECAST_CACHE_SECONDS
        ):
            return _cached_forecast[horizon_hours]

        model_bundle = _load_bundle()
        bundle = model_bundle.get("models", {}).get(horizon_hours)
        if bundle is None:
            raise ForecastError(
                "The forecasting model does not include that horizon. "
                "From the backend folder, run `python train_forecast_model.py`."
            )
        air = _request_hourly(AIR_QUALITY_URL, AIR_QUALITY_FIELDS)
        weather = _request_hourly(WEATHER_URL, WEATHER_FIELDS)
        if air["time"] != weather["time"]:
            raise ForecastError("Air-quality and weather forecast times do not align.")

        timestamps = pd.to_datetime(air["time"]).tz_localize(TIMEZONE)
        data = pd.DataFrame(
            {
                "pm25": air["pm2_5"],
                "pm10": air["pm10"],
                "no2": air["nitrogen_dioxide"],
                "wind_speed": weather["wind_speed_10m"],
                "wind_direction": weather["wind_direction_10m"],
                "temperature_2m": weather["temperature_2m"],
                "relative_humidity_2m": weather["relative_humidity_2m"],
                "precipitation": weather["precipitation"],
            },
            index=timestamps,
        )
        features, _, target_times = build_feature_frame(data, horizon_hours)
        now = datetime.now(TIMEZONE).replace(minute=0, second=0, microsecond=0)
        available = features.notna().all(axis=1) & (features.index <= now)
        candidates = features.loc[available]
        if candidates.empty:
            raise ForecastError("Open-Meteo has not provided enough recent data to forecast.")
        issued_at = candidates.index[-1]
        if now - issued_at.to_pydatetime() > timedelta(hours=6):
            raise ForecastError(
                "The latest complete Open-Meteo air-quality input is over six hours old."
            )

        prediction = max(
            0.0,
            float(bundle["model"].predict(candidates.iloc[[-1]])[0]),
        )
        residual_low, residual_high = bundle["residual_interval"]
        row_index = features.index.get_loc(issued_at)
        forecast_time = target_times.iloc[row_index]
        result = {
            "location": "Bengaluru",
            "issued_at": issued_at.isoformat(),
            "forecast_for": forecast_time.isoformat(),
            "horizon_hours": horizon_hours,
            "current_pm25": float(candidates.iloc[-1]["pm25_lag_0h"]),
            "pm25_prediction": round(prediction, 1),
            "prediction_interval": {
                "low": round(max(0.0, prediction + float(residual_low)), 1),
                "high": round(max(0.0, prediction + float(residual_high)), 1),
                "coverage": "approximate 90% holdout residual interval",
            },
            "model_mae": round(bundle["model_metrics"]["mae"], 2),
            "persistence_mae": round(bundle["persistence_metrics"]["mae"], 2),
            "seasonal_mae": round(bundle["seasonal_metrics"]["mae"], 2),
            "beats_baselines": bundle["beats_baselines"],
            "training_examples": bundle["sample_count"],
            "trained_through": bundle["trained_through"],
            "source": bundle["source"],
            "data_note": (
                "Modelled city-grid forecast; roughly 45 km air-quality grid. "
                "Not a monitor reading, official AQI, or neighborhood prediction."
            ),
        }
        if _cached_forecast is None:
            _cached_forecast = {}
        _cached_forecast[horizon_hours] = result
        _cache_updated_at = monotonic()
        return result
