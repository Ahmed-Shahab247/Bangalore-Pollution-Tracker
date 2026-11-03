"""Ground-station PM2.5 forecasting with seasonal chronological evaluation."""

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
WEATHER_DATA_PATH = BASE_DIR / "data" / "open_meteo_bengaluru_history.csv"
GROUND_STATION_DATA_PATH = BASE_DIR / "data" / "historical_bengaluru_aqi.csv"
DATA_PATH = WEATHER_DATA_PATH
MODEL_PATH = BASE_DIR / "artifacts" / "pm25_forecaster.joblib"
TIMEZONE = ZoneInfo("Asia/Kolkata")
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZONS = (3, 12)
MINIMUM_TRAIN_SAMPLES = 500
MINIMUM_SEASON_TEST_SAMPLES = 24
MINIMUM_TOTAL_SAMPLES = 1000
MODEL_SCHEMA_VERSION = 2
WEATHER_FEATURE_COLUMNS = (
    "wind_speed",
    "wind_u",
    "wind_v",
    "temperature_2m",
    "surface_pressure",
    "boundary_layer_height",
    "relative_humidity_2m",
    "precipitation",
)
REQUIRED_WEATHER_FEATURE_COLUMNS = (
    "wind_speed",
    "wind_u",
    "wind_v",
    "temperature_2m",
    "surface_pressure",
    "boundary_layer_height",
)
REQUIRED_WEATHER_INPUT_COLUMNS = (
    "wind_speed",
    "temperature_2m",
    "surface_pressure",
    "boundary_layer_height",
)
TIME_FEATURE_COLUMNS = (
    "target_hour_sin",
    "target_hour_cos",
    "target_day_sin",
    "target_day_cos",
)
FEATURE_COLUMNS = WEATHER_FEATURE_COLUMNS + TIME_FEATURE_COLUMNS
WEATHER_API_FIELDS = {
    "wind_speed": "wind_speed_10m",
    "wind_direction": "wind_direction_10m",
    "temperature_2m": "temperature_2m",
    "surface_pressure": "surface_pressure",
    "boundary_layer_height": "boundary_layer_height",
    "relative_humidity_2m": "relative_humidity_2m",
    "precipitation": "precipitation",
}
MODEL_SOURCE = (
    "CPCB/KSPCB-attributed OpenAQ ground-station PM2.5 targets; "
    "Open-Meteo weather predictors"
)
_cache_lock = Lock()
_cached_forecast = None
_cache_updated_at = 0.0
FORECAST_CACHE_SECONDS = 600


class ForecastError(Exception):
    """A safe, user-facing model or forecast-data failure."""


def _hourly_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Convert observations to a de-duplicated local-hourly frame."""
    data = frame.copy()
    if "observed_at" in data.columns:
        timestamps = pd.to_datetime(data.pop("observed_at"), utc=True, errors="coerce")
        valid = timestamps.notna()
        data = data.loc[valid]
        data.index = timestamps.loc[valid].dt.tz_convert(TIMEZONE)
    else:
        timestamps = pd.to_datetime(data.index, utc=True, errors="coerce")
        valid = ~timestamps.isna()
        data = data.loc[valid]
        data.index = timestamps[valid].tz_convert(TIMEZONE)

    data.index = data.index.floor("h")
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.groupby(level=0).mean(numeric_only=True).sort_index()
    if data.empty:
        raise ValueError(f"No valid timestamped {label} records were found.")
    return data.asfreq("h")


def load_training_data(
    weather_path: Path = WEATHER_DATA_PATH,
    ground_station_path: Path = GROUND_STATION_DATA_PATH,
):
    """Load exogenous weather and attributed monitor observations separately."""
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Historical weather data not found: {weather_path}. "
            "Run fetch_open_meteo_history.py first."
        )
    if not ground_station_path.exists():
        raise FileNotFoundError(
            f"Ground-station PM2.5 data not found: {ground_station_path}. "
            "Provide historical CPCB/KSPCB-attributed station observations."
        )

    ground = pd.read_csv(ground_station_path)
    required_ground = {"location_name", "observed_at", "source", "pm25"}
    missing_ground = required_ground.difference(ground.columns)
    if missing_ground:
        raise ValueError(
            "Ground-station data is missing required columns: "
            f"{', '.join(sorted(missing_ground))}."
        )
    if ground.empty:
        raise ValueError(
            "No ground-station observations are available. "
            "data/historical_bengaluru_aqi.csv currently contains no PM2.5 rows; "
            "modeled Open-Meteo/CAMS pollutant values are not accepted as targets."
        )

    weather = pd.read_csv(weather_path)
    required_weather = {"observed_at", *REQUIRED_WEATHER_INPUT_COLUMNS}
    missing_weather = required_weather.difference(weather.columns)
    if missing_weather:
        raise ValueError(
            "Historical weather data is missing required exogenous variables: "
            f"{', '.join(sorted(missing_weather))}. "
            "Re-run fetch_open_meteo_history.py to collect the updated schema."
        )
    if "wind_u" not in weather or "wind_v" not in weather:
        if not {"wind_speed", "wind_direction"}.issubset(weather.columns):
            raise ValueError(
                "Historical weather data needs wind_u/wind_v or wind_speed/"
                "wind_direction columns."
            )
        direction = np.deg2rad(pd.to_numeric(weather["wind_direction"], errors="coerce"))
        speed = pd.to_numeric(weather["wind_speed"], errors="coerce")
        weather["wind_u"] = -speed * np.sin(direction)
        weather["wind_v"] = -speed * np.cos(direction)
    for optional in WEATHER_FEATURE_COLUMNS:
        if optional not in weather:
            weather[optional] = np.nan
    weather_frame = _hourly_frame(
        weather.loc[:, ["observed_at", *WEATHER_FEATURE_COLUMNS]],
        "historical weather",
    )
    if not weather_frame.loc[:, REQUIRED_WEATHER_FEATURE_COLUMNS].notna().any().all():
        raise ValueError(
            "Historical weather data is missing one or more required exogenous "
            "variables. Refresh it with fetch_open_meteo_history.py."
        )

    ground["observed_at"] = pd.to_datetime(
        ground["observed_at"], utc=True, errors="coerce"
    )
    ground["location_name"] = (
        ground["location_name"].astype("string").str.strip().replace("", pd.NA)
    )
    official_source = ground["source"].fillna("").astype(str).str.contains(
        r"\b(?:cpcb|central pollution control board|kspcb|karnataka state pollution control board)\b",
        case=False,
        regex=True,
    )
    ground = ground.loc[official_source].copy()
    if ground.empty:
        raise ValueError(
            "No CPCB/KSPCB-attributed ground-station rows found. "
            "OpenAQ may be the transport, but the station operator must be attributed."
        )

    ground["pm25"] = pd.to_numeric(ground["pm25"], errors="coerce")
    valid = (
        ground["observed_at"].notna()
        & ground["location_name"].notna()
        & ground["pm25"].notna()
        & np.isfinite(ground["pm25"])
        & (ground["pm25"] >= 0)
        & (ground["observed_at"] <= pd.Timestamp.now(tz="UTC"))
    )
    ground = ground.loc[valid].copy()

    if "retrieved_at" in ground.columns:
        retrieved_at = pd.to_datetime(
            ground["retrieved_at"], utc=True, errors="coerce"
        )
        record_age = retrieved_at - ground["observed_at"]
        fresh = (
            retrieved_at.notna()
            & (record_age >= pd.Timedelta(0))
            & (record_age <= pd.Timedelta(hours=24))
        )
        ground = ground.loc[fresh].copy()
    elif "data_type" in ground.columns:
        snapshot_types = {"latest", "current", "snapshot"}
        is_snapshot = ground["data_type"].fillna("").astype(str).str.casefold().isin(
            snapshot_types
        )
        if is_snapshot.any():
            raise ValueError(
                "Ground-station snapshot rows must include retrieved_at so readings "
                "older than 24 hours can be excluded. Historical archives may omit it."
            )

    if ground.empty:
        raise ValueError(
            "No valid, fresh CPCB/KSPCB PM2.5 observations remain after validation."
        )

    ground["observed_at"] = (
        ground["observed_at"].dt.tz_convert(TIMEZONE).dt.floor("h")
    )
    station_hourly = (
        ground.groupby(["location_name", "observed_at"], observed=True)["pm25"]
        .mean()
        .rename("pm25")
        .reset_index()
    )
    city_hourly = station_hourly.groupby("observed_at")["pm25"].mean().sort_index()
    city_hourly = city_hourly.asfreq("h")
    return weather_frame, city_hourly, int(station_hourly["location_name"].nunique())


def build_feature_frame(weather: pd.DataFrame, horizon_hours: int):
    """Create issue-time weather and future-known calendar features only."""
    if horizon_hours not in FORECAST_HORIZONS:
        raise ValueError(f"Unsupported forecast horizon: {horizon_hours}.")
    frame = _hourly_frame(weather.copy(), "weather")
    missing = set(WEATHER_FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            f"Weather input is missing feature columns: {', '.join(sorted(missing))}."
        )

    features = frame.loc[:, WEATHER_FEATURE_COLUMNS].copy()
    target_index = features.index + pd.Timedelta(hours=horizon_hours)
    target_hour = target_index.hour.to_numpy() * (2 * math.pi / 24)
    target_day = target_index.dayofyear.to_numpy() * (2 * math.pi / 365.25)
    features["target_hour_sin"] = np.sin(target_hour)
    features["target_hour_cos"] = np.cos(target_hour)
    features["target_day_sin"] = np.sin(target_day)
    features["target_day_cos"] = np.cos(target_day)
    return features.loc[:, FEATURE_COLUMNS], pd.DatetimeIndex(target_index)


def _metrics(actual, predicted):
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def _make_samples(weather: pd.DataFrame, city_pm25: pd.Series, horizon_hours: int):
    features, target_times = build_feature_frame(weather, horizon_hours)
    samples = features.copy()
    samples["target_pm25"] = city_pm25.reindex(target_times).to_numpy()
    samples["target_time"] = target_times
    samples["persistence_pm25"] = city_pm25.reindex(features.index).to_numpy()
    seasonal_times = target_times - pd.Timedelta(hours=24)
    samples["seasonal_pm25"] = city_pm25.reindex(seasonal_times).to_numpy()
    samples = samples.replace([np.inf, -np.inf], np.nan).dropna()
    samples = samples[samples["target_pm25"] >= 0]
    return samples


def _season_windows(start: pd.Timestamp, end: pd.Timestamp):
    """Yield June-Sep monsoon and Nov-Feb winter/post-monsoon windows."""
    for year in range(start.year, end.year + 1):
        windows = (
            (
                f"monsoon_{year}",
                "monsoon",
                pd.Timestamp(year=year, month=6, day=1, tz=TIMEZONE),
                pd.Timestamp(year=year, month=10, day=1, tz=TIMEZONE),
            ),
            (
                f"winter_{year}_{year + 1}",
                "winter_post_monsoon",
                pd.Timestamp(year=year, month=11, day=1, tz=TIMEZONE),
                pd.Timestamp(year=year + 1, month=3, day=1, tz=TIMEZONE),
            ),
        )
        for key, regime, window_start, window_end in windows:
            if window_end > start and window_start <= end:
                yield key, regime, window_start, window_end


def _new_estimators():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


def _fit_evaluate_horizon(samples: pd.DataFrame, horizon_hours: int):
    if len(samples) < MINIMUM_TOTAL_SAMPLES:
        raise ValueError(
            f"Only {len(samples)} complete ground-station forecast examples are "
            f"available for {horizon_hours} hours; at least "
            f"{MINIMUM_TOTAL_SAMPLES} are required."
        )

    model_names = ("hist_gradient_boosting", "ridge", "persistence", "seasonal_24h")
    predictions = {name: [] for name in model_names}
    actuals = []
    season_results = {}
    skipped_seasons = {}

    for key, regime, window_start, window_end in _season_windows(
        samples["target_time"].min(), samples["target_time"].max()
    ):
        test = samples[
            (samples["target_time"] >= window_start)
            & (samples["target_time"] < window_end)
        ]
        train = samples[samples["target_time"] < window_start]
        if test.empty:
            continue
        if len(train) < MINIMUM_TRAIN_SAMPLES:
            skipped_seasons[key] = (
                f"Only {len(train)} earlier training examples; "
                f"{MINIMUM_TRAIN_SAMPLES} required."
            )
            continue
        if len(test) < MINIMUM_SEASON_TEST_SAMPLES:
            skipped_seasons[key] = (
                f"Only {len(test)} complete test examples; "
                f"{MINIMUM_SEASON_TEST_SAMPLES} required."
            )
            continue
        if train["target_time"].max() >= test["target_time"].min():
            raise ValueError(f"Temporal leakage detected in holdout {key}.")

        model_predictions = {}
        for name, estimator in _new_estimators().items():
            estimator.fit(train.loc[:, FEATURE_COLUMNS], train["target_pm25"])
            model_predictions[name] = np.maximum(
                0, estimator.predict(test.loc[:, FEATURE_COLUMNS])
            )
        model_predictions["persistence"] = test["persistence_pm25"].to_numpy()
        model_predictions["seasonal_24h"] = test["seasonal_pm25"].to_numpy()
        actual = test["target_pm25"].to_numpy()
        actuals.extend(actual)
        for name in model_names:
            predictions[name].extend(model_predictions[name])
        season_results[key] = {
            "regime": regime,
            "test_start": window_start.isoformat(),
            "test_end_exclusive": window_end.isoformat(),
            "train_examples": int(len(train)),
            "test_examples": int(len(test)),
            "metrics": {
                name: _metrics(actual, prediction)
                for name, prediction in model_predictions.items()
            },
        }

    evaluated_regimes = {result["regime"] for result in season_results.values()}
    required_regimes = {"monsoon", "winter_post_monsoon"}
    if not required_regimes.issubset(evaluated_regimes):
        missing_regimes = sorted(required_regimes - evaluated_regimes)
        raise ValueError(
            "Insufficient chronological ground-station coverage for seasonal "
            f"evaluation; missing valid holdouts for: {', '.join(missing_regimes)}. "
            f"Skipped windows: {skipped_seasons or 'none'}."
        )

    overall_metrics = {
        name: _metrics(actuals, predicted)
        for name, predicted in predictions.items()
    }
    estimator_bundle = _new_estimators()
    for estimator in estimator_bundle.values():
        estimator.fit(samples.loc[:, FEATURE_COLUMNS], samples["target_pm25"])

    residuals = np.asarray(actuals) - np.asarray(predictions["hist_gradient_boosting"])
    residual_interval = np.quantile(residuals, [0.05, 0.95])
    return {
        "model": estimator_bundle["hist_gradient_boosting"],
        "comparison_model": estimator_bundle["ridge"],
        "feature_columns": list(FEATURE_COLUMNS),
        "horizon_hours": horizon_hours,
        "sample_count": int(len(samples)),
        "trained_through": samples["target_time"].max().isoformat(),
        "selected_model": "hist_gradient_boosting",
        "model_metrics": overall_metrics["hist_gradient_boosting"],
        "ridge_metrics": overall_metrics["ridge"],
        "persistence_metrics": overall_metrics["persistence"],
        "seasonal_metrics": overall_metrics["seasonal_24h"],
        "beats_baselines": overall_metrics["hist_gradient_boosting"]["mae"]
        < min(
            overall_metrics["persistence"]["mae"],
            overall_metrics["seasonal_24h"]["mae"],
        ),
        "residual_interval": [float(value) for value in residual_interval],
        "evaluation": {
            "protocol": "expanding-window chronological seasonal holdouts",
            "minimum_train_examples": MINIMUM_TRAIN_SAMPLES,
            "minimum_test_examples_per_season": MINIMUM_SEASON_TEST_SAMPLES,
            "overall": overall_metrics,
            "evaluated_examples": int(len(actuals)),
            "seasonal_holdouts": season_results,
            "skipped_seasons": skipped_seasons,
        },
        "source": MODEL_SOURCE,
        "target_definition": (
            "Hourly mean PM2.5 across available CPCB/KSPCB-attributed Bengaluru "
            "reference stations."
        ),
    }


def train_forecaster(
    weather_path: Path = WEATHER_DATA_PATH,
    ground_station_path: Path = GROUND_STATION_DATA_PATH,
    model_path: Path = MODEL_PATH,
):
    weather, city_pm25, station_count = load_training_data(
        weather_path, ground_station_path
    )
    models = {}
    for horizon in FORECAST_HORIZONS:
        samples = _make_samples(weather, city_pm25, horizon)
        models[horizon] = _fit_evaluate_horizon(samples, horizon)
        models[horizon]["station_count"] = station_count
    bundle = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_source": "CPCB/KSPCB-attributed ground-station observations",
        "weather_source": "Open-Meteo historical weather predictors",
        "supported_horizons": list(FORECAST_HORIZONS),
        "models": models,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def _request_weather():
    params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "hourly": ",".join(WEATHER_API_FIELDS.values()),
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    try:
        response = requests.get(WEATHER_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise ForecastError(
            "Could not retrieve current Bengaluru weather inputs from Open-Meteo."
        ) from error
    except ValueError as error:
        raise ForecastError("Open-Meteo returned an invalid weather response.") from error

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ForecastError("Open-Meteo returned an unexpected hourly weather format.")
    columns = {"observed_at": hourly["time"]}
    for output, api_name in WEATHER_API_FIELDS.items():
        values = hourly.get(api_name)
        if not isinstance(values, list) or len(values) != len(hourly["time"]):
            raise ForecastError(f"Open-Meteo did not return valid {api_name} data.")
        columns[output] = values

    weather = pd.DataFrame(columns)
    direction = np.deg2rad(pd.to_numeric(weather.pop("wind_direction"), errors="coerce"))
    speed = pd.to_numeric(weather["wind_speed"], errors="coerce")
    weather["wind_u"] = -speed * np.sin(direction)
    weather["wind_v"] = -speed * np.cos(direction)
    for column in WEATHER_FEATURE_COLUMNS:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(weather.pop("observed_at"))
    ).tz_localize(TIMEZONE)
    weather.index = timestamps
    return weather


def _load_bundle():
    if not MODEL_PATH.exists():
        raise ForecastError(
            "The ground-station PM2.5 forecasting model is not trained. "
            "Provide CPCB/KSPCB historical observations, then run "
            "`python train_forecast_model.py` from the backend folder."
        )
    try:
        bundle = joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError, AttributeError, ImportError) as error:
        raise ForecastError(
            "The PM2.5 forecasting model is incompatible with this Python environment. "
            "Retrain it using the same Python environment that runs the API."
        ) from error
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema_version") != MODEL_SCHEMA_VERSION
        or bundle.get("target_source")
        != "CPCB/KSPCB-attributed ground-station observations"
    ):
        raise ForecastError(
            "The saved model predates ground-station target validation. "
            "Supply historical station observations and run "
            "`python train_forecast_model.py` from the backend folder."
        )
    return bundle


def get_forecast_status():
    """Report model readiness without turning an expected unavailable state into an API error."""
    try:
        bundle = _load_bundle()
    except ForecastError as error:
        detail = str(error)
        if "predates ground-station target validation" in detail:
            message = (
                "Forecasts are paused until historical CPCB/KSPCB ground-station "
                "PM2.5 observations are added and the model is retrained. The saved "
                "model used modelled pollution targets and is not used as a "
                "ground-station forecast."
            )
        else:
            message = detail
        return {
            "available": False,
            "supported_horizons": list(FORECAST_HORIZONS),
            "message": message,
        }

    return {
        "available": True,
        "supported_horizons": bundle["supported_horizons"],
        "message": "",
    }


def get_pm25_forecast(horizon_hours: int = 3, force_refresh: bool = False):
    global _cached_forecast, _cache_updated_at
    if horizon_hours not in FORECAST_HORIZONS:
        raise ForecastError(
            f"Unsupported forecast horizon. Choose one of: "
            f"{', '.join(map(str, FORECAST_HORIZONS))} hours."
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
        model = model_bundle["models"].get(horizon_hours)
        if model is None:
            raise ForecastError(
                "The ground-station forecasting model does not include that horizon. "
                "Run `python train_forecast_model.py` from the backend folder."
            )
        weather = _request_weather()
        features, target_times = build_feature_frame(weather, horizon_hours)
        now = datetime.now(TIMEZONE).replace(minute=0, second=0, microsecond=0)
        available = features.notna().all(axis=1) & (features.index <= now)
        candidates = features.loc[available]
        if candidates.empty:
            raise ForecastError("Open-Meteo has not provided complete recent weather inputs.")
        issued_at = candidates.index[-1]
        if now - issued_at.to_pydatetime() > timedelta(hours=6):
            raise ForecastError(
                "The latest complete Open-Meteo weather input is over six hours old."
            )

        prediction = max(0.0, float(model["model"].predict(candidates.iloc[[-1]])[0]))
        residual_low, residual_high = model["residual_interval"]
        row_index = features.index.get_loc(issued_at)
        forecast_time = target_times[row_index]
        result = {
            "location": "Bengaluru",
            "issued_at": issued_at.isoformat(),
            "forecast_for": forecast_time.isoformat(),
            "horizon_hours": horizon_hours,
            "pm25_prediction": round(prediction, 1),
            "prediction_interval": {
                "low": round(max(0.0, prediction + float(residual_low)), 1),
                "high": round(max(0.0, prediction + float(residual_high)), 1),
                "coverage": "approximate 90% seasonal-holdout residual interval",
            },
            "model_mae": round(model["model_metrics"]["mae"], 2),
            "persistence_mae": round(model["persistence_metrics"]["mae"], 2),
            "seasonal_mae": round(model["seasonal_metrics"]["mae"], 2),
            "ridge_mae": round(model["ridge_metrics"]["mae"], 2),
            "beats_baselines": model["beats_baselines"],
            "training_examples": model["sample_count"],
            "station_count": model["station_count"],
            "trained_through": model["trained_through"],
            "seasonal_evaluation": model["evaluation"],
            "target_source": model_bundle["target_source"],
            "target_definition": model["target_definition"],
            "source": model["source"],
            "data_note": (
                "Weather-driven city-level estimate trained against historical "
                "CPCB/KSPCB-attributed station PM2.5. It is not a live monitor "
                "reading, official AQI, or neighborhood-specific forecast."
            ),
        }
        if _cached_forecast is None:
            _cached_forecast = {}
        _cached_forecast[horizon_hours] = result
        _cache_updated_at = monotonic()
        return result
