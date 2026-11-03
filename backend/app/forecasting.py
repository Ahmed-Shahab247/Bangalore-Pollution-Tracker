"""Ground-station PM2.5 forecasting with seasonal chronological evaluation."""

from datetime import datetime, timedelta, timezone
import math
import os
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
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_BENGALURU_BOUNDS = "77.25,12.75,78.00,13.35"
FORECAST_HORIZONS = (3, 12)
MINIMUM_TRAIN_SAMPLES = 500
MINIMUM_SEASON_TEST_SAMPLES = 24
MINIMUM_TOTAL_SAMPLES = 1000
MODEL_SCHEMA_VERSION = 4
MIN_STATIONS_PER_HOUR = 3
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
    "issue_hour_sin",
    "issue_hour_cos",
)
POLLUTION_LAG_HOURS = (1, 3, 6, 24)
ROLLING_WINDOWS = (3, 6)
POLLUTION_LAG_FEATURE_COLUMNS = tuple(
    f"{pollutant}_lag_{hours}h"
    for pollutant in ("pm25", "pm10")
    for hours in POLLUTION_LAG_HOURS
)
POLLUTION_FEATURE_COLUMNS = POLLUTION_LAG_FEATURE_COLUMNS + tuple(
    f"{pollutant}_ma_{hours}h"
    for pollutant in ("pm25", "pm10", "wind_speed")
    for hours in ROLLING_WINDOWS
) + (
    "pm25_trend_1h_3h",
    "pm25_max_24h",
    "pm25_std_24h",
    "pm10_max_24h",
    "pm10_std_24h",
    "pm25_pm10_ratio",
)
TARGET_WEATHER_COLUMNS = (
    "wind_speed_target",
    "precipitation_target",
    "boundary_layer_height_target",
    "wind_speed_delta",
    "precipitation_delta",
    "boundary_layer_height_delta",
)
FEATURE_COLUMNS = (
    WEATHER_FEATURE_COLUMNS
    + POLLUTION_FEATURE_COLUMNS
    + TARGET_WEATHER_COLUMNS
    + TIME_FEATURE_COLUMNS
)
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
_cached_pollution_history = None
_pollution_history_updated_at = 0.0
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
    required_ground = {"location_name", "observed_at", "source", "pm25", "pm10"}
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
    ground["pm10"] = pd.to_numeric(ground["pm10"], errors="coerce")
    valid = (
        ground["observed_at"].notna()
        & ground["location_name"].notna()
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
            "No valid, fresh CPCB/KSPCB observations remain after validation."
        )

    ground["observed_at"] = (
        ground["observed_at"].dt.tz_convert(TIMEZONE).dt.floor("h")
    )
    valid_pm25 = (
        ground["pm25"].notna()
        & np.isfinite(ground["pm25"])
        & (ground["pm25"] >= 0)
        & (ground["pm25"] <= 2000)
    )
    valid_pm10 = (
        ground["pm10"].notna()
        & np.isfinite(ground["pm10"])
        & (ground["pm10"] >= 0)
        & (ground["pm10"] <= 2000)
    )
    ground.loc[~valid_pm25, "pm25"] = np.nan
    ground.loc[~valid_pm10, "pm10"] = np.nan
    ground = ground.dropna(subset=["pm25", "pm10"], how="all")
    pm25_station_hourly = (
        ground.dropna(subset=["pm25"])
        .groupby(["location_name", "observed_at"], observed=True)["pm25"]
        .mean()
    )
    if pm25_station_hourly.empty:
        raise ValueError(
            "No valid, fresh CPCB/KSPCB PM2.5 target observations remain after validation."
        )
    pm10_station_hourly = (
        ground.dropna(subset=["pm10"])
        .groupby(["location_name", "observed_at"], observed=True)["pm10"]
        .mean()
    )
    pm25_station_counts = pm25_station_hourly.groupby(level="observed_at").size()
    city_hourly = pd.DataFrame(
        {
            "pm25": pm25_station_hourly.groupby(level="observed_at").mean(),
            "pm10": pm10_station_hourly.groupby(level="observed_at").mean(),
            "station_count": pm25_station_counts,
        }
    ).sort_index().asfreq("h")
    city_hourly.loc[
        city_hourly["station_count"] < MIN_STATIONS_PER_HOUR, "pm25"
    ] = np.nan
    city_hourly.loc[
        city_hourly["station_count"] < MIN_STATIONS_PER_HOUR, "pm10"
    ] = np.nan
    station_names = sorted(
        pm25_station_hourly.index.get_level_values("location_name").unique().tolist()
    )
    return weather_frame, city_hourly, station_names


def build_feature_frame(
    weather: pd.DataFrame,
    horizon_hours: int,
    pollution: pd.DataFrame,
):
    """Create forecast features using only weather and pollution known at issue time."""
    if horizon_hours not in FORECAST_HORIZONS:
        raise ValueError(f"Unsupported forecast horizon: {horizon_hours}.")
    frame = _hourly_frame(weather.copy(), "weather")
    missing = set(WEATHER_FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            f"Weather input is missing feature columns: {', '.join(sorted(missing))}."
        )

    features = frame.loc[:, WEATHER_FEATURE_COLUMNS].copy()
    pollution_frame = _hourly_frame(pollution.copy(), "ground-station pollution")
    missing_pollution = {"pm25", "pm10"}.difference(pollution_frame.columns)
    if missing_pollution:
        raise ValueError(
            "Ground-station history is missing pollutant columns: "
            f"{', '.join(sorted(missing_pollution))}."
        )
    past_pollution = pollution_frame.reindex(features.index).ffill(limit=3)
    for pollutant in ("pm25", "pm10"):
        for lag_hours in POLLUTION_LAG_HOURS:
            features[f"{pollutant}_lag_{lag_hours}h"] = past_pollution[
                pollutant
            ].shift(lag_hours)
        prior_readings = past_pollution[pollutant].shift(1)
        for window in ROLLING_WINDOWS:
            features[f"{pollutant}_ma_{window}h"] = prior_readings.rolling(
                window=window, min_periods=window
            ).mean()
        features[f"{pollutant}_max_24h"] = prior_readings.rolling(
            window=24, min_periods=24
        ).max()
        features[f"{pollutant}_std_24h"] = prior_readings.rolling(
            window=24, min_periods=24
        ).std()
    for window in ROLLING_WINDOWS:
        features[f"wind_speed_ma_{window}h"] = frame["wind_speed"].shift(1).rolling(
            window=window, min_periods=window
        ).mean()

    target_index = features.index + pd.Timedelta(hours=horizon_hours)
    issue_hour = features.index.hour.to_numpy() * (2 * math.pi / 24)
    features["issue_hour_sin"] = np.sin(issue_hour)
    features["issue_hour_cos"] = np.cos(issue_hour)
    pm25_lag_1 = features["pm25_lag_1h"]
    pm25_lag_3 = features["pm25_lag_3h"]
    features["pm25_trend_1h_3h"] = pm25_lag_1 - pm25_lag_3
    pm25_lagged = past_pollution["pm25"].shift(1)
    pm10_lagged = past_pollution["pm10"].shift(1)
    features["pm25_pm10_ratio"] = pm25_lagged.div(
        pm10_lagged.where(pm10_lagged > 0)
    )

    target_weather = frame.reindex(pd.DatetimeIndex(target_index))
    for column in ("wind_speed", "precipitation", "boundary_layer_height"):
        target_column = f"{column}_target"
        features[target_column] = target_weather[column].to_numpy()
        features[f"{column}_delta"] = (
            target_weather[column].to_numpy() - features[column].to_numpy()
        )

    features["seasonal_24h_pm25"] = pollution_frame["pm25"].reindex(
        pd.DatetimeIndex(target_index) - pd.Timedelta(hours=24)
    ).to_numpy()
    return (
        features.loc[:, [*FEATURE_COLUMNS, "seasonal_24h_pm25"]],
        pd.DatetimeIndex(target_index),
    )


def _metrics(actual, predicted):
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def _make_samples(
    weather: pd.DataFrame, city_pollution: pd.DataFrame, horizon_hours: int
):
    features, target_times = build_feature_frame(
        weather, horizon_hours, city_pollution
    )
    samples = features.copy()
    samples["target_pm25"] = city_pollution["pm25"].reindex(target_times).to_numpy()
    samples["target_time"] = target_times
    samples["persistence_pm25"] = features["pm25_lag_1h"].to_numpy()
    samples["seasonal_pm25"] = features["seasonal_24h_pm25"].to_numpy()
    samples = samples.replace([np.inf, -np.inf], np.nan)
    return samples.dropna(subset=["target_pm25", "persistence_pm25"]).loc[
        lambda frame: frame["target_pm25"] >= 0
    ]


def _month_windows(start: pd.Timestamp, end: pd.Timestamp):
    """Yield chronological monthly holdout windows and their broad regimes."""
    cursor = pd.Timestamp(year=start.year, month=start.month, day=1, tz=TIMEZONE)
    while cursor <= end:
        window_end = cursor + pd.offsets.MonthBegin(1)
        if cursor.month in (6, 7, 8, 9):
            regime = "monsoon"
        elif cursor.month in (11, 12, 1, 2):
            regime = "winter_post_monsoon"
        else:
            regime = "interseason"
        yield cursor.strftime("%Y-%m"), regime, cursor, window_end
        cursor = window_end


def _new_estimators():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    return {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=160,
            max_depth=4,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        ),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
    }


def _fit_estimators(samples: pd.DataFrame):
    estimators = _new_estimators()
    correction = samples["target_pm25"] - samples["persistence_pm25"]
    for estimator in estimators.values():
        estimator.fit(samples.loc[:, FEATURE_COLUMNS], correction)
    return estimators


def _predict_estimators(estimators, samples: pd.DataFrame):
    baseline = samples["persistence_pm25"].to_numpy(dtype=float)
    predictions = {
        name: np.maximum(
            0,
            baseline + estimator.predict(samples.loc[:, FEATURE_COLUMNS]),
        )
        for name, estimator in estimators.items()
    }
    predictions["blend"] = (
        predictions["hist_gradient_boosting"] + predictions["ridge"]
    ) / 2
    predictions["persistence"] = baseline
    predictions["seasonal_24h"] = samples["seasonal_pm25"].to_numpy(dtype=float)
    return predictions


def _block_bootstrap_difference(folds, repetitions=1000, block_hours=24):
    """Estimate a daily-block-bootstrap CI for selected-model MAE minus persistence."""
    rng = np.random.default_rng(42)
    fold_blocks = []
    for fold in folds:
        data = pd.DataFrame(
            {
                "target_time": fold["target_time"],
                "actual": fold["actual"],
                "selected": fold["selected"],
                "persistence": fold["persistence"],
            }
        )
        blocks = [
            group[["actual", "selected", "persistence"]].to_numpy(dtype=float)
            for _, group in data.groupby(
                data["target_time"].dt.floor(f"{block_hours}h")
            )
        ]
        if blocks:
            fold_blocks.append((blocks, len(data)))

    if not fold_blocks:
        raise ValueError("No chronological holdout blocks are available for bootstrap.")

    differences = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        actual_parts = []
        selected_parts = []
        persistence_parts = []
        for blocks, sample_size in fold_blocks:
            sampled_parts = []
            while sum(len(part) for part in sampled_parts) < sample_size:
                sampled_parts.append(blocks[rng.integers(0, len(blocks))])
            sample = np.concatenate(sampled_parts, axis=0)[:sample_size]
            actual_parts.append(sample[:, 0])
            selected_parts.append(sample[:, 1])
            persistence_parts.append(sample[:, 2])
        actual = np.concatenate(actual_parts)
        selected = np.concatenate(selected_parts)
        persistence = np.concatenate(persistence_parts)
        differences[iteration] = (
            np.mean(np.abs(actual - selected))
            - np.mean(np.abs(actual - persistence))
        )

    return {
        "metric": "selected_model_mae_minus_persistence_mae",
        "block_hours": block_hours,
        "repetitions": repetitions,
        "confidence_level": 0.95,
        "low": float(np.quantile(differences, 0.025)),
        "high": float(np.quantile(differences, 0.975)),
    }


def _fit_evaluate_horizon(samples: pd.DataFrame, horizon_hours: int):
    if len(samples) < MINIMUM_TOTAL_SAMPLES:
        raise ValueError(
            f"Only {len(samples)} complete ground-station forecast examples are "
            f"available for {horizon_hours} hours; at least "
            f"{MINIMUM_TOTAL_SAMPLES} are required."
        )

    model_names = (
        "hist_gradient_boosting",
        "ridge",
        "blend",
        "persistence",
        "seasonal_24h",
    )
    predictions = {name: [] for name in model_names}
    actuals = []
    monthly_results = {}
    skipped_months = {}
    evaluation_folds = []

    for key, regime, window_start, window_end in _month_windows(
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
            skipped_months[key] = (
                f"Only {len(train)} earlier training examples; "
                f"{MINIMUM_TRAIN_SAMPLES} required."
            )
            continue
        test = test.dropna(subset=["seasonal_pm25"])
        if len(test) < MINIMUM_SEASON_TEST_SAMPLES:
            skipped_months[key] = (
                f"Only {len(test)} comparable test examples; "
                f"{MINIMUM_SEASON_TEST_SAMPLES} required."
            )
            continue
        if train["target_time"].max() >= test["target_time"].min():
            raise ValueError(f"Temporal leakage detected in holdout {key}.")

        fold_estimators = _fit_estimators(train)
        model_predictions = _predict_estimators(fold_estimators, test)
        actual = test["target_pm25"].to_numpy()
        actuals.extend(actual)
        for name in model_names:
            predictions[name].extend(model_predictions[name])
        monthly_results[key] = {
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
        evaluation_folds.append(
            {
                "target_time": test["target_time"].reset_index(drop=True),
                "actual": actual,
                "selected_predictions": model_predictions,
            }
        )

    evaluated_regimes = {result["regime"] for result in monthly_results.values()}
    if len(monthly_results) < 2:
        raise ValueError(
            "Insufficient chronological ground-station coverage for at least two "
            f"monthly holdouts. Skipped months: {skipped_months or 'none'}."
        )

    overall_metrics = {
        name: _metrics(actuals, predicted)
        for name, predicted in predictions.items()
    }
    candidate_mae = {
        name: overall_metrics[name]["mae"]
        for name in ("hist_gradient_boosting", "ridge", "blend")
    }
    best_candidate = min(candidate_mae, key=candidate_mae.get)
    best_baseline = min(
        ("persistence", "seasonal_24h"),
        key=lambda name: overall_metrics[name]["mae"],
    )
    selected_model = best_candidate
    if candidate_mae[best_candidate] >= overall_metrics[best_baseline]["mae"]:
        selected_model = best_baseline

    estimator_bundle = _fit_estimators(samples)
    residuals = (
        np.asarray(actuals) - np.asarray(predictions[selected_model])
    )
    residual_interval = np.quantile(residuals, [0.05, 0.95])
    for fold in evaluation_folds:
        fold["selected"] = fold["selected_predictions"][selected_model]
        fold["persistence"] = fold["selected_predictions"]["persistence"]
    skill_score = 1 - (
        overall_metrics[selected_model]["mae"]
        / overall_metrics["persistence"]["mae"]
    )
    return {
        "model": estimator_bundle.get(selected_model),
        "comparison_model": estimator_bundle["ridge"],
        "estimators": estimator_bundle,
        "feature_columns": list(FEATURE_COLUMNS),
        "horizon_hours": horizon_hours,
        "sample_count": int(len(samples)),
        "trained_through": samples["target_time"].max().isoformat(),
        "selected_model": selected_model,
        "model_metrics": overall_metrics[selected_model],
        "ridge_metrics": overall_metrics["ridge"],
        "hist_gradient_boosting_metrics": overall_metrics[
            "hist_gradient_boosting"
        ],
        "blend_metrics": overall_metrics["blend"],
        "persistence_metrics": overall_metrics["persistence"],
        "seasonal_metrics": overall_metrics["seasonal_24h"],
        "beats_baselines": selected_model in candidate_mae
        and candidate_mae[selected_model]
        < min(
            overall_metrics["persistence"]["mae"],
            overall_metrics["seasonal_24h"]["mae"],
        ),
        "skill_score_vs_persistence": float(skill_score),
        "residual_interval": [float(value) for value in residual_interval],
        "evaluation": {
            "protocol": "expanding-window chronological monthly holdouts",
            "minimum_train_examples": MINIMUM_TRAIN_SAMPLES,
            "minimum_test_examples_per_month": MINIMUM_SEASON_TEST_SAMPLES,
            "overall": overall_metrics,
            "evaluated_examples": int(len(actuals)),
            "monthly_holdouts": monthly_results,
            "evaluated_regimes": sorted(evaluated_regimes),
            "skipped_months": skipped_months,
            "selected_model": selected_model,
            "persistence_mae_difference_ci": _block_bootstrap_difference(
                evaluation_folds
            ),
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
    weather, city_pollution, station_names = load_training_data(
        weather_path, ground_station_path
    )
    models = {}
    for horizon in FORECAST_HORIZONS:
        samples = _make_samples(weather, city_pollution, horizon)
        models[horizon] = _fit_evaluate_horizon(samples, horizon)
        models[horizon]["station_count"] = len(station_names)
        models[horizon]["station_names"] = station_names
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


def _request_pollution_history(station_names):
    global _cached_pollution_history, _pollution_history_updated_at
    if (
        _cached_pollution_history is not None
        and monotonic() - _pollution_history_updated_at < FORECAST_CACHE_SECONDS
    ):
        return _cached_pollution_history.copy()

    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise ForecastError(
            "Ground-station lag inputs are not configured. Set OPENAQ_API_KEY "
            "on the backend and restart it."
        )

    headers = {"X-API-Key": api_key}
    try:
        response = requests.get(
            f"{OPENAQ_BASE_URL}/locations",
            headers=headers,
            params={
                "bbox": OPENAQ_BENGALURU_BOUNDS,
                "iso": "IN",
                "monitor": "true",
                "limit": 1000,
                "page": 1,
            },
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise ForecastError(
            "Could not retrieve Bengaluru ground-station locations from OpenAQ."
        ) from error
    locations = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(locations, list):
        raise ForecastError("OpenAQ returned an unexpected locations response.")

    def normalized_name(value):
        return " ".join(str(value).casefold().split())

    wanted_names = {normalized_name(name) for name in station_names}
    selected_locations = []
    for location in locations:
        if location.get("isMonitor") is not True or location.get("isMobile") is not False:
            continue
        name = str(location.get("name") or "").strip()
        owner = location.get("owner") or {}
        provider = location.get("provider") or {}
        attribution_names = [
            item.get("name", "")
            for item in (owner, provider)
            if isinstance(item, dict)
        ]
        attribution_names.extend(
            (license_info.get("attribution") or {}).get("name", "")
            for license_info in location.get("licenses") or []
            if isinstance(license_info, dict)
        )
        attribution = " ".join(str(item) for item in attribution_names).casefold()
        official = any(
            source in attribution
            for source in (
                "cpcb",
                "central pollution control board",
                "kspcb",
                "karnataka state pollution control board",
            )
        )
        if (
            official
            and normalized_name(name) in wanted_names
            and location.get("id") is not None
        ):
            selected_locations.append(location)
    if not selected_locations:
        raise ForecastError(
            "OpenAQ did not return the CPCB/KSPCB training stations needed for "
            "the model's PM2.5/PM10 lag features."
        )

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=48)
    records = []
    for location in selected_locations:
        name = str(location.get("name") or "").strip()
        sensors = {}
        for sensor in location.get("sensors") or []:
            parameter = sensor.get("parameter") or {}
            parameter_name = (
                str(parameter.get("name") or "").casefold().replace(".", "")
            )
            pollutant = {"pm25": "pm25", "pm10": "pm10"}.get(parameter_name)
            unit = str(parameter.get("units") or "").casefold().replace("μ", "µ")
            unit = unit.replace("µg", "ug").replace("³", "3").replace(" ", "")
            if (
                pollutant
                and unit in {"ug/m3", "ugm-3", "microgram/m3", "micrograms/m3"}
                and sensor.get("id") is not None
            ):
                sensors[sensor["id"]] = pollutant

        for sensor_id, pollutant in sensors.items():
            try:
                response = requests.get(
                    f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements",
                    headers=headers,
                    params={
                        "datetime_from": start.isoformat().replace("+00:00", "Z"),
                        "datetime_to": end.isoformat().replace("+00:00", "Z"),
                        "limit": 1000,
                        "page": 1,
                    },
                    timeout=12,
                )
                response.raise_for_status()
                sensor_payload = response.json()
            except (requests.RequestException, ValueError) as error:
                raise ForecastError(
                    "Could not retrieve recent ground-station pollution history "
                    "from OpenAQ."
                ) from error
            measurements = (
                sensor_payload.get("results")
                if isinstance(sensor_payload, dict)
                else None
            )
            if not isinstance(measurements, list):
                raise ForecastError(
                    "OpenAQ returned an unexpected sensor-measurements response."
                )
            for measurement in measurements:
                period = measurement.get("period") or {}
                timestamp_value = (period.get("datetimeTo") or {}).get("utc")
                timestamp = pd.to_datetime(timestamp_value, utc=True, errors="coerce")
                value = measurement.get("value")
                interval = period.get("interval")
                try:
                    if isinstance(interval, str):
                        interval = pd.Timedelta(interval)
                    else:
                        period_start = pd.to_datetime(
                            (period.get("datetimeFrom") or {}).get("utc"),
                            utc=True,
                            errors="coerce",
                        )
                        interval = (
                            timestamp - period_start
                            if pd.notna(period_start) and pd.notna(timestamp)
                            else None
                        )
                except ValueError:
                    interval = None
                if (
                    pd.isna(timestamp)
                    or timestamp < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
                    or timestamp > pd.Timestamp.now(tz="UTC")
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or not 0 <= value <= 2000
                    or (interval is not None and interval > pd.Timedelta(hours=3))
                ):
                    continue
                records.append(
                    {
                        "location_name": name,
                        "observed_at": timestamp,
                        pollutant: float(value),
                    }
                )

    if not records:
        raise ForecastError(
            "No valid CPCB/KSPCB PM2.5/PM10 observations are available from "
            "OpenAQ for the model's lag inputs."
        )
    observations = pd.DataFrame(records)
    observations["observed_at"] = (
        pd.to_datetime(observations["observed_at"], utc=True)
        .dt.tz_convert(TIMEZONE)
        .dt.floor("h")
    )
    station_hourly = observations.groupby(
        ["location_name", "observed_at"], observed=True
    )[["pm25", "pm10"]].mean()
    city_hourly = station_hourly.groupby(level="observed_at").mean().sort_index()
    city_hourly = city_hourly.asfreq("h")
    _cached_pollution_history = city_hourly
    _pollution_history_updated_at = monotonic()
    return city_hourly.copy()


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
            "The saved model does not include the current ground-station lag "
            "features. Run `python train_forecast_model.py` from the backend folder."
        )
    return bundle


def get_forecast_status():
    """Report model readiness without turning an expected unavailable state into an API error."""
    try:
        bundle = _load_bundle()
    except ForecastError as error:
        detail = str(error)
        if "does not include the current ground-station lag features" in detail:
            message = (
                "Forecasts are paused until the lag-feature model is retrained "
                "with historical CPCB/KSPCB ground-station observations."
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
        pollution = _request_pollution_history(model["station_names"])
        feature_frame, target_times = build_feature_frame(
            weather, horizon_hours, pollution
        )
        now = datetime.now(TIMEZONE).replace(minute=0, second=0, microsecond=0)
        required_baseline = (
            "seasonal_24h_pm25"
            if model["selected_model"] == "seasonal_24h"
            else "pm25_lag_1h"
        )
        available = (
            feature_frame["pm25_lag_1h"].notna()
            & feature_frame[required_baseline].notna()
            & (feature_frame.index <= now)
        )
        candidates = feature_frame.loc[available]
        if candidates.empty:
            raise ForecastError(
                "Open-Meteo and CPCB/KSPCB have not provided the recent inputs "
                "required by the selected forecast model."
            )
        issued_at = candidates.index[-1]
        if now - issued_at.to_pydatetime() > timedelta(hours=6):
            raise ForecastError(
                "The latest complete Open-Meteo weather input is over six hours old."
            )

        row = candidates.iloc[[-1]]
        selected_model = model["selected_model"]
        if selected_model == "persistence":
            prediction = float(row["pm25_lag_1h"].iloc[0])
        elif selected_model == "seasonal_24h":
            prediction = float(row["seasonal_24h_pm25"].iloc[0])
        elif selected_model == "blend":
            estimators = model["estimators"]
            corrections = [
                float(estimators[name].predict(row.loc[:, FEATURE_COLUMNS])[0])
                for name in ("hist_gradient_boosting", "ridge")
            ]
            prediction = float(row["pm25_lag_1h"].iloc[0]) + sum(corrections) / 2
        else:
            correction = float(
                model["estimators"][selected_model].predict(
                    row.loc[:, FEATURE_COLUMNS]
                )[0]
            )
            prediction = float(row["pm25_lag_1h"].iloc[0]) + correction
        prediction = max(0.0, prediction)
        residual_low, residual_high = model["residual_interval"]
        row_index = feature_frame.index.get_loc(issued_at)
        forecast_time = target_times[row_index]
        result = {
            "location": "Bengaluru",
            "issued_at": issued_at.isoformat(),
            "forecast_for": forecast_time.isoformat(),
            "horizon_hours": horizon_hours,
            "pm25_prediction": round(prediction, 1),
            "selected_model": selected_model,
            "prediction_interval": {
                "low": round(max(0.0, prediction + float(residual_low)), 1),
                "high": round(max(0.0, prediction + float(residual_high)), 1),
                "coverage": "approximate 90% monthly-holdout residual interval",
            },
            "model_mae": round(model["model_metrics"]["mae"], 2),
            "persistence_mae": round(model["persistence_metrics"]["mae"], 2),
            "seasonal_mae": round(model["seasonal_metrics"]["mae"], 2),
            "ridge_mae": round(model["ridge_metrics"]["mae"], 2),
            "hist_gradient_boosting_mae": round(
                model["hist_gradient_boosting_metrics"]["mae"], 2
            ),
            "blend_mae": round(model["blend_metrics"]["mae"], 2),
            "beats_baselines": model["beats_baselines"],
            "skill_score_vs_persistence": round(
                model["skill_score_vs_persistence"], 4
            ),
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
