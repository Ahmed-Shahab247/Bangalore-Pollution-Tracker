"""Train an offline clustering model from attributed historical measurements."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "historical_bengaluru_aqi.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REQUIRED_COLUMNS = ["location_name", "observed_at", "source", "pm25", "pm10", "no2"]
BASE_FEATURES = ["pm25", "pm10", "no2"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Historical measurement data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Training data is missing required columns: {', '.join(missing_columns)}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}."
        )

    official_source = df["source"].fillna("").str.contains(
        r"\b(?:cpcb|central pollution control board|kspcb|karnataka state pollution control board)\b",
        case=False,
        regex=True,
    )
    official_data = df.loc[official_source].copy()
    if official_data.empty:
        raise ValueError(
            "No CPCB/KSPCB-attributed readings found; synthetic or unattributed "
            "sample data cannot be used to train this model."
        )

    official_data["location_name"] = (
        official_data["location_name"].astype("string").str.strip().replace("", pd.NA)
    )
    official_data["observed_at"] = pd.to_datetime(
        official_data["observed_at"], errors="coerce", utc=True
    )
    features = BASE_FEATURES.copy()
    if "wind_speed" in official_data.columns and official_data["wind_speed"].notna().all():
        features.append("wind_speed")

    for feature in features:
        official_data[feature] = pd.to_numeric(official_data[feature], errors="coerce")
    training_data = official_data.dropna(
        subset=["location_name", "observed_at", *features]
    )
    if len(training_data) < 3:
        raise ValueError(
            "At least three complete, CPCB/KSPCB-attributed historical readings "
            "are required for clustering."
        )

    values = training_data[features]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)
    model = KMeans(
        n_clusters=min(3, len(training_data)),
        random_state=42,
        n_init=10,
    )
    with threadpool_limits(limits=1):
        model.fit(scaled_values)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(
        f"Trained on {len(training_data)} CPCB/KSPCB-attributed readings "
        f"using {', '.join(features)}. Saved artifacts to {ARTIFACTS_DIR}."
    )


if __name__ == "__main__":
    main()
