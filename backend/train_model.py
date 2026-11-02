"""Train and save the pollution clustering model."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "historical_bengaluru_aqi.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FEATURES = ["pm25", "pm10", "no2", "wind_speed"]

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate or load sample Greater Bengaluru dataset (covering core + outer areas)
def create_sample_dataset():
    np.random.seed(42)
    locations = [
        "Silk Board", "Peenya Industrial", "Hebbal", "Hennur", 
        "Hoskote", "Nelamangala", "Devanahalli", "Indiranagar", 
        "Electronic City", "Whitefield", "Doddaballapur"
    ]
    
    data = []
    for loc in locations:
        # Simulate varying pollution profiles (e.g. Industrial/Highway vs Residential)
        is_industrial = loc in ["Peenya Industrial", "Electronic City", "Hoskote", "Nelamangala"]
        base_pm25 = np.random.uniform(90, 160) if is_industrial else np.random.uniform(35, 85)
        base_pm10 = base_pm25 * np.random.uniform(1.4, 1.8)
        base_no2 = np.random.uniform(40, 90) if is_industrial else np.random.uniform(15, 45)
        wind_speed = np.random.uniform(1.2, 4.5)
        
        data.append({
            "location_name": loc,
            "pm25": base_pm25,
            "pm10": base_pm10,
            "no2": base_no2,
            "wind_speed": wind_speed
        })
    
    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)
    return df

if not DATA_PATH.exists():
    df = create_sample_dataset()
else:
    try:
        df = pd.read_csv(DATA_PATH)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    if df.empty:
        df = create_sample_dataset()
    else:
        missing_features = [feature for feature in FEATURES if feature not in df.columns]
        if missing_features:
            raise ValueError(
                f"Training data at {DATA_PATH} is missing required columns: "
                f"{', '.join(missing_features)}. Expected: {', '.join(FEATURES)}."
            )

# 2. Preprocess & Scale features for Clustering
X = df[FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit K-Means Model (e.g., 3 clusters: Traffic/Industrial, Moderate Urban, Clean/Residential)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 4. Save model and scaler locally in VS Code workspace
joblib.dump(kmeans, ARTIFACTS_DIR / "kmeans_model.joblib")
joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

print(f"Model and scaler successfully trained and saved in {ARTIFACTS_DIR}!")