from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from math import asin, cos, radians, sin, sqrt
from threading import Lock
from time import monotonic, sleep
import joblib
import numpy as np
import pandas as pd
import os
import requests

app = FastAPI(title="Bengaluru Hyper-Local Pollution & Clustering API")
_geocoding_lock = Lock()
_last_geocoding_request = 0.0

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load local ML artifacts at startup
MODEL_PATH = "artifacts/kmeans_model.joblib"
SCALER_PATH = "artifacts/scaler.joblib"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None

class PollutionPayload(BaseModel):
    location_name: str
    pm25: float
    pm10: float
    no2: float
    wind_speed: float

# Cluster label mapping helper
def get_cluster_label(cluster_id: int):
    mapping = {
        0: "Moderate Urban Zone",
        1: "Heavy Industrial & Freight Corridor",
        2: "Balanced / Residential Zone"
    }
    return mapping.get(cluster_id, "Unknown Zone Profile")

@app.get("/")
def read_root():
    return {"status": "Active", "region": "Greater Bengaluru Air Quality Engine"}

@app.post("/api/predict-cluster")
def predict_cluster(payload: PollutionPayload):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="ML Model artifacts not found. Run train_model.py first.")
    
    input_data = np.array([[payload.pm25, payload.pm10, payload.no2, payload.wind_speed]])
    input_scaled = scaler.transform(input_data)
    cluster_id = int(model.predict(input_scaled)[0])
    
    # Calculate simple CPCB AQI approximation based on PM2.5 for instant color-coding
    pm = payload.pm25
    if pm <= 30: aqi, category, color = 50, "Good", "green"
    elif pm <= 60: aqi, category, color = 100, "Satisfactory", "lightgreen"
    elif pm <= 90: aqi, category, color = 200, "Moderate", "orange"
    elif pm <= 120: aqi, category, color = 300, "Poor", "red"
    elif pm <= 250: aqi, category, color = 400, "Very Poor", "purple"
    else: aqi, category, color = 500, "Severe", "maroon"

    return {
        "location": payload.location_name,
        "aqi": aqi,
        "category": category,
        "color_code": color,
        "cluster_id": cluster_id,
        "cluster_profile": get_cluster_label(cluster_id)
    }

@app.get("/api/regions")
def get_all_regions():
    """Return representative Bengaluru areas with illustrative demo readings."""
    return [
        {"name": "Arekere", "lat": 12.8877, "lon": 77.5970, "pm25": 66, "pm10": 108, "no2": 34, "wind_speed": 2.2},
        {"name": "Bagalur", "lat": 13.1340, "lon": 77.6680, "pm25": 72, "pm10": 119, "no2": 36, "wind_speed": 2.8},
        {"name": "Banashankari", "lat": 12.9255, "lon": 77.5642, "pm25": 62, "pm10": 101, "no2": 31, "wind_speed": 2.4},
        {"name": "Basaveshwaranagar", "lat": 12.9905, "lon": 77.5380, "pm25": 59, "pm10": 96, "no2": 29, "wind_speed": 2.5},
        {"name": "BEL Road", "lat": 13.0340, "lon": 77.5700, "pm25": 57, "pm10": 93, "no2": 27, "wind_speed": 2.7},
        {"name": "Begur", "lat": 12.8788, "lon": 77.6248, "pm25": 75, "pm10": 124, "no2": 39, "wind_speed": 2.0},
        {"name": "Bellandur", "lat": 12.9304, "lon": 77.6784, "pm25": 88, "pm10": 145, "no2": 48, "wind_speed": 2.1},
        {"name": "Bommanahalli", "lat": 12.9089, "lon": 77.6243, "pm25": 82, "pm10": 134, "no2": 44, "wind_speed": 1.9},
        {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101, "pm25": 76, "pm10": 125, "no2": 41, "wind_speed": 2.0},
        {"name": "CV Raman Nagar", "lat": 12.9850, "lon": 77.6630, "pm25": 68, "pm10": 111, "no2": 35, "wind_speed": 2.3},
        {"name": "Devanahalli", "lat": 13.2477, "lon": 77.7130, "pm25": 48, "pm10": 79, "no2": 22, "wind_speed": 3.2},
        {"name": "Domlur", "lat": 12.9609, "lon": 77.6387, "pm25": 72, "pm10": 118, "no2": 39, "wind_speed": 2.1},
        {"name": "Doddaballapur", "lat": 13.2927, "lon": 77.5389, "pm25": 51, "pm10": 83, "no2": 24, "wind_speed": 3.0},
        {"name": "Electronic City", "lat": 12.8399, "lon": 77.6770, "pm25": 105, "pm10": 172, "no2": 59, "wind_speed": 2.4},
        {"name": "Ejipura", "lat": 12.9454, "lon": 77.6266, "pm25": 80, "pm10": 131, "no2": 45, "wind_speed": 1.8},
        {"name": "Hebbal", "lat": 13.0358, "lon": 77.5970, "pm25": 73, "pm10": 120, "no2": 39, "wind_speed": 2.6},
        {"name": "Hennur", "lat": 13.0358, "lon": 77.6395, "pm25": 58, "pm10": 95, "no2": 28, "wind_speed": 2.1},
        {"name": "Hoskote", "lat": 13.0731, "lon": 77.7997, "pm25": 115, "pm10": 180, "no2": 65, "wind_speed": 3.4},
        {"name": "HSR Layout", "lat": 12.9116, "lon": 77.6389, "pm25": 70, "pm10": 115, "no2": 37, "wind_speed": 2.2},
        {"name": "Indiranagar", "lat": 12.9784, "lon": 77.6408, "pm25": 65, "pm10": 105, "no2": 32, "wind_speed": 2.5},
        {"name": "Jakkur", "lat": 13.0785, "lon": 77.6060, "pm25": 55, "pm10": 90, "no2": 26, "wind_speed": 2.9},
        {"name": "Jalahalli", "lat": 13.0500, "lon": 77.5460, "pm25": 63, "pm10": 103, "no2": 30, "wind_speed": 2.7},
        {"name": "Jayanagar", "lat": 12.9250, "lon": 77.5838, "pm25": 54, "pm10": 88, "no2": 25, "wind_speed": 2.4},
        {"name": "JP Nagar", "lat": 12.9063, "lon": 77.5857, "pm25": 61, "pm10": 100, "no2": 30, "wind_speed": 2.3},
        {"name": "Kengeri", "lat": 12.9177, "lon": 77.4838, "pm25": 67, "pm10": 110, "no2": 34, "wind_speed": 2.8},
        {"name": "Koramangala", "lat": 12.9352, "lon": 77.6245, "pm25": 78, "pm10": 128, "no2": 43, "wind_speed": 1.9},
        {"name": "KR Puram", "lat": 13.0075, "lon": 77.6959, "pm25": 91, "pm10": 150, "no2": 51, "wind_speed": 2.2},
        {"name": "Mahadevapura", "lat": 12.9918, "lon": 77.6878, "pm25": 86, "pm10": 141, "no2": 47, "wind_speed": 2.1},
        {"name": "Malleshwaram", "lat": 13.0031, "lon": 77.5643, "pm25": 56, "pm10": 92, "no2": 27, "wind_speed": 2.5},
        {"name": "Marathahalli", "lat": 12.9591, "lon": 77.6974, "pm25": 93, "pm10": 153, "no2": 53, "wind_speed": 2.0},
        {"name": "MG Road", "lat": 12.9757, "lon": 77.6068, "pm25": 74, "pm10": 121, "no2": 42, "wind_speed": 1.8},
        {"name": "Nagawara", "lat": 13.0434, "lon": 77.6200, "pm25": 77, "pm10": 126, "no2": 43, "wind_speed": 2.2},
        {"name": "Nagarbhavi", "lat": 12.9591, "lon": 77.5122, "pm25": 64, "pm10": 105, "no2": 32, "wind_speed": 2.6},
        {"name": "Nelamangala", "lat": 13.1027, "lon": 77.3942, "pm25": 125, "pm10": 195, "no2": 72, "wind_speed": 2.8},
        {"name": "Peenya", "lat": 13.0329, "lon": 77.5273, "pm25": 98, "pm10": 161, "no2": 57, "wind_speed": 2.5},
        {"name": "Rajajinagar", "lat": 12.9917, "lon": 77.5525, "pm25": 60, "pm10": 98, "no2": 30, "wind_speed": 2.4},
        {"name": "Rajarajeshwari Nagar", "lat": 12.9274, "lon": 77.5154, "pm25": 57, "pm10": 93, "no2": 27, "wind_speed": 2.6},
        {"name": "Sahakara Nagar", "lat": 13.0626, "lon": 77.5870, "pm25": 53, "pm10": 87, "no2": 24, "wind_speed": 2.8},
        {"name": "Sarjapur", "lat": 12.9105, "lon": 77.6870, "pm25": 84, "pm10": 138, "no2": 46, "wind_speed": 2.0},
        {"name": "Shivajinagar", "lat": 12.9857, "lon": 77.6057, "pm25": 71, "pm10": 116, "no2": 40, "wind_speed": 1.9},
        {"name": "Silk Board", "lat": 12.9177, "lon": 77.6238, "pm25": 142, "pm10": 210, "no2": 85, "wind_speed": 1.5},
        {"name": "Thanisandra", "lat": 13.0569, "lon": 77.6320, "pm25": 69, "pm10": 113, "no2": 35, "wind_speed": 2.6},
        {"name": "Uttarahalli", "lat": 12.9050, "lon": 77.5522, "pm25": 55, "pm10": 90, "no2": 25, "wind_speed": 2.5},
        {"name": "Vasanth Nagar", "lat": 12.9911, "lon": 77.5913, "pm25": 63, "pm10": 103, "no2": 33, "wind_speed": 2.0},
        {"name": "Vijayanagar", "lat": 12.9719, "lon": 77.5360, "pm25": 66, "pm10": 108, "no2": 34, "wind_speed": 2.3},
        {"name": "Whitefield", "lat": 12.9698, "lon": 77.7500, "pm25": 89, "pm10": 146, "no2": 49, "wind_speed": 2.1},
        {"name": "Yelahanka", "lat": 13.1007, "lon": 77.5963, "pm25": 58, "pm10": 95, "no2": 28, "wind_speed": 2.9},
        {"name": "Yeshwanthpur", "lat": 13.0280, "lon": 77.5407, "pm25": 82, "pm10": 135, "no2": 49, "wind_speed": 2.2},
        {"name": "Old Airport Road", "lat": 12.9590, "lon": 77.6680, "pm25": 79, "pm10": 130, "no2": 44, "wind_speed": 2.0},
        {"name": "Kattigenahalli", "lat": 13.1150, "lon": 77.6250, "pm25": 56, "pm10": 92, "no2": 26, "wind_speed": 2.8},
    ]


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two coordinates."""
    lat1_rad, lat2_rad = radians(lat1), radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    haversine = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    )
    return 6371 * 2 * asin(sqrt(haversine))


@app.get("/api/regions/nearest")
def get_nearest_region(query: str):
    """Find the listed area nearest to a geocoded Bengaluru place name."""
    normalized_query = query.strip()
    if len(normalized_query) < 2 or len(normalized_query) > 100:
        raise HTTPException(status_code=400, detail="Enter an area name between 2 and 100 characters.")

    global _last_geocoding_request
    with _geocoding_lock:
        delay = 1.0 - (monotonic() - _last_geocoding_request)
        if delay > 0:
            sleep(delay)
        _last_geocoding_request = monotonic()

        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": f"{normalized_query}, Bengaluru, Karnataka, India",
                    "format": "json",
                    "limit": 1,
                    "countrycodes": "in",
                    "viewbox": "77.25,13.35,77.85,12.75",
                    "bounded": 1,
                },
                headers={"User-Agent": "BengaluruPollutionTracker/1.0"},
                timeout=8,
            )
            response.raise_for_status()
            geocoded_places = response.json()
        except requests.RequestException as error:
            raise HTTPException(
                status_code=502,
                detail="The area lookup service is unavailable. Please try again.",
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=502,
                detail="The area lookup service returned an invalid response.",
            ) from error

    if not geocoded_places:
        raise HTTPException(
            status_code=404,
            detail="Could not find that place within the Bengaluru area.",
        )

    place = geocoded_places[0]
    try:
        latitude = float(place["lat"])
        longitude = float(place["lon"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail="The area lookup service returned invalid coordinates.",
        ) from error

    nearest_region = min(
        get_all_regions(),
        key=lambda region: calculate_distance_km(
            latitude, longitude, region["lat"], region["lon"]
        ),
    )
    distance_km = calculate_distance_km(
        latitude, longitude, nearest_region["lat"], nearest_region["lon"]
    )

    return {
        "searched_area": normalized_query,
        "region": nearest_region,
        "distance_km": round(distance_km, 1),
    }