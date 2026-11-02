from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from math import asin, cos, radians, sin, sqrt
import os
from pathlib import Path
from typing import Optional
from threading import Lock
from time import monotonic, sleep
import requests
from app.data_pipeline import (
    AirQualityDataError,
    get_live_regions,
    predict_live_cluster,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

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

class PollutionPayload(BaseModel):
    location_name: str
    pm25: float
    pm10: float
    no2: float
    wind_speed: Optional[float] = None

@app.get("/")
def read_root():
    return {
        "status": "Active",
        "region": "Greater Bengaluru Air Quality Engine",
        "live_data_configured": bool(os.getenv("OPENAQ_API_KEY")),
    }

@app.post("/api/predict-cluster")
def predict_cluster(payload: PollutionPayload):
    try:
        cluster = predict_live_cluster(payload)
    except AirQualityDataError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"location": payload.location_name, **cluster}

def get_area_locations():
    """Return display locations; measurements are always fetched separately."""
    areas = [
        {"name": "Arekere", "lat": 12.8877, "lon": 77.5970},
        {"name": "Bagalur", "lat": 13.1340, "lon": 77.6680},
        {"name": "Banashankari", "lat": 12.9255, "lon": 77.5642},
        {"name": "Basaveshwaranagar", "lat": 12.9905, "lon": 77.5380},
        {"name": "BEL Road", "lat": 13.0340, "lon": 77.5700},
        {"name": "Begur", "lat": 12.8788, "lon": 77.6248},
        {"name": "Bellandur", "lat": 12.9304, "lon": 77.6784},
        {"name": "Bommanahalli", "lat": 12.9089, "lon": 77.6243},
        {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101},
        {"name": "CV Raman Nagar", "lat": 12.9850, "lon": 77.6630},
        {"name": "Devanahalli", "lat": 13.2477, "lon": 77.7130},
        {"name": "Domlur", "lat": 12.9609, "lon": 77.6387},
        {"name": "Doddaballapur", "lat": 13.2927, "lon": 77.5389},
        {"name": "Electronic City", "lat": 12.8399, "lon": 77.6770},
        {"name": "Ejipura", "lat": 12.9454, "lon": 77.6266},
        {"name": "Hebbal", "lat": 13.0358, "lon": 77.5970},
        {"name": "Hennur", "lat": 13.0358, "lon": 77.6395},
        {"name": "Hoskote", "lat": 13.0731, "lon": 77.7997},
        {"name": "HSR Layout", "lat": 12.9116, "lon": 77.6389},
        {"name": "Indiranagar", "lat": 12.9784, "lon": 77.6408},
        {"name": "Jakkur", "lat": 13.0785, "lon": 77.6060},
        {"name": "Jalahalli", "lat": 13.0500, "lon": 77.5460},
        {"name": "Jayanagar", "lat": 12.9250, "lon": 77.5838},
        {"name": "JP Nagar", "lat": 12.9063, "lon": 77.5857},
        {"name": "Kengeri", "lat": 12.9177, "lon": 77.4838},
        {"name": "Koramangala", "lat": 12.9352, "lon": 77.6245},
        {"name": "KR Puram", "lat": 13.0075, "lon": 77.6959},
        {"name": "Mahadevapura", "lat": 12.9918, "lon": 77.6878},
        {"name": "Malleshwaram", "lat": 13.0031, "lon": 77.5643},
        {"name": "Marathahalli", "lat": 12.9591, "lon": 77.6974},
        {"name": "MG Road", "lat": 12.9757, "lon": 77.6068},
        {"name": "Nagawara", "lat": 13.0434, "lon": 77.6200},
        {"name": "Nagarbhavi", "lat": 12.9591, "lon": 77.5122},
        {"name": "Nelamangala", "lat": 13.1027, "lon": 77.3942},
        {"name": "Peenya", "lat": 13.0329, "lon": 77.5273},
        {"name": "Rajajinagar", "lat": 12.9917, "lon": 77.5525},
        {"name": "Rajarajeshwari Nagar", "lat": 12.9274, "lon": 77.5154},
        {"name": "Sahakara Nagar", "lat": 13.0626, "lon": 77.5870},
        {"name": "Sarjapur", "lat": 12.9105, "lon": 77.6870},
        {"name": "Shivajinagar", "lat": 12.9857, "lon": 77.6057},
        {"name": "Silk Board", "lat": 12.9177, "lon": 77.6238},
        {"name": "Thanisandra", "lat": 13.0569, "lon": 77.6320},
        {"name": "Uttarahalli", "lat": 12.9050, "lon": 77.5522},
        {"name": "Vasanth Nagar", "lat": 12.9911, "lon": 77.5913},
        {"name": "Vijayanagar", "lat": 12.9719, "lon": 77.5360},
        {"name": "Whitefield", "lat": 12.9698, "lon": 77.7500},
        {"name": "Yelahanka", "lat": 13.1007, "lon": 77.5963},
        {"name": "Yeshwanthpur", "lat": 13.0280, "lon": 77.5407},
        {"name": "Old Airport Road", "lat": 12.9590, "lon": 77.6680},
        {"name": "Kattigenahalli", "lat": 13.1150, "lon": 77.6250},
    ]
    return [
        {"name": area["name"], "lat": area["lat"], "lon": area["lon"]}
        for area in areas
    ]

@app.get("/api/regions")
def get_all_regions():
    try:
        return get_live_regions(get_area_locations())
    except AirQualityDataError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

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

    try:
        regions = get_all_regions()
    except HTTPException as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error
    named_areas = [region for region in regions if region.get("is_area")]
    if not named_areas:
        raise HTTPException(
            status_code=503,
            detail="No named neighborhood areas are available for this lookup.",
        )
    nearest_region = min(
        named_areas,
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