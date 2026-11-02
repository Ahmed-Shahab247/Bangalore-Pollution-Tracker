"""Live official-station readings and distance-weighted area estimates."""

from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from os import getenv
import re
from threading import Lock
from time import monotonic

import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_BENGALURU_BOUNDS = "77.25,12.75,78.00,13.35"
OFFICIAL_SOURCE_PATTERN = re.compile(
    r"\b(?:cpcb|central pollution control board|kspcb|"
    r"karnataka state pollution control board)\b"
)
POLLUTANT_NAMES = {
    "pm25": {"pm25", "pm2.5"},
    "pm10": {"pm10"},
    "no2": {"no2"},
    "so2": {"so2"},
    "o3": {"o3"},
    "co": {"co"},
    "wind_speed": {"wind_speed", "wind speed", "windspeed"},
}
UNIT_CONVERSIONS = {
    ("no2", "ppb"): 1.88,
    ("so2", "ppb"): 2.62,
    ("o3", "ppb"): 1.96,
    ("co", "ppm"): 1.145,
    ("co", "ppb"): 0.001145,
}
CLUSTER_BASE_FEATURES = ("pm25", "pm10", "no2")
PM25_BREAKPOINTS = (
    (0, 30, 0, 50, "Good"),
    (30, 60, 50, 100, "Satisfactory"),
    (60, 90, 100, 200, "Moderate"),
    (90, 120, 200, 300, "Poor"),
    (120, 250, 300, 400, "Very Poor"),
    (250, 500, 400, 500, "Severe"),
)
AQI_POLLUTANT_BREAKPOINTS = {
    "pm25": ((0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 500)),
    "pm10": ((0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 600)),
    "no2": ((0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 1000)),
    "so2": ((0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 2000)),
    "o3": ((0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 1000)),
    "co": ((0, 1), (1.1, 2), (2.1, 10), (10.1, 17), (17.1, 34), (34.1, 50)),
}
AQI_INDEX_RANGES = ((0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500))
AQI_CATEGORIES = (
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
)
MAX_STATION_AGE = timedelta(hours=6)
MAX_INTERPOLATION_DISTANCE_KM = 30
INTERPOLATION_STATION_COUNT = 4
_cache_lock = Lock()
_cache_updated_at = 0.0
_cached_regions = None


class AirQualityDataError(Exception):
    """A safe, user-facing data-provider failure."""


def _distance_km(lat1, lon1, lat2, lon2):
    lat1_rad, lat2_rad = radians(lat1), radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    haversine = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    )
    return 6371 * 2 * asin(sqrt(haversine))


def _source_name(location):
    owners = (
        location.get("owner"),
        location.get("provider"),
    )
    names = [
        entry.get("name")
        for entry in owners
        if isinstance(entry, dict)
    ]
    licenses = location.get("licenses") or []
    names.extend(
        license_info.get("attribution", {}).get("name")
        for license_info in licenses
        if isinstance(license_info, dict)
        and isinstance(license_info.get("attribution"), dict)
    )
    return " / ".join(
        dict.fromkeys(
            name.strip()
            for name in names
            if isinstance(name, str) and name.strip()
        )
    )


def _is_official_source(location):
    source = _source_name(location).casefold()
    return OFFICIAL_SOURCE_PATTERN.search(source) is not None


def _parse_datetime(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _normalize_unit(unit):
    return (
        (unit or "").lower()
        .replace("μ", "µ")
        .replace(" ", "")
        .replace("³", "3")
    )


def _unit_is_supported(parameter, unit):
    normalized = _normalize_unit(unit)
    if parameter == "wind_speed":
        if normalized in {"m/s", "ms-1"}:
            return 1.0
        if normalized in {"km/h", "kmh-1"}:
            return 1 / 3.6
    if parameter == "co":
        if normalized in {"µg/m3", "ug/m3", "µgm-3", "ugm-3", "microgram/m3"}:
            return 0.001
        if normalized in {"mg/m3", "mgm-3", "milligram/m3"}:
            return 1.0
        return UNIT_CONVERSIONS.get((parameter, normalized))
    if normalized in {"µg/m3", "ug/m3", "µgm-3", "ugm-3", "microgram/m3"}:
        return 1.0
    return UNIT_CONVERSIONS.get((parameter, normalized))


def add_pm25_index(region):
    """Attach an indicative PM2.5 sub-index, not an official CPCB AQI."""
    concentration = region.get("pm25")
    if not isinstance(concentration, (int, float)):
        region["pm25_sub_index"] = None
        region["pm25_category"] = None
        return

    for low, high, index_low, index_high, category in PM25_BREAKPOINTS:
        if low <= concentration <= high:
            sub_index = round(
                (index_high - index_low) * (concentration - low) / (high - low)
                + index_low
            )
            region["pm25_sub_index"] = sub_index
            region["pm25_category"] = category
            region["pm25_index_basis"] = "Indicative PM2.5 sub-index; not official CPCB AQI"
            return

    region["pm25_sub_index"] = 500
    region["pm25_category"] = "Severe"
    region["pm25_index_basis"] = "Indicative PM2.5 sub-index; not official CPCB AQI"


def add_indicative_aqi(region):
    """Estimate an AQI from fresh latest readings, not official averaged CPCB data."""
    particulate_available = any(
        isinstance(region.get(parameter), (int, float))
        and np.isfinite(region[parameter])
        and region[parameter] >= 0
        for parameter in ("pm25", "pm10")
    )
    sub_indices = {}
    for parameter, ranges in AQI_POLLUTANT_BREAKPOINTS.items():
        concentration = region.get(parameter)
        if (
            not isinstance(concentration, (int, float))
            or not np.isfinite(concentration)
            or concentration < 0
        ):
            continue

        for band, ((concentration_low, concentration_high), (index_low, index_high)) in enumerate(
            zip(ranges, AQI_INDEX_RANGES)
        ):
            if concentration <= concentration_high or band == len(ranges) - 1:
                ratio = min(
                    1.0,
                    max(
                        0.0,
                        (concentration - concentration_low)
                        / (concentration_high - concentration_low),
                    ),
                )
                sub_indices[parameter] = round(
                    index_low + ratio * (index_high - index_low)
                )
                break

    region["indicative_aqi"] = None
    region["indicative_aqi_category"] = None
    region["indicative_aqi_dominant_pollutant"] = None
    region["indicative_aqi_pollutants"] = sorted(
        parameter.upper() for parameter in sub_indices
    )
    region["indicative_aqi_basis"] = (
        "Indicative estimate from latest available readings; CPCB averaging periods "
        "and completeness rules are not applied."
    )
    if not particulate_available or len(sub_indices) < 3:
        region["indicative_aqi_unavailable_reason"] = (
            "At least three current pollutant readings, including PM2.5 or PM10, are required."
        )
        return

    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    aqi = sub_indices[dominant_pollutant]
    region["indicative_aqi"] = aqi
    region["indicative_aqi_dominant_pollutant"] = dominant_pollutant.upper()
    region["indicative_aqi_category"] = next(
        category for maximum, category in AQI_CATEGORIES if aqi <= maximum
    )
    region["indicative_aqi_unavailable_reason"] = None


def _cluster_features(stations):
    complete_base_stations = [
        station
        for station in stations
        if all(isinstance(station.get(feature), (int, float)) for feature in CLUSTER_BASE_FEATURES)
    ]
    if len(complete_base_stations) >= 3 and all(
        isinstance(station.get("wind_speed"), (int, float))
        for station in complete_base_stations
    ):
        return (*CLUSTER_BASE_FEATURES, "wind_speed")
    return CLUSTER_BASE_FEATURES


def _request_openaq(path, api_key, params=None):
    try:
        response = requests.get(
            f"{OPENAQ_BASE_URL}{path}",
            headers={"X-API-Key": api_key},
            params=params,
            timeout=12,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise AirQualityDataError(
            "Could not retrieve live station data from OpenAQ. Check the API key and network, then try again."
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise AirQualityDataError("OpenAQ returned an invalid response.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise AirQualityDataError("OpenAQ returned data in an unexpected format.")
    return payload["results"]


def fetch_official_stations(api_key=None):
    """Fetch fresh measurements only from CPCB/KSPCB reference stations."""
    api_key = api_key or getenv("OPENAQ_API_KEY")
    if not api_key:
        raise AirQualityDataError(
            "Live readings are not configured. Set OPENAQ_API_KEY on the backend and restart it."
        )

    locations = _request_openaq(
        "/locations",
        api_key,
        params={
            "bbox": OPENAQ_BENGALURU_BOUNDS,
            "iso": "IN",
            "monitor": "true",
            "limit": 1000,
            "page": 1,
        },
    )
    official_locations = [
        location
        for location in locations
        if location.get("isMonitor") is True
        and location.get("isMobile") is False
        and _is_official_source(location)
        and isinstance(location.get("coordinates"), dict)
        and location.get("coordinates", {}).get("latitude") is not None
        and location.get("coordinates", {}).get("longitude") is not None
    ]
    if not official_locations:
        raise AirQualityDataError(
            "OpenAQ returned no CPCB/KSPCB reference stations in the Bengaluru coverage area."
        )

    cutoff = datetime.now(timezone.utc) - MAX_STATION_AGE
    stations = []
    for location in official_locations:
        location_id = location.get("id")
        if location_id is None:
            continue

        sensors = {}
        for sensor in location.get("sensors") or []:
            parameter_info = sensor.get("parameter") or {}
            parameter_name = str(parameter_info.get("name") or "").casefold()
            conversion = None
            parameter = next(
                (
                    canonical
                    for canonical, aliases in POLLUTANT_NAMES.items()
                    if parameter_name in aliases
                ),
                None,
            )
            if parameter:
                conversion = _unit_is_supported(
                    parameter, parameter_info.get("units")
                )
            if parameter and conversion is not None and sensor.get("id") is not None:
                sensors[sensor["id"]] = (
                    parameter,
                    conversion,
                )
        if not sensors:
            continue

        latest = _request_openaq(
            f"/locations/{location_id}/latest",
            api_key,
            params={"limit": 100, "page": 1},
        )
        measurements = {}
        measurement_times = []
        for measurement in latest:
            sensor = sensors.get(measurement.get("sensorsId"))
            observed_at = _parse_datetime(
                (measurement.get("datetime") or {}).get("utc")
            )
            value = measurement.get("value")
            if (
                sensor is None
                or observed_at is None
                or observed_at < cutoff
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                continue
            parameter, conversion = sensor
            value = float(value) * conversion
            previous = measurements.get(parameter)
            if previous is None or observed_at > previous[1]:
                measurements[parameter] = (value, observed_at)
            measurement_times.append(observed_at)

        if not measurements:
            continue

        coordinates = location["coordinates"]
        station = {
            "name": location.get("name") or f"Station {location_id}",
            "id": f"openaq-location-{location_id}",
            "lat": float(coordinates["latitude"]),
            "lon": float(coordinates["longitude"]),
            "source": _source_name(location),
            "data_method": "measured",
            "is_interpolated": False,
            "source_station_count": 1,
            "source_station_names": [location.get("name") or f"Station {location_id}"],
            "observed_at": max(measurement_times).isoformat(),
        }
        station.update(
            {
                parameter: round(value, 2)
                for parameter, (value, _) in measurements.items()
            }
        )
        stations.append(station)

    if not stations:
        raise AirQualityDataError(
            "No fresh pollutant measurements are available from CPCB/KSPCB reference stations."
        )
    return stations


def interpolate_area(area, stations):
    """Estimate an area's pollutants using nearby fresh official stations."""
    result = {
        "name": area["name"],
        "id": f"area-{area['name'].casefold().replace(' ', '-')}",
        "is_area": True,
        "lat": area["lat"],
        "lon": area["lon"],
        "source": "OpenAQ CPCB/KSPCB reference stations",
        "data_method": "distance_weighted_estimate",
        "is_interpolated": True,
        "observed_at": None,
        "nearest_station_distance_km": None,
        "source_station_count": 0,
        "source_station_names": [],
    }

    distances = sorted(
        (
            (
                _distance_km(area["lat"], area["lon"], station["lat"], station["lon"]),
                station,
            )
            for station in stations
        ),
        key=lambda item: item[0],
    )
    result["nearest_station_distance_km"] = round(distances[0][0], 1)
    selected_stations = [
        (distance, station)
        for distance, station in distances
        if distance <= MAX_INTERPOLATION_DISTANCE_KM
    ][:INTERPOLATION_STATION_COUNT]
    if selected_stations and selected_stations[0][0] < 0.1:
        station = selected_stations[0][1]
        result.update(station)
        result["name"] = area["name"]
        result["nearest_station_distance_km"] = 0.0
        add_pm25_index(result)
        return result

    used_stations = {}
    used_timestamps = []
    for parameter in POLLUTANT_NAMES:
        available = [
            (distance, station)
            for distance, station in selected_stations
            if parameter in station
        ]
        if not available:
            continue
        weights = [1 / max(distance, 0.1) ** 2 for distance, _ in available]
        result[parameter] = round(
            sum(weight * station[parameter] for weight, (_, station) in zip(weights, available))
            / sum(weights),
            2,
        )
        used_stations.update(
            (station["id"], station["name"]) for _, station in available
        )
        used_timestamps.extend(
            station["observed_at"] for _, station in available if station.get("observed_at")
        )

    result["source_station_count"] = len(used_stations)
    result["source_station_names"] = sorted(set(used_stations.values()))
    if used_timestamps:
        result["observed_at"] = min(used_timestamps)
    if not any(parameter in result for parameter in POLLUTANT_NAMES):
        result["data_method"] = "unavailable"
    add_pm25_index(result)
    return result


def add_live_clusters(stations, areas):
    """Fit zones from fresh measured stations, then predict for estimated areas."""
    for region in [*stations, *areas]:
        region["cluster_id"] = None
        region["cluster_profile"] = "Not enough complete live stations"
    for station in stations:
        add_pm25_index(station)

    features = _cluster_features(stations)
    usable_stations = [
        station
        for station in stations
        if all(
            isinstance(station.get(feature), (int, float))
            for feature in features
        )
    ]
    if len(usable_stations) < 3:
        for region in [*stations, *areas]:
            region["cluster_id"] = None
            region["cluster_profile"] = "Not enough complete live stations"
        return

    scaler = StandardScaler()
    station_values = np.array(
        [[station[feature] for feature in features] for station in usable_stations]
    )
    cluster_count = min(3, len(usable_stations), len(np.unique(station_values, axis=0)))
    if cluster_count < 2:
        for region in [*stations, *areas]:
            region["cluster_profile"] = "Not enough distinct live stations"
        return
    scaled_values = scaler.fit_transform(station_values)
    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10,
    )
    with threadpool_limits(limits=1):
        station_clusters = model.fit_predict(scaled_values)
    cluster_levels = {}
    for cluster_id in range(model.n_clusters):
        members = station_values[station_clusters == cluster_id]
        cluster_levels[cluster_id] = float(members[:, 0].mean())
    ordered_clusters = sorted(cluster_levels, key=cluster_levels.get)
    profiles = {
        cluster_id: f"Live zone {rank + 1} of {len(ordered_clusters)} (relative PM2.5)"
        for rank, cluster_id in enumerate(ordered_clusters)
    }
    for station, cluster_id in zip(usable_stations, station_clusters):
        station["cluster_id"] = int(cluster_id)
        station["cluster_profile"] = profiles[int(cluster_id)]
        add_pm25_index(station)

    usable_areas = [
        area
        for area in areas
        if all(
            isinstance(area.get(feature), (int, float))
            for feature in features
        )
    ]
    if usable_areas:
        area_values = np.array(
            [[area[feature] for feature in features] for area in usable_areas]
        )
        with threadpool_limits(limits=1):
            predicted_clusters = model.predict(scaler.transform(area_values))
        for area, cluster_id in zip(usable_areas, predicted_clusters):
            area["cluster_id"] = int(cluster_id)
            area["cluster_profile"] = profiles[int(cluster_id)]

def get_live_regions(area_locations, force_refresh=False):
    """Return fresh measured station nodes and interpolated named areas."""
    global _cache_updated_at, _cached_regions
    with _cache_lock:
        if (
            not force_refresh
            and _cached_regions is not None
            and monotonic() - _cache_updated_at < 300
        ):
            return _cached_regions

        stations = fetch_official_stations()
        areas = [interpolate_area(area, stations) for area in area_locations]
        for region in [*stations, *areas]:
            add_indicative_aqi(region)
        add_live_clusters(stations, areas)
        result = stations + areas
        _cached_regions = result
        _cache_updated_at = monotonic()
        return result


def predict_live_cluster(payload):
    """Fit the current station model and predict a pollutant fingerprint."""
    stations = fetch_official_stations()
    features = _cluster_features(stations)
    usable = [
        station
        for station in stations
        if all(
            isinstance(station.get(feature), (int, float))
            for feature in features
        )
    ]
    if len(usable) < 3:
        raise AirQualityDataError(
            "At least three fresh complete CPCB/KSPCB station observations are required for live clustering."
        )

    if "wind_speed" in features and payload.wind_speed is None:
        raise AirQualityDataError(
            "The live station model includes wind speed; provide wind_speed to classify this reading."
        )
    values = np.array([[station[feature] for feature in features] for station in usable])
    cluster_count = min(3, len(usable), len(np.unique(values, axis=0)))
    if cluster_count < 2:
        raise AirQualityDataError(
            "The current station readings do not contain enough distinct pollutant profiles for live clustering."
        )
    scaler = StandardScaler().fit(values)
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit(
            scaler.transform(values)
        )
        predicted_cluster = model.predict(
            scaler.transform(
                [[getattr(payload, feature) for feature in features]]
            )
        )
        cluster_id = int(predicted_cluster[0])
    cluster_levels = {
        cluster: float(values[model.labels_ == cluster, 0].mean())
        for cluster in range(model.n_clusters)
    }
    ordered_clusters = sorted(cluster_levels, key=cluster_levels.get)
    profile = (
        f"Live zone {ordered_clusters.index(cluster_id) + 1} "
        f"of {len(ordered_clusters)} (relative PM2.5)"
    )
    return {
        "cluster_id": cluster_id,
        "cluster_profile": profile,
        "cluster_features": list(features),
        "model_source": "Current CPCB/KSPCB OpenAQ station readings",
    }
