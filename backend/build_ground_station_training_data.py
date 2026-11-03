"""Build a station-observation/weather training CSV from OpenAQ v3 and Open-Meteo."""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
import tempfile
import time
from time import monotonic
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEATHER = BASE_DIR / "data" / "open_meteo_bengaluru_history.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "bengaluru_ground_station_training.csv"
OPENAQ_URL = "https://api.openaq.org/v3"
BENGALURU_BOUNDS = "77.25,12.75,78.00,13.35"
LOCAL_TIMEZONE = ZoneInfo("Asia/Kolkata")
PAGE_SIZE = 1000
QUERY_CHUNK = timedelta(days=90)
MIN_REQUEST_INTERVAL_SECONDS = 1.0
MAX_MEASUREMENT_INTERVAL = pd.Timedelta(hours=3)
MAX_PM_CONCENTRATION = 2000.0
OFFICIAL_SOURCE = re.compile(
    r"\b(?:cpcb|central pollution control board|kspcb|"
    r"karnataka state pollution control board)\b",
    re.IGNORECASE,
)
OUTPUT_COLUMNS = (
    "location_name",
    "observed_at",
    "source",
    "latitude",
    "longitude",
    "pm25",
    "pm10",
    "wind_speed",
    "wind_direction",
    "wind_u",
    "wind_v",
    "temperature_2m",
    "relative_humidity_2m",
)
WEATHER_COLUMNS = (
    "wind_speed",
    "wind_direction",
    "wind_u",
    "wind_v",
    "temperature_2m",
    "relative_humidity_2m",
)
UNIT_ALIASES = {
    "µg/m3",
    "ug/m3",
    "µgm-3",
    "ugm-3",
    "microgram/m3",
    "micrograms/m3",
}


def _source_name(location):
    names = []
    for key in ("owner", "provider"):
        value = location.get(key)
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            names.append(value["name"].strip())
    for license_info in location.get("licenses") or []:
        if not isinstance(license_info, dict):
            continue
        attribution = license_info.get("attribution")
        if isinstance(attribution, dict) and isinstance(attribution.get("name"), str):
            names.append(attribution["name"].strip())
    return " / ".join(dict.fromkeys(name for name in names if name))


def _normalize_unit(unit):
    return (
        str(unit or "")
        .casefold()
        .replace("μ", "µ")
        .replace(" ", "")
        .replace("³", "3")
    )


def _parse_utc(value):
    if not isinstance(value, str):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp.tz_convert("UTC")


def _period_interval(period):
    interval = period.get("interval")
    if isinstance(interval, str):
        try:
            return pd.Timedelta(interval)
        except ValueError:
            return None
    start = _parse_utc((period.get("datetimeFrom") or {}).get("utc"))
    end = _parse_utc((period.get("datetimeTo") or {}).get("utc"))
    if start is not None and end is not None:
        return end - start
    return None


class OpenAQClient:
    def __init__(self, api_key):
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})
        self.last_request_at = 0.0

    def get_results(self, path, params=None):
        url = f"{OPENAQ_URL}{path}"
        for attempt in range(5):
            elapsed = monotonic() - self.last_request_at
            if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
            try:
                response = self.session.get(url, params=params, timeout=(10, 60))
                self.last_request_at = monotonic()
            except requests.RequestException as error:
                if attempt == 4:
                    raise RuntimeError(f"OpenAQ request failed for {path}: {error}") from error
                time.sleep(2**attempt)
                continue

            if (
                response.status_code in {408, 425, 429}
                or response.status_code >= 500
            ):
                if attempt == 4:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(60.0, max(1.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = 2**attempt
                time.sleep(delay)
                continue
            try:
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as error:
                raise RuntimeError(f"OpenAQ returned an invalid response for {path}.") from error
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise RuntimeError(f"OpenAQ returned an unexpected response for {path}.")
            return payload["results"]
        raise RuntimeError(f"OpenAQ request exhausted retries for {path}.")

    def locations(self):
        results = self.get_results(
            "/locations",
            params={
                "bbox": BENGALURU_BOUNDS,
                "iso": "IN",
                "monitor": "true",
                "limit": PAGE_SIZE,
                "page": 1,
            },
        )
        return [
            location
            for location in results
            if location.get("isMonitor") is True
            and location.get("isMobile") is False
            and OFFICIAL_SOURCE.search(_source_name(location))
            and isinstance(location.get("coordinates"), dict)
        ]

    def sensor_metadata(self, sensor_id):
        results = self.get_results(f"/sensors/{sensor_id}")
        return results[0] if results else {}

    def sensor_measurements(self, sensor_id, start, end):
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + QUERY_CHUNK, end)
            page = 1
            while True:
                results = self.get_results(
                    f"/sensors/{sensor_id}/measurements",
                    params={
                        "datetime_from": cursor.isoformat().replace("+00:00", "Z"),
                        "datetime_to": chunk_end.isoformat().replace("+00:00", "Z"),
                        "limit": PAGE_SIZE,
                        "page": page,
                    },
                )
                yield from results
                if len(results) < PAGE_SIZE:
                    break
                page += 1
            cursor = chunk_end


def _sensor_parameters(location):
    for sensor in location.get("sensors") or []:
        parameter = sensor.get("parameter") or {}
        name = str(parameter.get("name") or "").casefold().replace(".", "")
        if (
            name in {"pm25", "pm10"}
            and sensor.get("id") is not None
            and _normalize_unit(parameter.get("units")) in UNIT_ALIASES
        ):
            yield name, sensor["id"]


def fetch_station_rows(client, start, end, station_filters=None):
    measurements = defaultdict(list)
    stations_seen = 0
    sensors_seen = 0
    normalized_filters = tuple(
        name.casefold().strip() for name in (station_filters or ()) if name.strip()
    )
    for location in client.locations():
        station_name = str(location.get("name") or "").strip()
        if normalized_filters and not any(
            name in station_name.casefold() for name in normalized_filters
        ):
            continue
        sensors = list(_sensor_parameters(location))
        if not sensors:
            continue
        coordinates = location["coordinates"]
        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")
        if latitude is None or longitude is None:
            continue
        source = _source_name(location)
        if not station_name:
            continue

        stations_seen += 1
        print(
            f"Collecting {len(sensors)} PM sensor(s) from {station_name}.",
            flush=True,
        )
        for pollutant, sensor_id in sensors:
            sensors_seen += 1
            metadata = client.sensor_metadata(sensor_id)
            first = _parse_utc((metadata.get("datetimeFirst") or {}).get("utc"))
            last = _parse_utc((metadata.get("datetimeLast") or {}).get("utc"))
            if (last is not None and last < start) or (
                first is not None and first > end
            ):
                continue
            for result in client.sensor_measurements(sensor_id, start, end):
                if (result.get("flagInfo") or {}).get("hasFlags") is True:
                    continue
                value = result.get("value")
                try:
                    concentration = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(concentration) or not 0 <= concentration <= MAX_PM_CONCENTRATION:
                    continue

                period = result.get("period")
                if not isinstance(period, dict):
                    continue
                interval = _period_interval(period)
                if (
                    interval is None
                    or interval <= pd.Timedelta(0)
                    or interval > MAX_MEASUREMENT_INTERVAL
                ):
                    continue
                observed_at = _parse_utc((period.get("datetimeTo") or {}).get("utc"))
                if observed_at is None:
                    observed_at = _parse_utc((period.get("datetimeFrom") or {}).get("utc"))
                if observed_at is None or observed_at < start or observed_at > end:
                    continue

                local_hour = observed_at.tz_convert(LOCAL_TIMEZONE).floor("h")
                key = (station_name, local_hour)
                measurements[key].append(
                    {
                        "pollutant": pollutant,
                        "value": concentration,
                        "source": source,
                        "latitude": float(latitude),
                        "longitude": float(longitude),
                    }
                )
    return measurements, stations_seen, sensors_seen


def load_weather(path):
    if not path.exists():
        raise FileNotFoundError(f"Open-Meteo weather CSV not found: {path}")
    weather = pd.read_csv(path)
    required = {"observed_at", *WEATHER_COLUMNS}
    missing = required.difference(weather.columns)
    if missing:
        raise ValueError(
            f"Weather CSV is missing required columns: {', '.join(sorted(missing))}."
        )

    weather["observed_at"] = pd.to_datetime(
        weather["observed_at"], utc=True, errors="coerce"
    )
    for column in WEATHER_COLUMNS:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    weather = weather.dropna(subset=["observed_at", *WEATHER_COLUMNS])
    weather["observed_at"] = (
        weather["observed_at"].dt.tz_convert(LOCAL_TIMEZONE).dt.floor("h")
    )
    weather = (
        weather.groupby("observed_at", as_index=False)[list(WEATHER_COLUMNS)]
        .mean()
        .set_index("observed_at")
        .sort_index()
    )
    if weather.empty:
        raise ValueError(f"No complete hourly weather records found in {path}.")
    return weather


def merge_training_rows(measurements, weather):
    grouped = defaultdict(lambda: {"pm25": [], "pm10": [], "sources": [], "coords": []})
    for (station_name, local_hour), records in measurements.items():
        for record in records:
            entry = grouped[(station_name, local_hour)]
            entry[record["pollutant"]].append(record["value"])
            entry["sources"].append(record["source"])
            entry["coords"].append((record["latitude"], record["longitude"]))

    rows = []
    for (station_name, local_hour), values in sorted(grouped.items()):
        if local_hour not in weather.index:
            continue
        station_weather = weather.loc[local_hour]
        if isinstance(station_weather, pd.DataFrame):
            station_weather = station_weather.mean(numeric_only=True)
        if not station_weather.loc[list(WEATHER_COLUMNS)].notna().all():
            continue
        if not values["pm25"] and not values["pm10"]:
            continue
        latitude, longitude = values["coords"][0]
        rows.append(
            {
                "location_name": station_name,
                "observed_at": local_hour.isoformat(),
                "source": "OpenAQ / " + " / ".join(dict.fromkeys(values["sources"])),
                "latitude": latitude,
                "longitude": longitude,
                "pm25": float(np.mean(values["pm25"])) if values["pm25"] else np.nan,
                "pm10": float(np.mean(values["pm10"])) if values["pm10"] else np.nan,
                **{column: float(station_weather[column]) for column in WEATHER_COLUMNS},
            }
        )

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if frame.empty:
        return frame
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(
        subset=[
            "location_name",
            "observed_at",
            "source",
            "latitude",
            "longitude",
            *WEATHER_COLUMNS,
        ]
    )
    frame = frame[frame["pm25"].notna() | frame["pm10"].notna()]
    frame = frame.sort_values(["observed_at", "location_name"]).reset_index(drop=True)
    return frame.loc[:, OUTPUT_COLUMNS]


def write_csv(frame, output):
    if frame.empty:
        raise ValueError("Cannot write an empty training dataset.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        frame.to_csv(temporary, index=False, columns=OUTPUT_COLUMNS, float_format="%.4f")
    temporary_path.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months",
        type=int,
        choices=(12, 24),
        default=24,
        help="Historical query window (default: 24 months).",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=[],
        metavar="NAME",
        help=(
            "Optional case-insensitive station-name filters (for example: "
            "'Peenya' 'Silk Board'). By default, all eligible Bengaluru monitors are queried."
        ),
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=DEFAULT_WEATHER,
        help=f"Open-Meteo meteorology CSV (default: {DEFAULT_WEATHER}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output training CSV (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        parser.error(
            "OPENAQ_API_KEY is not set. Add it to backend/.env or the environment."
        )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=round(args.months * 365.2425 / 12))
    client = OpenAQClient(api_key)
    measurements, station_count, sensor_count = fetch_station_rows(
        client, start, end, args.stations
    )
    weather = load_weather(args.weather)
    merged = merge_training_rows(measurements, weather)
    if merged.empty:
        raise SystemExit(
            "No valid ground-station observations overlap the selected date range "
            "and Open-Meteo weather timestamps. No output file was written. "
            "OpenAQ may not expose recent historical measurements for these stations."
        )
    write_csv(merged, args.output)
    print(
        f"Collected {len(merged)} matched hourly rows from {station_count} "
        f"eligible stations and {sensor_count} PM sensors."
    )
    print(f"Date range: {start.isoformat()} through {end.isoformat()}.")
    print(f"Saved unified ground-station/weather dataset to {args.output}.")
    print(
        "Rows use CPCB/KSPCB-attributed OpenAQ measurements. Historical readings "
        "are retained within the requested window; records with invalid values, "
        "unsupported units, or measurement intervals over 3 hours are excluded."
    )


if __name__ == "__main__":
    main()
