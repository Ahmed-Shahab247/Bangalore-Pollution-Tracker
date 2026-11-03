"""Download modelled Bengaluru air-quality and weather history from Open-Meteo."""

import argparse
import csv
from datetime import date, datetime, timedelta
import math
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo

import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "open_meteo_bengaluru_history.csv"
LATITUDE = 12.9716
LONGITUDE = 77.5946
LOCATION_NAME = "Bengaluru"
TIMEZONE = ZoneInfo("Asia/Kolkata")
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SOURCE = "Open-Meteo CAMS global air quality + historical forecast weather"
START_OF_AIR_QUALITY_ARCHIVE = date(2022, 8, 1)
AIR_QUALITY_FIELDS = ("pm2_5", "pm10", "nitrogen_dioxide")
WEATHER_FIELDS = (
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
)
CSV_FIELDS = (
    "location_name",
    "observed_at",
    "source",
    "data_type",
    "latitude",
    "longitude",
    "pm25",
    "pm10",
    "no2",
    "wind_speed",
    "wind_direction",
    "wind_u",
    "wind_v",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "air_quality_spatial_resolution_km",
    "air_quality_temporal_resolution_hours",
)


def _chunks(start: date, end: date):
    """Split the inclusive range into calendar-year requests."""
    cursor = start
    while cursor <= end:
        chunk_end = min(end, date(cursor.year, 12, 31))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _get_hourly(url: str, fields: tuple[str, ...], start: date, end: date):
    response = requests.get(
        url,
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(fields),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "Asia/Kolkata",
            **({"domains": "cams_global"} if url == AIR_QUALITY_URL else {}),
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ValueError(f"{url} returned no hourly time series for {start} to {end}.")

    times = hourly["time"]
    values = {}
    for field in fields:
        series = hourly.get(field)
        if not isinstance(series, list) or len(series) != len(times):
            raise ValueError(f"{url} returned an invalid hourly series for {field}.")
        values[field] = series

    return payload, times, values


def _finite_or_empty(value):
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return round(number, 3)


def fetch_history(start: date, end: date):
    rows_by_time = {}
    for chunk_start, chunk_end in _chunks(start, end):
        air_payload, air_times, air_values = _get_hourly(
            AIR_QUALITY_URL, AIR_QUALITY_FIELDS, chunk_start, chunk_end
        )
        _, weather_times, weather_values = _get_hourly(
            WEATHER_URL, WEATHER_FIELDS, chunk_start, chunk_end
        )
        if air_times != weather_times:
            raise ValueError(
                f"Air-quality and weather timestamps do not align for "
                f"{chunk_start} to {chunk_end}."
            )

        latitude = air_payload.get("latitude", LATITUDE)
        longitude = air_payload.get("longitude", LONGITUDE)
        for index, timestamp in enumerate(air_times):
            parsed_time = datetime.fromisoformat(timestamp).replace(tzinfo=TIMEZONE)
            speed = weather_values["wind_speed_10m"][index]
            direction = weather_values["wind_direction_10m"][index]
            wind_u = ""
            wind_v = ""
            if speed is not None and direction is not None:
                direction_radians = math.radians(float(direction))
                wind_u = _finite_or_empty(-float(speed) * math.sin(direction_radians))
                wind_v = _finite_or_empty(-float(speed) * math.cos(direction_radians))

            rows_by_time[parsed_time.isoformat()] = {
                "location_name": LOCATION_NAME,
                "observed_at": parsed_time.isoformat(),
                "source": SOURCE,
                "data_type": "modelled",
                "latitude": _finite_or_empty(latitude),
                "longitude": _finite_or_empty(longitude),
                "pm25": _finite_or_empty(air_values["pm2_5"][index]),
                "pm10": _finite_or_empty(air_values["pm10"][index]),
                "no2": _finite_or_empty(air_values["nitrogen_dioxide"][index]),
                "wind_speed": _finite_or_empty(speed),
                "wind_direction": _finite_or_empty(direction),
                "wind_u": wind_u,
                "wind_v": wind_v,
                "temperature_2m": _finite_or_empty(
                    weather_values["temperature_2m"][index]
                ),
                "relative_humidity_2m": _finite_or_empty(
                    weather_values["relative_humidity_2m"][index]
                ),
                "precipitation": _finite_or_empty(
                    weather_values["precipitation"][index]
                ),
                "air_quality_spatial_resolution_km": 45,
                "air_quality_temporal_resolution_hours": 3,
            }
        print(f"Fetched {chunk_start} through {chunk_end}: {len(air_times)} hourly rows.")
        time.sleep(0.2)

    return rows_by_time


def write_rows(output: Path, new_rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if output.exists():
        with output.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != list(CSV_FIELDS):
                raise ValueError(f"{output} has unexpected columns; refusing to overwrite it.")
            for row in reader:
                merged[row["observed_at"]] = row
    merged.update(new_rows)

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output.parent, delete=False
    ) as file:
        temporary_path = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(merged[key] for key in sorted(merged))
    temporary_path.replace(output)
    return len(merged)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical modelled air-quality and weather data for Bengaluru."
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=START_OF_AIR_QUALITY_ARCHIVE,
        help="Inclusive local date (default: 2022-08-01).",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=datetime.now(TIMEZONE).date() - timedelta(days=1),
        help="Inclusive local date (default: yesterday in Asia/Kolkata).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    if args.start_date < START_OF_AIR_QUALITY_ARCHIVE:
        parser.error("Open-Meteo's global air-quality archive starts in August 2022.")
    if args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date.")
    if args.end_date >= datetime.now(TIMEZONE).date():
        parser.error("--end-date must be before today; this collector only downloads history.")

    rows = fetch_history(args.start_date, args.end_date)
    total_rows = write_rows(args.output, rows)
    print(f"Saved {len(rows)} fetched rows ({total_rows} total) to {args.output}.")


if __name__ == "__main__":
    main()
