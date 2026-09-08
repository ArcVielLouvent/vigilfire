"""
fetch_firms.py

Fetches historical/recent active-fire hotspot detections from the NASA FIRMS
Area API for a given bounding box and day range.

Docs: https://firms.modaps.eosdis.nasa.gov/api/area/
Rate limit: 5,000 transactions / 10-minute window per MAP_KEY (free, instant
registration at https://firms.modaps.eosdis.nasa.gov/api/map_key/).

NOTE: This module must be run from an environment with internet access to
firms.modaps.eosdis.nasa.gov — it will NOT work inside this sandbox, which
only allowlists package-registry domains. Run it locally.
"""

import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# VIIRS_NOAA20_NRT / VIIRS_SNPP_NRT / MODIS_NRT are common near-real-time
# sensors. For long historical backtesting use the *_SP (standard processing)
# variants instead, which cover further back in time.
DEFAULT_SENSOR = "VIIRS_SNPP_NRT"


def fetch_fire_hotspots(
    bbox: tuple[float, float, float, float],
    day_range: int = 10,
    sensor: str = DEFAULT_SENSOR,
    map_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch fire hotspot detections within a bounding box.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) — note FIRMS uses
              lon,lat,lon,lat order, not lat,lon.
        day_range: number of days back from today to fetch (max 10 for NRT
              sensors in a single call; loop with explicit dates for longer
              historical windows).
        sensor: FIRMS sensor identifier.
        map_key: FIRMS MAP_KEY. Falls back to FIRMS_MAP_KEY env var.

    Returns:
        DataFrame with columns including latitude, longitude, acq_date,
        acq_time, confidence, frp (fire radiative power), etc.
    """
    map_key = map_key or os.getenv("FIRMS_MAP_KEY")
    if not map_key:
        raise ValueError(
            "FIRMS_MAP_KEY not set. Copy .env.example to .env and fill it in, "
            "or pass map_key= explicitly."
        )

    min_lon, min_lat, max_lon, max_lat = bbox
    area_coords = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    url = f"{BASE_URL}/{map_key}/{sensor}/{area_coords}/{day_range}"

    df = pd.read_csv(url)
    return df


def fetch_fire_hotspots_historical(
    bbox: tuple[float, float, float, float],
    date: str,
    day_range: int = 1,
    sensor: str = "VIIRS_SNPP_SP",
    map_key: str | None = None,
) -> pd.DataFrame:
    """
    Same as fetch_fire_hotspots, but for a specific historical date.
    Use the *_SP (Standard Processing) sensor variants for dates older than
    ~2 months, since NRT data typically only covers the recent past.

    Args:
        date: 'YYYY-MM-DD' — the anchor date to fetch from.
    """
    map_key = map_key or os.getenv("FIRMS_MAP_KEY")
    if not map_key:
        raise ValueError("FIRMS_MAP_KEY not set. See .env.example.")

    min_lon, min_lat, max_lon, max_lat = bbox
    area_coords = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    url = f"{BASE_URL}/{map_key}/{sensor}/{area_coords}/{day_range}/{date}"

    df = pd.read_csv(url)
    return df


if __name__ == "__main__":
    # Quick manual smoke test — run this locally (not in the sandbox).
    # Example bbox: Kalimantan, Indonesia (rough box)
    kalimantan_bbox = (108.5, -4.5, 119.0, 4.5)
    df = fetch_fire_hotspots(kalimantan_bbox, day_range=5)
    print(df.head())
    print(f"Total detections: {len(df)}")
