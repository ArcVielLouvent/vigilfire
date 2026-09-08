"""
fetch_power.py

Fetches daily meteorological data (temperature, humidity, precipitation,
wind speed) from the NASA POWER API for a single point and date range.
No API key required.

Docs: https://power.larc.nasa.gov/docs/services/api/temporal/daily/

NOTE: Must be run from an environment with internet access to
power.larc.nasa.gov — will NOT work inside this sandbox. Run it locally.
"""

import pandas as pd
import requests

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameters chosen specifically because they are established drivers of
# fire risk: max temperature, min relative humidity, precipitation
# (dryness), and wind speed (spread rate).
DEFAULT_PARAMETERS = ["T2M_MAX", "RH2M", "PRECTOTCORR", "WS10M"]


def fetch_weather_point(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    parameters: list[str] | None = None,
    community: str = "AG",
) -> pd.DataFrame:
    """
    Fetch daily weather data for a single point.

    Args:
        lat, lon: coordinates.
        start_date, end_date: 'YYYYMMDD' format (NASA POWER convention).
        parameters: list of POWER parameter codes. Defaults to fire-relevant
            variables. Max 20 parameters per request.
        community: 'AG' (agroclimatology), 'RE' (renewable energy), or
            'SB' (sustainable buildings). 'AG' has the most complete daily
            coverage for the variables we need.

    Returns:
        DataFrame indexed by date, one column per parameter.
    """
    parameters = parameters or DEFAULT_PARAMETERS
    params = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    param_data = payload["properties"]["parameter"]
    df = pd.DataFrame(param_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df


def compute_dryness_streak(df: pd.DataFrame, precip_col: str = "PRECTOTCORR", threshold_mm: float = 1.0) -> pd.Series:
    """
    Derived feature: consecutive number of days with precipitation below
    `threshold_mm`. This is a standard proxy for fuel dryness used in fire
    risk indices (e.g. Keetch-Byram Drought Index uses a similar idea).
    """
    is_dry = df[precip_col] < threshold_mm
    # Cumulative-sum the dry flags within each "wet-day-delimited" group.
    # (A naive .cumcount()+1 approach off-by-ones here: the wet day that
    # starts a new group would itself occupy count=0, pushing every dry
    # day after it one higher than the true consecutive count.)
    streak = is_dry.astype(int).groupby((~is_dry).cumsum()).cumsum()
    return streak


if __name__ == "__main__":
    # Quick manual smoke test — run this locally (not in the sandbox).
    df = fetch_weather_point(
        lat=-2.5, lon=113.0,  # Kalimantan, Indonesia
        start_date="20260801", end_date="20260907",
    )
    df["dryness_streak"] = compute_dryness_streak(df)
    print(df.tail())
