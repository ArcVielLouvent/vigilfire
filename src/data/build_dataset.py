"""
build_dataset.py

Builds a labeled training dataset by combining:
  - Positive examples: locations/dates where FIRMS recorded an actual fire
    detection (label = 1).
  - Negative examples: randomly sampled locations/dates within the same
    region where no fire was detected within a buffer window (label = 0).

Each row gets joined with the preceding weather conditions from NASA POWER
(e.g. the week leading up to the date), since fire risk is driven by
*antecedent* dryness/heat, not same-day weather.

Run this locally (needs internet access to both FIRMS and POWER APIs).
"""

import random
from datetime import datetime, timedelta

import pandas as pd

from src.data.fetch_firms import fetch_fire_hotspots_historical
from src.data.fetch_power import compute_dryness_streak, fetch_weather_point

LOOKBACK_DAYS = 7  # how many days of preceding weather to summarize per sample


def _weather_features_for(lat: float, lon: float, anchor_date: datetime) -> dict:
    """Summarize the LOOKBACK_DAYS of weather preceding anchor_date into features."""
    start = (anchor_date - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    end = (anchor_date - timedelta(days=1)).strftime("%Y%m%d")

    weather = fetch_weather_point(lat, lon, start, end)
    weather["dryness_streak"] = compute_dryness_streak(weather)

    return {
        "t2m_max_avg": weather["T2M_MAX"].mean(),
        "rh2m_min": weather["RH2M"].min(),
        "precip_total": weather["PRECTOTCORR"].sum(),
        "wind_max": weather["WS10M"].max(),
        "dryness_streak_max": weather["dryness_streak"].max(),
    }


def build_positive_samples(bbox: tuple, date: str, sensor: str = "VIIRS_SNPP_SP") -> pd.DataFrame:
    """One row per actual fire detection, joined with preceding weather."""
    fires = fetch_fire_hotspots_historical(bbox, date=date, sensor=sensor)
    anchor_date = datetime.strptime(date, "%Y-%m-%d")

    rows = []
    for _, fire in fires.iterrows():
        try:
            feats = _weather_features_for(fire["latitude"], fire["longitude"], anchor_date)
            feats.update({"latitude": fire["latitude"], "longitude": fire["longitude"], "label": 1})
            rows.append(feats)
        except Exception as e:  # noqa: BLE001 — log and skip bad points, don't kill the whole batch
            print(f"skip point ({fire['latitude']}, {fire['longitude']}): {e}")

    return pd.DataFrame(rows)


def build_negative_samples(bbox: tuple, date: str, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Randomly sampled points/date assumed fire-free (no FIRMS hit nearby)."""
    random.seed(seed)
    min_lon, min_lat, max_lon, max_lat = bbox
    anchor_date = datetime.strptime(date, "%Y-%m-%d")

    rows = []
    for _ in range(n_samples):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        try:
            feats = _weather_features_for(lat, lon, anchor_date)
            feats.update({"latitude": lat, "longitude": lon, "label": 0})
            rows.append(feats)
        except Exception as e:  # noqa: BLE001
            print(f"skip point ({lat}, {lon}): {e}")

    return pd.DataFrame(rows)


def build_dataset(bbox: tuple, dates: list[str], negatives_per_date: int = 20) -> pd.DataFrame:
    """
    Full pipeline across multiple historical dates, so the model sees
    fires under a range of seasonal/weather conditions rather than just
    one snapshot.
    """
    frames = []
    for date in dates:
        pos = build_positive_samples(bbox, date)
        neg = build_negative_samples(bbox, date, n_samples=negatives_per_date)
        frames.extend([pos, neg])

    dataset = pd.concat(frames, ignore_index=True).dropna()
    return dataset



# Final case-study regions for the pitch (see README for the sourcing/story
# behind each). Anchor dates are approximate from news reporting — verify
# against the actual earliest FIRMS detection in each bbox/window before
# using them in the final backtest numbers (rule 10: accurate claims only).
CASE_STUDIES = {
    "los_angeles_2025": {
        "bbox": (-119.0, 33.7, -117.5, 34.5),
        "date": "2025-01-07",  # Palisades Fire ignition
    },
    "south_korea_2025": {
        "bbox": (128.3, 36.0, 129.3, 37.0),
        "date": "2025-03-22",  # Gyeongsangbuk-do outbreak
    },
    "patagonia_2025": {
        "bbox": (-72.5, -45.0, -68.0, -40.0),
        "date": "2025-01-20",  # Argentina/Chile wildfire season
    },
}


if __name__ == "__main__":
    all_frames = []
    for name, cfg in CASE_STUDIES.items():
        print(f"Building samples for {name}...")
        df_case = build_dataset(cfg["bbox"], [cfg["date"]], negatives_per_date=15)
        df_case["case_study"] = name
        all_frames.append(df_case)

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv("data/processed/training_data.csv", index=False)
    print(f"Saved {len(df)} rows -> data/processed/training_data.csv")
    print(df["label"].value_counts())
    print(df["case_study"].value_counts())
