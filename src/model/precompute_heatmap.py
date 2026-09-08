"""
precompute_heatmap.py

Builds a small risk-score grid (default 5x5 = 25 points) for each locked
case-study region, using the weather conditions in the week *before* that
event's ignition date. This directly demonstrates the model's core claim:
"the area that actually burned was already flagged high-risk beforehand."

Deliberately precomputed and saved to CSV rather than queried live —
scanning a full grid live would mean 25+ NASA POWER calls per view, which
is too slow for a smooth demo (see the "scope overflow" discussion: a
live, arbitrary-region, high-resolution heatmap was intentionally cut to
keep this shippable in the hackathon window). The Streamlit app just reads
the precomputed CSV.

Run this locally once you have a trained fire_risk_model.joblib:
    python -m src.model.precompute_heatmap
"""

import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from src.data.build_dataset import CASE_STUDIES
from src.data.fetch_power import compute_dryness_streak, fetch_weather_point, has_sufficient_data
from src.model.train_model import FEATURE_COLUMNS

GRID_SIZE = 5  # 5x5 = 25 points per region — enough to show a spatial pattern
               # without racking up excessive API calls or runtime.


def _features_for_grid_point(lat: float, lon: float, anchor_date: datetime) -> dict | None:
    start = (anchor_date - timedelta(days=7)).strftime("%Y%m%d")
    end = (anchor_date - timedelta(days=1)).strftime("%Y%m%d")

    weather = fetch_weather_point(lat, lon, start, end)
    if not has_sufficient_data(weather):
        return None

    weather["dryness_streak"] = compute_dryness_streak(weather)
    return {
        "t2m_max_avg": weather["T2M_MAX"].mean(),
        "rh2m_min": weather["RH2M"].min(),
        "precip_total": weather["PRECTOTCORR"].sum(),
        "wind_max": weather["WS10M"].max(),
        "dryness_streak_max": weather["dryness_streak"].max(),
    }


def build_heatmap_for_case_study(name: str, cfg: dict, clf, grid_size: int = GRID_SIZE) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = cfg["bbox"]
    anchor_date = datetime.strptime(cfg["date"], "%Y-%m-%d")

    lats = np.linspace(min_lat, max_lat, grid_size)
    lons = np.linspace(min_lon, max_lon, grid_size)

    rows = []
    total = grid_size * grid_size
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            point_num = i * grid_size + j + 1
            print(f"[{name}] point {point_num}/{total}: ({lat:.3f}, {lon:.3f})")

            feats = _features_for_grid_point(lat, lon, anchor_date)
            if feats is None:
                print(f"  skipped: insufficient POWER data for this point/date")
                continue

            X = pd.DataFrame([feats])[FEATURE_COLUMNS]
            risk = clf.predict_proba(X)[0, 1]
            rows.append({"latitude": lat, "longitude": lon, "risk_score": risk})

            time.sleep(0.2)  # light throttle — be a polite API citizen

    df = pd.DataFrame(rows)
    df["case_study"] = name
    return df


if __name__ == "__main__":
    clf = joblib.load("fire_risk_model.joblib")

    all_grids = []
    for name, cfg in CASE_STUDIES.items():
        print(f"\n=== Building heatmap grid for {name} ===")
        grid_df = build_heatmap_for_case_study(name, cfg, clf)
        all_grids.append(grid_df)
        grid_df.to_csv(f"data/processed/heatmap_{name}.csv", index=False)
        print(f"Saved data/processed/heatmap_{name}.csv ({len(grid_df)} points)")

    combined = pd.concat(all_grids, ignore_index=True)
    combined.to_csv("data/processed/heatmap_all.csv", index=False)
    print(f"\nSaved combined heatmap -> data/processed/heatmap_all.csv ({len(combined)} points total)")
