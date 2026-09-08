"""
backtest.py

Answers the "measurable environmental impact" requirement directly:
for each historical fire in the test set, how many days in advance would
this model have flagged elevated risk, based purely on the weather trend
leading up to it?

This produces the headline number for the pitch/README, e.g.:
"On historical fires in [region], the model flagged elevated risk an
average of N days before the fire was actually detected by satellite."
"""

import joblib
import pandas as pd

from src.data.fetch_power import compute_dryness_streak, fetch_weather_point
from src.model.train_model import FEATURE_COLUMNS


def lead_time_for_fire(clf, lat: float, lon: float, fire_date: str, window_days: int = 14, risk_threshold: float = 0.5) -> int | None:
    """
    Walk backward day-by-day from fire_date, recomputing the 7-day
    antecedent weather features each time, and find the earliest day the
    model's predicted risk probability crosses risk_threshold.

    Returns:
        Number of days of advance warning, or None if the model never
        crossed the threshold within `window_days`.
    """
    from datetime import datetime, timedelta

    fire_dt = datetime.strptime(fire_date, "%Y-%m-%d")
    start_window = fire_dt - timedelta(days=window_days + 7)
    end_window = fire_dt - timedelta(days=1)

    weather = fetch_weather_point(
        lat, lon,
        start_window.strftime("%Y%m%d"),
        end_window.strftime("%Y%m%d"),
    )
    weather["dryness_streak"] = compute_dryness_streak(weather)

    for days_before in range(window_days, 0, -1):
        check_date = fire_dt - timedelta(days=days_before)
        window_start = check_date - timedelta(days=7)
        w = weather.loc[window_start:check_date - timedelta(days=1)]
        if w.empty:
            continue

        feats = pd.DataFrame([{
            "t2m_max_avg": w["T2M_MAX"].mean(),
            "rh2m_min": w["RH2M"].min(),
            "precip_total": w["PRECTOTCORR"].sum(),
            "wind_max": w["WS10M"].max(),
            "dryness_streak_max": w["dryness_streak"].max(),
        }])[FEATURE_COLUMNS]

        proba = clf.predict_proba(feats)[0, 1]
        if proba >= risk_threshold:
            return days_before

    return None


if __name__ == "__main__":
    clf = joblib.load("fire_risk_model.joblib")

    # Replace with real historical fire events pulled from FIRMS for your
    # chosen case-study regions (match the case studies used in the pitch).
    test_fires = [
        {"lat": -2.1, "lon": 113.5, "date": "2026-08-20"},
    ]

    lead_times = []
    for fire in test_fires:
        lt = lead_time_for_fire(clf, fire["lat"], fire["lon"], fire["date"])
        print(f"Fire at ({fire['lat']}, {fire['lon']}) on {fire['date']}: "
              f"{lt if lt is not None else 'no warning'} days advance warning")
        if lt is not None:
            lead_times.append(lt)

    if lead_times:
        print(f"\nAverage lead time: {sum(lead_times) / len(lead_times):.1f} days")
