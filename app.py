"""
app.py — Streamlit "scan area" demo.

Run locally with: streamlit run app.py

The user picks a region (preset or custom bounding box), the app pulls the
last 7 days of weather for the centroid from NASA POWER, runs it through
the trained risk model, and displays a risk verdict — this is the core
demo flow for the pitch video.
"""

from datetime import datetime, timedelta

import joblib
import pandas as pd
import streamlit as st

from src.data.fetch_power import compute_dryness_streak, fetch_weather_point
from src.model.train_model import FEATURE_COLUMNS

PRESET_REGIONS = {
    "Los Angeles, USA (Jan 2025 case study)": (34.05, -118.55),
    "Gyeongsangbuk-do, South Korea (Mar 2025 case study)": (36.5, 128.8),
    "Patagonia, Argentina/Chile (Jan 2025 case study)": (-42.5, -70.0),
    "Custom coordinates": None,
}


@st.cache_resource
def load_model():
    return joblib.load("fire_risk_model.joblib")


def get_features_for_point(lat: float, lon: float) -> pd.DataFrame:
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    weather = fetch_weather_point(lat, lon, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    weather["dryness_streak"] = compute_dryness_streak(weather)

    return pd.DataFrame([{
        "t2m_max_avg": weather["T2M_MAX"].mean(),
        "rh2m_min": weather["RH2M"].min(),
        "precip_total": weather["PRECTOTCORR"].sum(),
        "wind_max": weather["WS10M"].max(),
        "dryness_streak_max": weather["dryness_streak"].max(),
    }])[FEATURE_COLUMNS]


def main():
    st.set_page_config(page_title="Vigilfire — Wildfire Risk Scanner", page_icon="🔥")
    st.title("🔥 Vigilfire")
    st.caption("Scan any region for near-term wildfire risk, based on satellite-trained weather patterns.")

    choice = st.selectbox("Choose a region to scan", list(PRESET_REGIONS.keys()))

    if choice == "Custom coordinates":
        lat = st.number_input("Latitude", value=0.0, format="%.4f")
        lon = st.number_input("Longitude", value=0.0, format="%.4f")
    else:
        lat, lon = PRESET_REGIONS[choice]
        st.write(f"Coordinates: {lat}, {lon}")

    if st.button("Scan area"):
        with st.spinner("Pulling last 7 days of weather and scoring risk..."):
            clf = load_model()
            features = get_features_for_point(lat, lon)
            proba = clf.predict_proba(features)[0, 1]

        st.metric("Wildfire risk score", f"{proba:.0%}")

        if proba >= 0.7:
            st.error("High risk — dry, hot, windy conditions detected over the past week.")
        elif proba >= 0.4:
            st.warning("Moderate risk — conditions trending drier than usual.")
        else:
            st.success("Low risk — recent weather does not indicate elevated fire danger.")

        with st.expander("See underlying weather features"):
            st.dataframe(features)

        with st.expander("Why did the model say this? (feature importance)"):
            importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS)
            st.bar_chart(importances.sort_values(ascending=False))
            st.caption(
                "Higher bars = the model relied on that feature more heavily "
                "across its training data, not just for this one scan."
            )


if __name__ == "__main__":
    main()
