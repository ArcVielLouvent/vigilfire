"""
Smoke tests — actually call the real NASA FIRMS and NASA POWER APIs.

Skipped by default (see conftest.py). Run explicitly with:
    pytest -m smoke
    pytest --run-smoke

Requires internet access and a real FIRMS_MAP_KEY in your environment/.env.
Will NOT work inside the Claude sandbox (domain not allowlisted) — run
these locally or in CI with the FIRMS_MAP_KEY secret configured.
"""

import os

import pytest
from dotenv import load_dotenv

from src.data.build_dataset import CASE_STUDIES
from src.data.fetch_firms import fetch_fire_hotspots_historical
from src.data.fetch_power import fetch_weather_point

load_dotenv()


@pytest.mark.smoke
def test_power_api_returns_data():
    """Basic reachability + shape check against the real POWER API."""
    df = fetch_weather_point(
        lat=34.05, lon=-118.55,
        start_date="20250101", end_date="20250107",
    )
    assert not df.empty
    assert "T2M_MAX" in df.columns
    assert "RH2M" in df.columns


@pytest.mark.smoke
@pytest.mark.skipif(not os.getenv("FIRMS_MAP_KEY"), reason="FIRMS_MAP_KEY not set")
def test_firms_api_returns_data_for_known_fire():
    """LA case study bbox/date should return at least one hotspot."""
    cfg = CASE_STUDIES["los_angeles_2025"]
    df = fetch_fire_hotspots_historical(cfg["bbox"], date=cfg["date"], day_range=3)
    assert len(df) > 0, "Expected at least one FIRMS detection for a known major fire"
    assert "latitude" in df.columns
    assert "longitude" in df.columns


@pytest.mark.smoke
@pytest.mark.skipif(not os.getenv("FIRMS_MAP_KEY"), reason="FIRMS_MAP_KEY not set")
def test_all_case_study_bboxes_reachable():
    """Every locked-in case study should return a non-error response."""
    for name, cfg in CASE_STUDIES.items():
        df = fetch_fire_hotspots_historical(cfg["bbox"], date=cfg["date"], day_range=3)
        # Not asserting len > 0 here since some may need a wider day_range —
        # this just confirms the API call itself succeeds for each bbox.
        assert df is not None, f"{name}: FIRMS call failed"
