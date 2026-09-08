"""
Unit tests for src.data.fetch_power — pure logic, no network required.
These run on every commit / every CI build.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.fetch_power import compute_dryness_streak, has_sufficient_data


def test_dryness_streak_basic_pattern():
    df = pd.DataFrame({"PRECTOTCORR": [5, 3, 8, 0, 0, 0.5, 0, 0, 6, 0.2]})
    result = compute_dryness_streak(df).tolist()
    assert result == [0, 0, 0, 1, 2, 3, 4, 5, 0, 1]


def test_dryness_streak_dry_from_day_one():
    df = pd.DataFrame({"PRECTOTCORR": [0, 0, 0]})
    result = compute_dryness_streak(df).tolist()
    assert result == [1, 2, 3]


def test_dryness_streak_all_wet():
    df = pd.DataFrame({"PRECTOTCORR": [5, 5, 5]})
    result = compute_dryness_streak(df).tolist()
    assert result == [0, 0, 0]


def test_dryness_streak_single_day_gaps():
    """A single dry day between wet days should never inflate to 2+."""
    df = pd.DataFrame({"PRECTOTCORR": [5, 0, 5, 0, 5]})
    result = compute_dryness_streak(df).tolist()
    assert result == [0, 1, 0, 1, 0]


def test_dryness_streak_respects_custom_threshold():
    df = pd.DataFrame({"PRECTOTCORR": [2, 2, 2]})
    # with default threshold (1.0mm) these all count as "wet" (not dry)
    assert compute_dryness_streak(df).tolist() == [0, 0, 0]
    # with a higher threshold, the same values become "dry"
    assert compute_dryness_streak(df, threshold_mm=3.0).tolist() == [1, 2, 3]


def test_dryness_streak_empty_dataframe():
    df = pd.DataFrame({"PRECTOTCORR": []})
    result = compute_dryness_streak(df)
    assert len(result) == 0


@pytest.mark.parametrize("precip_value", [-1, 0, 0.999])
def test_dryness_streak_edge_values_near_threshold(precip_value):
    """Values right at/below the 1.0mm default threshold should count as dry."""
    df = pd.DataFrame({"PRECTOTCORR": [precip_value]})
    assert compute_dryness_streak(df).tolist() == [1]


def test_has_sufficient_data_rejects_all_nan():
    """Guards against the real bug: NASA POWER's -999 fill value, once
    converted to NaN, must not be silently treated as usable data."""
    df = pd.DataFrame({
        "T2M_MAX": [np.nan] * 5,
        "RH2M": [np.nan] * 5,
        "PRECTOTCORR": [np.nan] * 5,
        "WS10M": [np.nan] * 5,
    })
    assert has_sufficient_data(df) is False


def test_has_sufficient_data_accepts_clean_data():
    df = pd.DataFrame({
        "T2M_MAX": [30.0] * 5,
        "RH2M": [40.0] * 5,
        "PRECTOTCORR": [0.0] * 5,
        "WS10M": [5.0] * 5,
    })
    assert has_sufficient_data(df) is True


def test_has_sufficient_data_accepts_minor_gaps():
    df = pd.DataFrame({"T2M_MAX": [30.0, 31.0, np.nan, 29.0, 30.0]})
    assert has_sufficient_data(df) is True


def test_has_sufficient_data_rejects_empty_dataframe():
    assert has_sufficient_data(pd.DataFrame()) is False
