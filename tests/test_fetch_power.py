"""
Unit tests for src.data.fetch_power — pure logic, no network required.
These run on every commit / every CI build.
"""

import pandas as pd
import pytest

from src.data.fetch_power import compute_dryness_streak


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
