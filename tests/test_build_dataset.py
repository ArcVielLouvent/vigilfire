"""
Unit tests for src.data.build_dataset — structural checks only, no network.
"""

from src.data.build_dataset import CASE_STUDIES, _is_far_enough


def test_case_studies_have_required_keys():
    for name, cfg in CASE_STUDIES.items():
        assert "bbox" in cfg, f"{name} missing bbox"
        assert "date" in cfg, f"{name} missing date"


def test_case_studies_bbox_is_valid():
    for name, cfg in CASE_STUDIES.items():
        min_lon, min_lat, max_lon, max_lat = cfg["bbox"]
        assert min_lon < max_lon, f"{name}: min_lon must be < max_lon"
        assert min_lat < max_lat, f"{name}: min_lat must be < max_lat"
        assert -180 <= min_lon <= 180 and -180 <= max_lon <= 180, f"{name}: longitude out of range"
        assert -90 <= min_lat <= 90 and -90 <= max_lat <= 90, f"{name}: latitude out of range"


def test_case_studies_covers_three_continents():
    # Guards against accidentally collapsing back to a single-region demo.
    assert len(CASE_STUDIES) >= 3


def test_is_far_enough_rejects_point_too_close_to_fire():
    known_fires = [(34.05, -118.55)]
    # a point just 0.1 degrees away is well within a 0.3 degree exclusion buffer
    assert _is_far_enough(34.10, -118.55, known_fires, min_distance_deg=0.3) is False


def test_is_far_enough_accepts_point_far_from_fire():
    known_fires = [(34.05, -118.55)]
    # a point 5 degrees away is clearly outside any reasonable exclusion buffer
    assert _is_far_enough(39.05, -118.55, known_fires, min_distance_deg=0.3) is True


def test_is_far_enough_with_no_known_fires_always_accepts():
    assert _is_far_enough(0.0, 0.0, [], min_distance_deg=0.3) is True


def test_is_far_enough_checks_all_fires_not_just_first():
    known_fires = [(0.0, 0.0), (34.05, -118.55)]
    # far from the first fire, but too close to the second
    assert _is_far_enough(34.10, -118.55, known_fires, min_distance_deg=0.3) is False

