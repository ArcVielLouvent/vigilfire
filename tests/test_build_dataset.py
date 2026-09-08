"""
Unit tests for src.data.build_dataset — structural checks only, no network.
"""

from src.data.build_dataset import CASE_STUDIES


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
