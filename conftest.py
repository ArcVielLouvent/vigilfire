"""
conftest.py

Adds a `smoke` marker for tests that hit real external APIs (NASA FIRMS /
NASA POWER). Smoke tests are skipped by default — run them explicitly with
`pytest -m smoke` (locally, with FIRMS_MAP_KEY set) or `--run-smoke`.

This keeps the default `pytest` invocation fast, offline, and safe to run
on every commit in CI, while still giving you a real end-to-end check
whenever you want one.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-smoke",
        action="store_true",
        default=False,
        help="Run smoke tests that call the real NASA FIRMS/POWER APIs (needs internet + FIRMS_MAP_KEY).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: requires internet access and a real FIRMS_MAP_KEY")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-smoke"):
        return  # --run-smoke was passed, don't skip anything
    if "smoke" in (config.getoption("-m") or ""):
        return  # user explicitly selected smoke tests via -m, respect that
    skip_smoke = pytest.mark.skip(reason="need --run-smoke option (or `pytest -m smoke`) to run")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip_smoke)
