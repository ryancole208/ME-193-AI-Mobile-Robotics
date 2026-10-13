import sys
from pathlib import Path

import pytest

MIDTERM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDTERM))


def pytest_addoption(parser):
    parser.addoption("--hardware", action="store_true", default=False,
                     help="also run tests that need the Double Motor / webcam / MQTT broker")


def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: needs real hardware or network (run with --hardware)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--hardware"):
        return
    skip = pytest.mark.skip(reason="hardware test: run with --hardware")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)
