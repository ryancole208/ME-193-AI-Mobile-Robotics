import os
import sys
from pathlib import Path

import pytest

RHYTHM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RHYTHM))     # rhythm/config.py must win over ../config.py
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import config  # noqa: E402  (appends ../ for the reused Midterm modules)

REAL_SONGS = config.SONGS_DIR


@pytest.fixture(autouse=True)
def isolated_files(tmp_path, monkeypatch):
    """Tests never write to the real calibration.json / scores.json / songs, and start from default
    offsets. The game tests READ the real songs folder (the Incompetech charts, e.g. Electrodoodle)."""
    monkeypatch.setattr(config, "SONGS_DIR", REAL_SONGS)
    monkeypatch.setattr(config, "CALIBRATION_PATH", tmp_path / "calibration.json")
    monkeypatch.setattr(config, "SCORES_PATH", tmp_path / "scores.json")
    monkeypatch.setattr(config, "INPUT_OFFSET_MS", {"imu": 0.0, "keyboard": 0.0})
    monkeypatch.setattr(config, "VISUAL_OFFSET_MS", 0.0)
    monkeypatch.setattr(config, "HIT_BEATS", None)          # the charts' own patterns...
    monkeypatch.setattr(config, "BEATS_PER_BAR", 4)         # ...unless a test sets HIT_BEATS
    yield


def pytest_addoption(parser):
    parser.addoption("--hardware", action="store_true", default=False,
                     help="also run tests that need the Double Motor / audio device")


def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: needs real hardware (run with --hardware)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--hardware"):
        return
    skip = pytest.mark.skip(reason="hardware test: run with --hardware")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)
