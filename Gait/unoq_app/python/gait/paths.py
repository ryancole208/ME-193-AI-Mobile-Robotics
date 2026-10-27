"""Where every file lives. Everything is relative to the app folder.

Inside App Lab the app folder is mounted at /app; on a laptop or in the dev
container it is wherever unoq_app/ is. GAIT_APP_DIR overrides it (tests use a
temporary folder).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def default_app_dir() -> Path:
    env = os.environ.get("GAIT_APP_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]  # python/gait/paths.py -> app folder


@dataclass
class Paths:
    root: Path

    @classmethod
    def default(cls) -> "Paths":
        return cls(default_app_dir())

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def setup_yaml(self) -> Path:
        return self.root / "setup.yaml"

    @property
    def state(self) -> Path:
        return self.root / "state.json"   # last shown model (survives reboot)

    @property
    def run_mode(self) -> Path:
        return self.root / "RUN_MODE"     # one-shot mode switch: diag / setup / holdout / replay <csv>

    @property
    def setup_data(self) -> Path:
        return self.root / "data" / "setup"

    @property
    def holdout_data(self) -> Path:
        return self.root / "data" / "holdout"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> None:
        for d in (self.setup_data, self.holdout_data, self.logs, self.reports):
            d.mkdir(parents=True, exist_ok=True)
