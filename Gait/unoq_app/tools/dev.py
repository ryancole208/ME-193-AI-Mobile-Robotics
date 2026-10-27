#!/usr/bin/env python3
"""One entry point for the developer tools. Standard library only.

The board's own Python has no numpy, so on the UNO Q every tool runs inside
the same container image App Lab uses for the app (a throwaway `docker run
--rm`; nothing is installed on the board). On a laptop that already has
numpy, the tools run directly.

  python3 tools/dev.py preview              render every matrix frame (terminal + reports/matrix_preview.html)
  python3 tools/dev.py replay FILE.csv      replay a recording through the live pipeline (prints display commands)
  python3 tools/dev.py evaluate [FILES]     evaluate config.json on held-out recordings -> reports/evaluate.md

  python3 tools/dev.py diag                 write RUN_MODE=diag and restart the app (asks first), then follow logs
  python3 tools/dev.py mode setup|holdout|"replay data/x.csv"   same, for other one-shot modes
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/arduino/app-bricks/python-apps-base:0.12.0"
SCRIPTS = {
    "preview": ["tools/matrix_preview.py"],
    "replay": ["tools/replay.py"],
    "evaluate": ["tools/evaluate.py"],
}


def have_local_numpy() -> bool:
    try:
        import numpy  # noqa: F401
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def run_python(args: list[str]) -> int:
    if have_local_numpy() and not shutil.which("docker"):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(APP / "python"), str(APP / "bricks"), str(APP / "tools")]))
        return subprocess.call([sys.executable, *args], cwd=APP, env=env)
    if not shutil.which("docker"):
        sys.exit("While starting the tools: neither numpy nor docker is available. Likely fix: run this on the "
                 "UNO Q (it has docker), or `pip install numpy pyyaml` (+ matplotlib for plots) on your laptop")
    cmd = ["docker", "run", "--rm", "-i", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "-e", f"PYTHONPATH=/app/python:/app/bricks:/app/tools",
           "-e", "MPLCONFIGDIR=/tmp", "-v", f"{APP}:/app", "-w", "/app", "--entrypoint", "python", IMAGE, *args]
    return subprocess.call(cmd)


def one_shot(mode: str) -> int:
    print(f"This writes RUN_MODE='{mode}' and restarts the app (any running App is stopped).")
    if input("Continue? [y/N] ").strip().lower() != "y":
        return 1
    (APP / "RUN_MODE").write_text(mode + "\n")
    rc = subprocess.call(["arduino-app-cli", "app", "restart", str(APP)])
    if rc == 0:
        print("Following the app log (Ctrl+C to stop following; the app keeps running).")
        subprocess.call(["arduino-app-cli", "app", "logs", str(APP), "--follow"])
    return rc


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "diag":
        return one_shot("diag")
    if cmd == "mode":
        return one_shot(" ".join(rest))
    if cmd not in SCRIPTS:
        print(__doc__)
        return 2
    return run_python(SCRIPTS[cmd] + rest)


if __name__ == "__main__":
    sys.exit(main())
