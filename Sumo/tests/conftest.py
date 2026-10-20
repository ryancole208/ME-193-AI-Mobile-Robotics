"""Make the Sumo/ modules importable from the tests (they're plain modules, not a package)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
