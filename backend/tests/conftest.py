import os
import sys
from pathlib import Path

# Pin deterministic defaults before any app imports during collection.
os.environ["HARDWARE_PROFILE"] = "cpu_only"
os.environ["GPU_BACKEND"] = "cpu"

# Ensure app package is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
