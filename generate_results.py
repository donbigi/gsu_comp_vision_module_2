"""
Regenerate static/results.json by running the full Module 2 pipeline once.

Run this on a machine with opencv-python-headless, numpy and matplotlib
installed (NOT in the pod — the deployed app only serves the JSON this writes):

    python3 generate_results.py

The pipeline is deterministic, so the committed results.json is the source of
truth for the web demo.
"""

from __future__ import annotations

import json
from pathlib import Path

import measurement

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "static" / "results.json"


def main() -> None:
    results = measurement.run_pipeline()
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
