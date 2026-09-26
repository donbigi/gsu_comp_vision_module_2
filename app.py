"""
Web demonstration — serves the precomputed Module 2 pipeline results.

The calibration + measurement pipeline is deterministic but heavy (~0.5 GB
peak, seconds of CPU), so it is NOT run inside the pod. `generate_results.py`
runs it once and writes `static/results.json`; this app only serves that file,
so the pod uses ~30 MB and every request is instant — no OOM, no timeout.

Run directly (python app.py) or via Docker (see README / docker-compose.yml).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "static" / "results.json"


def _load_results() -> dict:
    with open(RESULTS_PATH, "r") as fh:
        return json.load(fh)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/results")
def api_results():
    return jsonify(_load_results())


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
