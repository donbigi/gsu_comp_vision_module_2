"""
Web demonstration — camera calibration and real-world 2-D measurement.

Serves an interactive page that runs the whole Module 2 pipeline and shows:

  - camera calibration (chessboard) — RMS reprojection error, intrinsic
    matrix K, distortion coefficients, and per-image corner detection;
  - measurement — the undistorted photo, detected paper boundary, the
    bar-detection masks, and the measured bars drawn on a rectified view;
  - validation — a measured-vs-truth table and aggregate error statistics,
    all derived from the perspective-projection equation  L = (dx/fx)·Z.

Run directly (python app.py) or via Docker (see README / docker-compose.yml).
"""

from __future__ import annotations

import os
import threading

from flask import Flask, jsonify, render_template, request

import measurement

app = Flask(__name__)

# The pipeline is compute-heavy (chessboard detection over ~20 images), so we
# run it once and cache the result; the page can force a fresh run on demand.
_cache: dict | None = None
_lock = threading.Lock()


def _get_results(force: bool = False) -> dict:
    global _cache
    with _lock:
        if _cache is None or force:
            _cache = measurement.run_pipeline()
        return _cache


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(silent=True) or {}
    force = bool(data.get("force"))
    try:
        return jsonify(_get_results(force=force))
    except Exception as exc:  # surface a clean error to the client
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    # Precompute the (deterministic) pipeline once before serving, so the
    # first request — and every request after it — is instant. This keeps the
    # heavy compute off the request path, which otherwise exceeds the pod's
    # memory limit and the Envoy timeout in the cluster.
    _get_results()

    app.run(host=host, port=port, debug=False)
