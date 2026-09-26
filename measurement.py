"""
Camera calibration + real-world 2-D measurement, factored into importable
functions so both the CLI script (calibrate_and_measure.py) and the Flask web
app (app.py) can share the exact same pipeline.

The three steps the module demonstrates:

  1. Calibrate a camera (chessboard) with OpenCV — recover the intrinsic
     matrix K and lens-distortion coefficients.
  2. Measure real-world bar lengths in a single smartphone photo using the
     perspective-projection equation  L = (dx / fx) * Z.
  3. Validate the measurements against 20 ground-truth lengths and report
     error statistics.

The web app calls :func:`run_pipeline`, which returns every number, table and
image (as base64 data URLs) the page needs to render the demonstration.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np

# ======================================================================
# CONFIGURATION (mirrors calibrate_and_measure.py)
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = BASE_DIR / "calibration"
MEASUREMENT_DIR = BASE_DIR / "measurement"
OUTPUT_DIR = BASE_DIR / "output"

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.023  # metres

# The white paper is an 11 x 14 inch sheet photographed in LANDSCAPE
# orientation (14 in = 35.56 cm wide, 11 in = 27.94 cm tall).  Its true size
# is used ONLY as documentation / a distance sanity check, never to set scale.
PAPER_WIDTH_CM = 35.56
PAPER_HEIGHT_CM = 27.94

# Measured camera-to-object distance (metres) — the "Z" in the perspective
# projection equation.  Replace with your physically measured value.
OBJECT_DISTANCE_M = 2.73

# Ground-truth bar lengths (cm), top-to-bottom, per image.
GROUND_TRUTH = {
    "IMG_6192.jpeg": [3, 30, 13, 18, 16, 22, 8, 2, 14, 25],
    "IMG_6193.jpeg": [7, 15, 20, 5, 9, 17, 24, 12, 21, 28],
}

# A bar must be at least this long (in cm) to count.
MIN_BAR_CM = 1.5

# Corner detection is run on the calibration images downscaled to at most this
# many pixels on the long side. The 12 MP originals make the segment-based
# detector allocate ~1 GB; downscaling bounds that, and the detected corners
# are scaled back to full-resolution coordinates so the calibrated intrinsics
# still refer to the original image size. (0.02% effect on the result.)
CALIB_MAX_SIDE = 1800

# Bar detection runs on an upsampled top-down view of the paper so the bars
# are large and horizontal.  This is an arbitrary processing-resolution factor
# and encodes no physical size.
UPSAMPLE = 4.0


# ======================================================================
# HELPERS
# ======================================================================


def _list_images(directory: Path) -> list[Path]:
    """Sorted image paths in a directory (jpeg/jpg/png)."""
    return sorted(
        list(directory.glob("*.jpeg"))
        + list(directory.glob("*.jpg"))
        + list(directory.glob("*.png"))
    )


def _encode_image(img: np.ndarray, max_side: int = 800, quality: int = 82) -> str:
    """Downscale an image for the browser and encode it as a JPEG data URL."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _object_points() -> np.ndarray:
    """3-D chessboard corner positions in the board's own frame (metres)."""
    pts = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    pts[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    pts *= SQUARE_SIZE
    return pts


# ======================================================================
# STEP 1 — CAMERA CALIBRATION
# ======================================================================


def calibrate_camera() -> dict:
    """Find chessboards, calibrate, and return every calibration result.

    Returns a dict with the RMS reprojection error, camera matrix, distortion
    coefficients, per-image errors, and a downscaled corner-detection
    thumbnail for each calibration image.
    """
    object_points = _object_points()
    calibration_images = _list_images(CALIBRATION_DIR)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    per_image = []
    corner_thumbs = []

    image_size = None

    for image_path in calibration_images:
        img = cv2.imread(str(image_path))
        if img is None:
            per_image.append({"name": image_path.name, "ok": False, "error": None})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        # Detect on a downscaled copy (see CALIB_MAX_SIDE) to bound memory.
        h, w = gray.shape
        scale = min(1.0, CALIB_MAX_SIDE / max(h, w))
        if scale < 1.0:
            gray_small = cv2.resize(
                gray, (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            gray_small = gray

        # Plain findChessboardCornersSB (no CALIB_CB_EXHAUSTIVE / ACCURACY
        # flags). Those flags make the search far more CPU- and memory-hungry
        # (~3+ GB peak vs well under 1 GB), which OOM-kills the pod in the
        # cluster; on these clear boards they change the result by <0.02%.
        found, corners = cv2.findChessboardCornersSB(gray_small, CHESSBOARD_SIZE)

        if found:
            obj_points.append(object_points.copy())
            if scale < 1.0:
                corners = (corners.astype(np.float64) / scale).astype(np.float32)
            img_points.append(corners)

            vis = img.copy()
            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, found)
            corner_thumbs.append(_encode_image(vis, max_side=400, quality=80))

            per_image.append({"name": image_path.name, "ok": True, "error": None})
        else:
            per_image.append({"name": image_path.name, "ok": False, "error": None})

    detected = len(obj_points)
    if detected < 5:
        raise RuntimeError("Not enough successful chessboard detections.")

    ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Per-image reprojection error.
    total_error = 0.0
    total_points = 0
    index = 0
    for entry in per_image:
        if not entry["ok"]:
            continue
        projected, _ = cv2.projectPoints(
            obj_points[index], rvecs[index], tvecs[index],
            camera_matrix, distortion,
        )
        actual = img_points[index].reshape(-1, 2).astype(np.float32)
        projected = projected.reshape(-1, 2).astype(np.float32)
        errors = np.linalg.norm(actual - projected, axis=1)
        entry["error"] = round(float(errors.mean()), 4)
        total_error += float(errors.sum())
        total_points += len(errors)
        index += 1

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    # Perspective projection scale (pixels per cm at the object plane).
    px_per_cm = (fx / OBJECT_DISTANCE_M) / 100.0
    rect_px_per_cm = px_per_cm * UPSAMPLE

    return {
        "num_images": len(calibration_images),
        "num_detected": detected,
        "rms": float(ret),
        "camera_matrix": camera_matrix.tolist(),
        "distortion": distortion.ravel().tolist(),
        "per_image": per_image,
        "mean_reprojection": float(total_error / total_points),
        "corner_thumbs": corner_thumbs,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "object_distance_m": OBJECT_DISTANCE_M,
        "pixels_per_cm": float(px_per_cm),
        "rect_pixels_per_cm": float(rect_px_per_cm),
        "camera_matrix_np": camera_matrix,
        "distortion_np": distortion,
    }


# ======================================================================
# PAPER + BAR DETECTION
# ======================================================================


def detect_paper_corners(image: np.ndarray) -> np.ndarray:
    """Locate the white paper as the largest solid 4-corner quadrilateral."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > 150).astype(np.uint8)

    # Close the printed bars so the paper becomes one solid blob.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_corners = None
    for contour in contours:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        if best_contour is None or cv2.contourArea(contour) > cv2.contourArea(best_contour):
            best_contour = contour
            best_corners = approx

    if best_corners is None:
        raise RuntimeError("Could not find the white paper.")

    corners = best_corners.reshape(-1, 2).astype(np.float32)
    tl = corners[np.argmin(corners[:, 0] + corners[:, 1])]
    tr = corners[np.argmax(corners[:, 0] - corners[:, 1])]
    br = corners[np.argmax(corners[:, 0] + corners[:, 1])]
    bl = corners[np.argmin(corners[:, 0] - corners[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rectify_paper(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Straighten the paper to an upsampled top-down view for robust detection."""
    paper_w_px = (
        np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])
    ) / 2.0
    paper_h_px = (
        np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])
    ) / 2.0

    dst_w = int(round(paper_w_px * UPSAMPLE))
    dst_h = int(round(paper_h_px * UPSAMPLE))

    destination = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (dst_w, dst_h))


def detect_bars(paper: np.ndarray, rect_px_per_cm: float):
    """Detect horizontal bars on the rectified paper and measure them in cm."""
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]

    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
    )

    kernel_w = max(3, int(round(0.8 * rect_px_per_cm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_bar_px = MIN_BAR_CM * rect_px_per_cm
    edge_margin = int(0.02 * height)

    segments = []
    for contour in contours:
        x, y, w, _ = cv2.boundingRect(contour)
        if w < min_bar_px:
            continue
        if y < edge_margin or y > height - edge_margin:
            continue
        segments.append((y, x, x + w))

    segments.sort()

    merge_tol = max(3, int(round(0.2 * rect_px_per_cm)))
    merged = []
    for y, x1, x2 in segments:
        for m in merged:
            if abs(y - m["y"]) <= merge_tol:
                m["x1"] = min(m["x1"], x1)
                m["x2"] = max(m["x2"], x2)
                m["y"] = (m["y"] * m["n"] + y) / (m["n"] + 1)
                m["n"] += 1
                break
        else:
            merged.append({"x1": x1, "x2": x2, "y": float(y), "n": 1})

    bars = []
    for m in merged:
        width_px = m["x2"] - m["x1"]
        bars.append(
            {
                "x": m["x1"],
                "y": int(round(m["y"])),
                "width_px": width_px,
                "length_cm": width_px / rect_px_per_cm,
            }
        )
    bars.sort(key=lambda item: item["y"])
    return bars, binary, horizontal


def _draw_paper_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return a copy of `image` with the detected paper corners annotated."""
    vis = image.copy()
    for i, point in enumerate(corners):
        x, y = point.astype(int)
        cv2.circle(vis, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(vis, str(i + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for i in range(4):
        cv2.line(vis, tuple(corners[i].astype(int)), tuple(corners[(i + 1) % 4].astype(int)),
                 (0, 0, 255), 3)
    return vis


def _draw_measurements(paper: np.ndarray, bars: list) -> np.ndarray:
    """Return a copy of the rectified paper with measured bars drawn on it."""
    result = paper.copy()
    for i, bar in enumerate(bars, start=1):
        x1 = bar["x"]
        x2 = x1 + bar["width_px"]
        y = bar["y"]
        cv2.line(result, (x1, y), (x2, y), (0, 0, 255), 5)
        cv2.circle(result, (x1, y), 8, (255, 0, 0), -1)
        cv2.circle(result, (x2, y), 8, (255, 0, 0), -1)
        cv2.putText(result, f"{i}: {bar['length_cm']:.2f} cm", (x1, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return result


# ======================================================================
# STEP 2 + 3 — MEASUREMENT + VALIDATION
# ======================================================================


def _measure_image(image_path: Path, calib: dict) -> dict:
    """Process one measurement image and return its results + images."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read {image_path}")

    camera_matrix = calib["camera_matrix_np"]
    distortion = calib["distortion_np"]
    fx = calib["fx"]
    fy = calib["fy"]
    rect_px_per_cm = calib["rect_pixels_per_cm"]

    undistorted = cv2.undistort(img, camera_matrix, distortion, None, camera_matrix)
    corners = detect_paper_corners(undistorted)

    # Paper-size sanity check via perspective projection.
    paper_w_px = (
        np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])
    ) / 2.0
    paper_h_px = (
        np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])
    ) / 2.0
    paper_w_cm = (paper_w_px / fx) * OBJECT_DISTANCE_M * 100.0
    paper_h_cm = (paper_h_px / fy) * OBJECT_DISTANCE_M * 100.0

    paper = rectify_paper(undistorted, corners)
    bars, binary, horizontal = detect_bars(paper, rect_px_per_cm)

    truth = GROUND_TRUTH.get(image_path.name)
    rows = []
    if truth is not None:
        for i, (bar, expected) in enumerate(zip(bars, truth), start=1):
            measured = bar["length_cm"]
            error = measured - expected
            rows.append(
                {
                    "index": i,
                    "measured_cm": round(float(measured), 3),
                    "truth_cm": expected,
                    "error_cm": round(float(error), 3),
                    "pct_error": round(100.0 * error / expected, 2),
                }
            )

    return {
        "image": image_path.name,
        "paper_w_cm": round(float(paper_w_cm), 2),
        "paper_h_cm": round(float(paper_h_cm), 2),
        "paper_truth_cm": [PAPER_WIDTH_CM, PAPER_HEIGHT_CM],
        "detected_bars": len(bars),
        "expected_bars": len(truth) if truth is not None else None,
        "undistorted": _encode_image(undistorted, max_side=720),
        "paper_corners": _encode_image(_draw_paper_corners(undistorted, corners), max_side=720),
        "binary": _encode_image(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), max_side=720),
        "horizontal": _encode_image(cv2.cvtColor(horizontal, cv2.COLOR_GRAY2BGR), max_side=720),
        "result": _encode_image(_draw_measurements(paper, bars), max_side=720),
        "bars": rows,
    }


# ======================================================================
# AGGREGATE STATISTICS
# ======================================================================


def _error_plot(all_measured_cm, all_truth_cm, all_errors_cm) -> str:
    """Measured-vs-truth scatter + error histogram, as a PNG data URL."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = np.array(all_measured_cm)
    truths = np.array(all_truth_cm)
    errs = np.array(all_errors_cm)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(truths, measured, color="#1f77b4", s=40, zorder=3)
    lim = [0, max(truths.max(), measured.max()) * 1.05]
    ax.plot(lim, lim, "--", color="#888888", lw=1.5, label="ideal (measured = truth)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Ground truth (cm)")
    ax.set_ylabel("Measured (cm)")
    ax.set_title("Measured vs ground truth")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.hist(errs, bins=15, color="#1f77b4", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#888888", lw=1.5, linestyle="--")
    ax.axvline(errs.mean(), color="#d62728", lw=1.5, label=f"mean = {errs.mean():+.2f} cm")
    ax.set_xlabel("Error (cm)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    ax.legend()

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ======================================================================
# TOP-LEVEL PIPELINE
# ======================================================================


def run_pipeline() -> dict:
    """Run the whole pipeline and return a JSON-serialisable result dict."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    calib = calibrate_camera()

    measurement_images = _list_images(MEASUREMENT_DIR)
    measurements = []
    all_measured_cm = []
    all_truth_cm = []
    all_errors_cm = []

    for image_path in measurement_images:
        result = _measure_image(image_path, calib)
        measurements.append(result)
        for row in result["bars"]:
            all_measured_cm.append(row["measured_cm"])
            all_truth_cm.append(row["truth_cm"])
            all_errors_cm.append(row["error_cm"])

    errors = np.array(all_errors_cm)
    truths = np.array(all_truth_cm)
    abs_errors = np.abs(errors)
    rel_errors = 100.0 * abs_errors / truths

    stats = {
        "n": len(errors),
        "mean_signed": round(float(errors.mean()), 3),
        "mean_abs": round(float(abs_errors.mean()), 3),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 3),
        "std": round(float(errors.std()), 3),
        "min_abs": round(float(abs_errors.min()), 3),
        "max_abs": round(float(abs_errors.max()), 3),
        "mean_abs_pct": round(float(rel_errors.mean()), 2),
        "max_abs_pct": round(float(rel_errors.max()), 2),
    }

    return {
        "calibration": {
            "num_images": calib["num_images"],
            "num_detected": calib["num_detected"],
            "rms": calib["rms"],
            "mean_reprojection": calib["mean_reprojection"],
            "camera_matrix": calib["camera_matrix"],
            "distortion": calib["distortion"],
            "per_image": calib["per_image"],
            "corner_thumbs": calib["corner_thumbs"],
        },
        "projection": {
            "fx": calib["fx"],
            "fy": calib["fy"],
            "cx": calib["cx"],
            "cy": calib["cy"],
            "object_distance_m": calib["object_distance_m"],
            "pixels_per_cm": calib["pixels_per_cm"],
        },
        "measurements": measurements,
        "stats": stats,
        "error_plot": _error_plot(all_measured_cm, all_truth_cm, all_errors_cm),
    }
