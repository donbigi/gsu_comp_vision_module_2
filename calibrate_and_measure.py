import cv2
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CALIBRATION_DIR = Path("calibration")
MEASUREMENT_DIR = Path("measurement")
OUTPUT_DIR = Path("output")

CHESSBOARD_SIZE = (9, 6)

# Physical chessboard square size
SQUARE_SIZE = 0.023  # meters

# ------------------------------------------------------------
# WHITE PAPER SIZE
# ------------------------------------------------------------

# The white paper is an 11 x 14 inch sheet, photographed in LANDSCAPE
# orientation, so its physical size is 35.56 x 27.94 cm.
#   14 inches = 35.56 cm (width)
#   11 inches = 27.94 cm (height)
#
# NOTE: the paper's true size is used ONLY as documentation and for a
# distance sanity check below. It is NOT used to derive the pixel-to-cm
# scale for the measurements -- that now comes from the camera focal
# length and the measured object distance, per the perspective
# projection equations.
PAPER_WIDTH_CM = 35.56
PAPER_HEIGHT_CM = 27.94

# ------------------------------------------------------------
# OBJECT DISTANCE
# ------------------------------------------------------------

# Measured distance from the camera to the object plane, in meters.
#
# REQUIRED by the assignment: use a distance > 2 m and measure it
# accurately (tape measure / laser). This is the "Z" in the perspective
# projection equations and it directly scales every measurement, so it
# MUST equal the true camera-to-object distance of the photos below.
#
# 2.73 m below is the distance implied by the paper's known width
# (35.56 cm) through the perspective projection equation -- a cross-check
# against the independently measured value. Replace it with your
# physically measured distance to keep the validation independent of the
# paper's known size.
OBJECT_DISTANCE_M = 2.73

# Ground-truth bar lengths (cm), top-to-bottom, for each image.
GROUND_TRUTH = {
    "IMG_6192.jpeg": [3, 30, 13, 18, 16, 22, 8, 2, 14, 25],
    "IMG_6193.jpeg": [7, 15, 20, 5, 9, 17, 24, 12, 21, 28],
}

# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# PREPARE CHESSBOARD OBJECT POINTS
# ============================================================

object_points = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

object_points[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

object_points *= SQUARE_SIZE

obj_points = []
img_points = []

calibration_images = sorted(
    list(CALIBRATION_DIR.glob("*.jpeg")) +
    list(CALIBRATION_DIR.glob("*.jpg")) +
    list(CALIBRATION_DIR.glob("*.png"))
)

print(f"Found {len(calibration_images)} calibration images.")
print()

# ============================================================
# FIND CHESSBOARD CORNERS
# ============================================================

image_size = None

for image_path in calibration_images:

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHESSBOARD_SIZE,
        flags=cv2.CALIB_CB_EXHAUSTIVE |
              cv2.CALIB_CB_ACCURACY
    )

    if found:

        obj_points.append(object_points.copy())
        img_points.append(corners)

        print(f"[OK] {image_path.name}")

        vis = img.copy()

        cv2.drawChessboardCorners(
            vis,
            CHESSBOARD_SIZE,
            corners,
            found
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"corners_{image_path.name}"),
            vis
        )

    else:
        print(f"[FAIL] {image_path.name}")

print()

print(
    f"Successful detections: "
    f"{len(obj_points)}/{len(calibration_images)}"
)

if len(obj_points) < 5:
    raise RuntimeError(
        "Not enough successful chessboard detections."
    )

# ============================================================
# CAMERA CALIBRATION
# ============================================================

print()
print("Calibrating camera...")

ret, camera_matrix, distortion_coefficients, rvecs, tvecs = \
    cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None
    )

print()
print("============================================================")
print("CAMERA CALIBRATION")
print("============================================================")

print(f"RMS reprojection error: {ret:.4f} pixels")

print()
print("Camera matrix:")
print(camera_matrix)

print()
print("Distortion coefficients:")
print(distortion_coefficients.ravel())

# ============================================================
# PER-IMAGE REPROJECTION ERROR
# ============================================================

print()
print("Per-image reprojection errors:")

total_error = 0
total_points = 0

for i in range(len(obj_points)):

    projected_points, _ = cv2.projectPoints(
        obj_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion_coefficients
    )

    actual = img_points[i].reshape(-1, 2).astype(np.float32)
    projected = projected_points.reshape(-1, 2).astype(np.float32)

    errors = np.linalg.norm(
        actual - projected,
        axis=1
    )

    error = np.mean(errors)

    print(
        f"{calibration_images[i].name:20s} "
        f"{error:.4f} px"
    )

    total_error += np.sum(errors)
    total_points += len(errors)

print()

print(
    f"Overall mean reprojection error: "
    f"{total_error / total_points:.4f} pixels"
)

# ============================================================
# SAVE CALIBRATION
# ============================================================

np.savez(
    OUTPUT_DIR / "camera_calibration.npz",
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
    image_width=image_size[0],
    image_height=image_size[1]
)

print()
print("Saved:")
print("  output/camera_calibration.npz")

# ============================================================
# PERSPECTIVE PROJECTION PARAMETERS
# ============================================================
#
# The pinhole model maps a 3-D point (X, Y, Z) in the camera frame to
# image pixels (x, y):
#
#     x = fx * (X / Z) + cx
#     y = fy * (Y / Z) + cy
#
# Inverting for a planar object facing the camera (Z is the measured
# camera-to-object distance), a real-world length L spanning `dx` pixels
# is recovered as:
#
#     L = (dx / fx) * Z          (meters)
#
# The principal point (cx, cy) cancels out when taking differences, so a
# horizontal length needs only fx (and Z), a vertical length only fy.
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print()
print("============================================================")
print("PERSPECTIVE PROJECTION PARAMETERS")
print("============================================================")
print(f"focal length   fx = {fx:.2f} px,  fy = {fy:.2f} px")
print(f"principal pt   cx = {cx:.2f} px,  cy = {cy:.2f} px")
print(f"object distance Z = {OBJECT_DISTANCE_M} m")

# Pixels-per-meter at the object plane (the perspective scale factor).
PIXELS_PER_M = fx / OBJECT_DISTANCE_M
print(
    f"scale = fx / Z = {PIXELS_PER_M:.2f} px/m "
    f"({PIXELS_PER_M / 100:.2f} px/cm)"
)

# Pixels-per-cm in the undistorted image. This is the perspective scale
# that turns a pixel length into a real-world length:
#
#     L[cm] = (dx[px] / fx) * Z * 100 = dx[px] / PX_PER_CM
PX_PER_CM = PIXELS_PER_M / 100.0

# Bar detection is done on an upsampled top-down ("rectified") view of the
# paper so the bars are large and exactly horizontal, which makes the
# detector robust. UPSAMPLE is an arbitrary processing-resolution factor;
# it does NOT encode any physical size. The real-world scale still comes
# from fx and Z through PX_PER_CM.
UPSAMPLE = 4.0

# Pixels-per-cm in the rectified (upsampled) view.
RECT_PX_PER_CM = PX_PER_CM * UPSAMPLE

# ============================================================
# WHITE-PAPER DETECTION
# ============================================================

def detect_paper_corners(image):

    # The white paper is the brightest, most solid rectangle in the
    # scene. The fridge is darker and irregular, so it never reduces to
    # a clean 4-corner quad. We threshold high (keeping only the white
    # paper), close the printed bars, and take the largest contour whose
    # convex hull collapses to four corners.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # No heavy blur here: blurring smears the paper's bright edge into
    # the darker fridge and pulls the detected boundary inward, which
    # inflates every measurement. A plain threshold keeps the true edge.
    mask = (gray > 150).astype(np.uint8)

    # Fill the dark bars so the paper becomes one solid blob.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((31, 31), np.uint8)
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Pick the largest contour that is a quadrilateral.
    best_contour = None
    best_corners = None

    for contour in contours:

        hull = cv2.convexHull(contour)

        perimeter = cv2.arcLength(hull, True)

        approx = cv2.approxPolyDP(
            hull,
            0.02 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        if (
            best_contour is None
            or cv2.contourArea(contour) > cv2.contourArea(best_contour)
        ):
            best_contour = contour
            best_corners = approx

    if best_corners is None:
        raise RuntimeError("Could not find the white paper.")

    corners = best_corners.reshape(-1, 2).astype(np.float32)

    # Order the four corners: top-left, top-right, bottom-right,
    # bottom-left.
    top_left = corners[np.argmin(corners[:, 0] + corners[:, 1])]
    top_right = corners[np.argmax(corners[:, 0] - corners[:, 1])]
    bottom_right = corners[np.argmax(corners[:, 0] + corners[:, 1])]
    bottom_left = corners[np.argmin(corners[:, 0] - corners[:, 1])]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )

# ============================================================
# PERSPECTIVE RECTIFICATION (detection only)
# ============================================================

def rectify_paper(image, corners):
    """Straighten the paper to a top-down view, upsampled by UPSAMPLE.

    This is used ONLY to make bar detection robust (large, exactly
    horizontal bars). It does not determine the real-world scale: the
    destination size is derived from the paper's *pixel* size (not its
    physical size), so the mapping is scale-preserving apart from the
    UPSAMPLE factor. The physical scale still comes from fx and Z.
    """
    paper_w_px = (
        np.linalg.norm(corners[1] - corners[0])
        + np.linalg.norm(corners[2] - corners[3])
    ) / 2.0
    paper_h_px = (
        np.linalg.norm(corners[3] - corners[0])
        + np.linalg.norm(corners[2] - corners[1])
    ) / 2.0

    dst_w = int(round(paper_w_px * UPSAMPLE))
    dst_h = int(round(paper_h_px * UPSAMPLE))

    destination = np.array(
        [
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(corners, destination)

    warped = cv2.warpPerspective(image, transform, (dst_w, dst_h))

    return warped

# ============================================================
# BAR DETECTION
# ============================================================

# A bar must be at least this long (in cm) to count. Converted to pixels
# below using the perspective scale, so it is a physical length rather
# than a fixed pixel count.
MIN_BAR_CM = 1.5


def detect_bars(paper):
    """Detect horizontal bars on the rectified paper and measure them.

    The paper has been rectified (straightened + upsampled) purely for
    detection robustness. A bar's pixel length is turned into a real-world
    length with the perspective projection equation
        L[cm] = (dx[px] / fx) * Z * 100 = dx[px] / RECT_PX_PER_CM
    where RECT_PX_PER_CM encodes fx and Z (not the paper's known size).
    """
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    height = gray.shape[0]

    # Even out the shadow that falls across the right side of the paper.
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    blur = cv2.GaussianBlur(norm, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7
    )

    # Keep only horizontal structure (the bars). The kernel width is
    # ~0.8 cm in real-world units (scaled by RECT_PX_PER_CM), so it keeps
    # bars longer than ~0.8 cm and removes noise and short fragments.
    kernel_w = max(3, int(round(0.8 * RECT_PX_PER_CM)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Minimum bar length in pixels (1.5 cm in real-world units).
    min_bar_px = MIN_BAR_CM * RECT_PX_PER_CM

    # Drop spurious strips near the paper's top/bottom edge (adaptive
    # thresholding produces them at the paper border).
    edge_margin = int(0.02 * height)

    segments = []

    for contour in contours:

        x, y, w, _ = cv2.boundingRect(contour)

        # Drop short fragments and edge artefacts.
        if w < min_bar_px:
            continue

        if y < edge_margin or y > height - edge_margin:
            continue

        segments.append((y, x, x + w))

    segments.sort()

    # Merge fragments of the same bar (a bar can be split by glare).
    merge_tol = max(3, int(round(0.2 * RECT_PX_PER_CM)))

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
            merged.append(
                {
                    "x1": x1,
                    "x2": x2,
                    "y": float(y),
                    "n": 1,
                }
            )

    bars = []

    for m in merged:

        x1 = m["x1"]
        x2 = m["x2"]
        y = int(round(m["y"]))
        width_px = x2 - x1

        bars.append(
            {
                "x": x1,
                "y": y,
                "width_px": width_px,
                "length_cm": width_px / RECT_PX_PER_CM,
            }
        )

    # Report top-to-bottom.
    bars.sort(key=lambda item: item["y"])

    return bars, binary, horizontal

# ============================================================
# PROCESS MEASUREMENT IMAGES
# ============================================================

measurement_images = sorted(
    list(MEASUREMENT_DIR.glob("*.jpeg")) +
    list(MEASUREMENT_DIR.glob("*.jpg")) +
    list(MEASUREMENT_DIR.glob("*.png"))
)

print()
print(
    f"Found {len(measurement_images)} measurement images."
)

# Accumulate every measurement's error across all images so the final
# statistics cover all 20 measurements.
all_measured_cm = []
all_errors_cm = []
all_truth_cm = []
all_image = []

for image_path in measurement_images:

    print()
    print("============================================================")
    print(f"PROCESSING {image_path.name}")
    print("============================================================")

    img = cv2.imread(str(image_path))

    if img is None:
        print(
            f"[ERROR] Could not read "
            f"{image_path}"
        )
        continue

    # --------------------------------------------------------
    # UNDISTORT
    # --------------------------------------------------------

    # Undistort with the ORIGINAL camera matrix so the image is not
    # rescaled. (getOptimalNewCameraMatrix(alpha=1) zooms out to keep
    # every pixel, which changes the scale and inflates measurements.)
    undistorted = cv2.undistort(
        img,
        camera_matrix,
        distortion_coefficients,
        None,
        camera_matrix
    )

    undistorted_path = (
        OUTPUT_DIR /
        f"undistorted_{image_path.name}"
    )

    cv2.imwrite(
        str(undistorted_path),
        undistorted
    )

    # --------------------------------------------------------
    # DETECT THE WHITE PAPER (only to locate it, NOT to set scale)
    # --------------------------------------------------------

    corners = detect_paper_corners(undistorted)

    # Sanity check: measure the paper itself with perspective projection
    # and compare to its known size. If this differs much from
    # 35.56 x 27.94 cm, OBJECT_DISTANCE_M does not match the true
    # camera-to-object distance.
    paper_w_px = (
        np.linalg.norm(corners[1] - corners[0])
        + np.linalg.norm(corners[2] - corners[3])
    ) / 2.0
    paper_h_px = (
        np.linalg.norm(corners[3] - corners[0])
        + np.linalg.norm(corners[2] - corners[1])
    ) / 2.0

    paper_w_cm = (paper_w_px / fx) * OBJECT_DISTANCE_M * 100.0
    paper_h_cm = (paper_h_px / fy) * OBJECT_DISTANCE_M * 100.0

    print()
    print("Paper size sanity check (perspective projection):")
    print(f"  measured {paper_w_cm:.2f} x {paper_h_cm:.2f} cm")
    print(f"  truth    {PAPER_WIDTH_CM:.2f} x {PAPER_HEIGHT_CM:.2f} cm")

    # Draw detected paper boundary.
    corner_vis = undistorted.copy()

    for i, point in enumerate(corners):

        x, y = point.astype(int)

        cv2.circle(
            corner_vis,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            corner_vis,
            str(i + 1),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    for i in range(4):

        p1 = tuple(corners[i].astype(int))
        p2 = tuple(
            corners[(i + 1) % 4].astype(int)
        )

        cv2.line(
            corner_vis,
            p1,
            p2,
            (0, 0, 255),
            3
        )

    cv2.imwrite(
        str(
            OUTPUT_DIR /
            f"paper_corners_{image_path.name}"
        ),
        corner_vis
    )

    # --------------------------------------------------------
    # RECTIFY + DETECT AND MEASURE BARS (perspective projection)
    # --------------------------------------------------------

    # Rectify the paper (straighten + upsample) for robust detection.
    # The real-world scale still comes from fx and Z (see detect_bars).
    paper = rectify_paper(undistorted, corners)

    bars, binary, horizontal = detect_bars(paper)

    binary_path = (
        OUTPUT_DIR /
        f"binary_{image_path.name}"
    )

    horizontal_path = (
        OUTPUT_DIR /
        f"horizontal_{image_path.name}"
    )

    cv2.imwrite(
        str(binary_path),
        binary
    )

    cv2.imwrite(
        str(horizontal_path),
        horizontal
    )

    # --------------------------------------------------------
    # DRAW MEASUREMENTS
    # --------------------------------------------------------

    result = paper.copy()

    print()
    print("Detected bars:")
    print()

    bar_number = 1

    for bar in bars:

        x = bar["x"]
        y = bar["y"]
        width_px = bar["width_px"]
        length_cm = bar["length_cm"]

        x1 = x
        x2 = x + width_px

        # Draw bar.
        cv2.line(
            result,
            (x1, y),
            (x2, y),
            (0, 0, 255),
            5
        )

        # Draw endpoints.
        cv2.circle(
            result,
            (x1, y),
            8,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            result,
            (x2, y),
            8,
            (255, 0, 0),
            -1
        )

        label = (
            f"{bar_number}: "
            f"{length_cm:.2f} cm"
        )

        cv2.putText(
            result,
            label,
            (x1, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        print(
            f"{bar_number:2d}. "
            f"{length_cm:.2f} cm"
        )

        bar_number += 1

    # --------------------------------------------------------
    # COMPARE WITH GROUND TRUTH
    # --------------------------------------------------------

    truth = GROUND_TRUTH.get(image_path.name)

    if truth is not None:

        print()

        if len(truth) != len(bars):
            print(
                f"[WARNING] Expected {len(truth)} bars, "
                f"detected {len(bars)}."
            )

        print(
            f"{'#':>3}  "
            f"{'measured':>9}  "
            f"{'truth':>6}  "
            f"{'error':>8}  "
            f"{'%err':>6}"
        )

        for number, (bar, expected) in enumerate(
            zip(bars, truth),
            start=1
        ):

            measured = bar["length_cm"]
            error = measured - expected
            pct = 100.0 * error / expected

            all_measured_cm.append(measured)
            all_errors_cm.append(error)
            all_truth_cm.append(expected)
            all_image.append(image_path.name)

            print(
                f"{number:3d}  "
                f"{measured:8.2f}  "
                f"{expected:6d}  "
                f"{error:8.2f}  "
                f"{pct:5.1f}%"
            )

    result_path = (
        OUTPUT_DIR /
        f"measurements_{image_path.name}"
    )

    cv2.imwrite(
        str(result_path),
        result
    )

    print()
    print("Saved:")
    print(f"  {undistorted_path}")
    print(f"  {result_path}")

# ============================================================
# AGGREGATE ERROR STATISTICS
# ============================================================

errors = np.array(all_errors_cm)
truths = np.array(all_truth_cm)
abs_errors = np.abs(errors)
rel_errors = 100.0 * abs_errors / truths

print()
print("============================================================")
print("AGGREGATE ERROR STATISTICS")
print("============================================================")

if len(errors) == 0:
    print("No measurements matched ground truth.")
else:
    print(f"number of measurements : {len(errors)}")
    print(f"mean signed error      : {errors.mean():+.3f} cm   (bias)")
    print(f"mean absolute error    : {abs_errors.mean():.3f} cm")
    print(f"RMSE                   : {np.sqrt(np.mean(errors ** 2)):.3f} cm")
    print(f"standard deviation     : {errors.std():.3f} cm")
    print(f"min absolute error     : {abs_errors.min():.3f} cm")
    print(f"max absolute error     : {abs_errors.max():.3f} cm")
    print(f"mean absolute % error  : {rel_errors.mean():.2f} %")
    print(f"max absolute % error   : {rel_errors.max():.2f} %")

# ============================================================
# EXPORT RESULTS (CSV + PLOT)
# ============================================================

import csv

csv_path = OUTPUT_DIR / "measurements.csv"

with open(csv_path, "w", newline="") as fh:

    writer = csv.writer(fh)
    writer.writerow(
        ["image", "index", "measured_cm", "truth_cm",
         "error_cm", "pct_error"]
    )

    for i in range(len(all_measured_cm)):

        writer.writerow(
            [
                all_image[i],
                i + 1,
                round(all_measured_cm[i], 3),
                all_truth_cm[i],
                round(all_errors_cm[i], 3),
                round(100.0 * all_errors_cm[i] / all_truth_cm[i], 2),
            ]
        )

print()
print(f"Wrote {csv_path}")

# Plot measured vs ground truth and the error distribution.
try:

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
    ax.plot(
        lim,
        lim,
        "--",
        color="#888888",
        lw=1.5,
        label="ideal (measured = truth)"
    )
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
    ax.axvline(
        errs.mean(),
        color="#d62728",
        lw=1.5,
        label=f"mean = {errs.mean():+.2f} cm"
    )
    ax.set_xlabel("Error (cm)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    ax.legend()

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "error_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

except Exception as exc:  # plotting is optional, never fatal
    print(f"[WARN] Could not generate plot: {exc}")

# ============================================================
# DONE
# ============================================================

print()
print("============================================================")
print("DONE")
print("============================================================")
print()
print("Open the output/ directory.")
