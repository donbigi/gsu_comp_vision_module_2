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
PAPER_WIDTH_CM = 35.56
PAPER_HEIGHT_CM = 27.94

# Pixel resolution of the rectified paper.
# 10 pixels per mm = 100 pixels per cm = 3556 x 2794
PIXELS_PER_CM = 100

WARP_WIDTH = int(PAPER_WIDTH_CM * PIXELS_PER_CM)
WARP_HEIGHT = int(PAPER_HEIGHT_CM * PIXELS_PER_CM)

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
# PERSPECTIVE RECTIFICATION
# ============================================================

def rectify_paper(image, corners):

    destination = np.array(
        [
            [0, 0],
            [WARP_WIDTH - 1, 0],
            [WARP_WIDTH - 1, WARP_HEIGHT - 1],
            [0, WARP_HEIGHT - 1]
        ],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        transform,
        (WARP_WIDTH, WARP_HEIGHT)
    )

    return warped

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
# BAR DETECTION
# ============================================================

# A bar must be at least this long to count (cm).
MIN_BAR_CM = 1.5

# Ignore segments within this many pixels of the paper's top/bottom
# edges (adaptive thresholding produces spurious strips at the borders).
EDGE_MARGIN_PX = 100


def detect_bars(paper):

    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

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

    # Keep only horizontal structure.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 3))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segments = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        # Drop short fragments and edge artefacts.
        if w < MIN_BAR_CM * PIXELS_PER_CM:
            continue

        if y < EDGE_MARGIN_PX or y > height - EDGE_MARGIN_PX:
            continue

        segments.append((y, x, x + w))

    segments.sort()

    # Merge fragments of the same bar (a bar can be split by glare).
    merged = []

    for y, x1, x2 in segments:

        for m in merged:

            if abs(y - m["y"]) <= 20:
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
        w = x2 - x1

        bars.append(
            {
                "x": x1,
                "y": y,
                "width": w,
                "length_cm": w / PIXELS_PER_CM,
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
    # DETECT THE WHITE PAPER (only the paper, not the fridge)
    # --------------------------------------------------------

    corners = detect_paper_corners(undistorted)

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
    # RECTIFY PAPER
    # --------------------------------------------------------

    paper = rectify_paper(undistorted, corners)

    paper_path = (
        OUTPUT_DIR /
        f"rectified_{image_path.name}"
    )

    cv2.imwrite(
        str(paper_path),
        paper
    )

    # --------------------------------------------------------
    # DETECT AND MEASURE BARS
    # --------------------------------------------------------

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
        width = bar["width"]
        length_cm = bar["length_cm"]

        x1 = x
        x2 = x + width

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
            f"{'error':>8}"
        )

        errors = []

        for number, (bar, expected) in enumerate(
            zip(bars, truth),
            start=1
        ):

            measured = bar["length_cm"]
            error = measured - expected

            errors.append(error)

            print(
                f"{number:3d}  "
                f"{measured:8.2f}  "
                f"{expected:6d}  "
                f"{error:8.2f} cm"
            )

        errors = np.array(errors)

        print()
        print(f"Mean abs error: {np.abs(errors).mean():.2f} cm")
        print(f"Max abs error:  {np.abs(errors).max():.2f} cm")

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
    print(f"  {paper_path}")
    print(f"  {result_path}")

# ============================================================
# DONE
# ============================================================

print()
print("============================================================")
print("DONE")
print("============================================================")
print()
print("Open the output/ directory.")
