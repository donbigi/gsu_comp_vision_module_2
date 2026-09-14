# GSU Computer Vision — Module 2

Camera calibration and real-world 2-D measurement from a single smartphone image using
perspective projection.

- **Step 1** — calibrate the camera (chessboard) with OpenCV.
- **Step 2** — measure real-world bar lengths with the perspective-projection equation
  `L = (dx / fx) · Z`.
- **Step 3** — validate against 20 ground-truth measurements and report error statistics.
- **Theory** — two-camera epipolar-geometry derivation (see `theory.md`).

---

## Requirements

- Python 3.9+
- `opencv-python`
- `numpy`
- `matplotlib` *(optional — used only to save the error plot; the script still runs without it)*

Install with:

```bash
pip install opencv-python numpy matplotlib
```

---

## Project structure

```
gsu_comp_vision_module_2/
├── calibrate_and_measure.py   # the whole pipeline (calibration + measurement)
├── calibration/               # chessboard photos for Step 1  (IMG_6164 … IMG_6184)
├── measurement/               # object photos for Step 3     (IMG_6192, IMG_6193)
├── output/                    # generated results (created on run)
├── theory.md / theory.docx                # Step 4: two-camera derivation
├── validation_report.md / .docx           # Step 3: experiment write-up
└── md_to_docx.py               # regenerates the .docx files from the .md files
```

---

## Run it

From **inside this directory** (the script uses relative paths):

```bash
cd gsu_comp_vision_module_2
python3 calibrate_and_measure.py
```

The script does two things in one pass:

1. **Calibrates** the camera from the `calibration/` chessboard images and saves
   `output/camera_calibration.npz`.
2. **Measures** the bars in the `measurement/` images, compares them to ground truth, and
   writes the results.

---

## Configuration

Edit the constants at the top of `calibrate_and_measure.py`:

| Constant | Meaning | Current value |
|---|---|---|
| `CHESSBOARD_SIZE` | Internal corners of the calibration board | `(9, 6)` |
| `SQUARE_SIZE` | Chessboard square size (metres) | `0.023` |
| `OBJECT_DISTANCE_M` | **Camera-to-object distance (metres)** | `2.73` |
| `PAPER_WIDTH_CM` / `PAPER_HEIGHT_CM` | Object (sheet) true size — used only for the sanity check | `35.56 / 27.94` |
| `GROUND_TRUTH` | Known bar lengths (cm), top-to-bottom, per image | see file |

> **Important:** `OBJECT_DISTANCE_M` must be your physically measured distance (> 2 m). The
> value `2.73` is the distance implied by the sheet's known width — replace it with your
> tape-measured / laser-measured value for a fully independent validation.

---

## Outputs

After a run, `output/` contains:

| File | Description |
|---|---|
| `camera_calibration.npz` | Camera matrix + distortion coefficients |
| `corners_*.jpeg` | Chessboard corners drawn on each calibration image |
| `undistorted_*.jpeg` | Measurement images after distortion correction |
| `paper_corners_*.jpeg` | Detected paper boundary |
| `binary_*.jpeg`, `horizontal_*.jpeg` | Bar-detection masks (debug) |
| `measurements_*.jpeg` | Bars drawn with measured lengths |
| `measurements.csv` | All 20 measurements + errors |
| `error_plot.png` | Measured-vs-truth scatter + error histogram |

The console also prints the calibration result (RMS reprojection error), the paper-size sanity
check, the per-measurement table, and the aggregate error statistics.

---

## Reproducing the Word reports

The Theory and Validation documents are kept as Markdown and converted to Word:

```bash
python3 md_to_docx.py                  # regenerates theory.docx and validation_report.docx
python3 md_to_docx.py theory.md        # convert just one file
```

This requires `python-docx`:

```bash
pip install python-docx
```
