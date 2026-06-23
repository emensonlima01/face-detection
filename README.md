# Face Detection

Captures webcam frames in real time, runs face detection on each frame with
[MediaPipe](https://github.com/google-ai-edge/mediapipe), and persists the detection
metadata to a local SQLite database for later analysis.

## How it works

1. The webcam is opened and frames are read continuously.
2. Each frame is passed through MediaPipe's `FaceDetection` model.
3. Every detection found is saved to SQLite, tagged with a `request_id` (a UUID generated
   once per run), so all detections from a single execution can be queried together.
4. The capture loop stops automatically once a **complete face** is found (see below), or
   when the `q` key is pressed.

### What counts as a "complete" face

A detection is considered complete when all of the following are true:

- **Confidence** is at or above `ACCEPTABLE_CONFIDENCE` (default `0.9`).
- **Bounding box** is fully inside the frame (the face isn't cut off at an edge).
- **Size** is at or above `MIN_FACE_SIZE_RATIO` (default `0.15`, i.e. at least 15% of the
  frame's width/height) — filters out faces too small/far to be useful.

Both constants are defined at the top of `main.py` and can be tuned as needed.

## Stored data

Each detection row in the `face_detections` table includes:

| Category | Fields |
|---|---|
| Run context | `request_id`, `frame_number`, `detected_at`, `camera_index`, `frame_width`, `frame_height` |
| Score | `confidence` |
| Bounding box | `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height` (relative, 0–1) and their pixel equivalents (`bbox_x_px`, `bbox_y_px`, `bbox_width_px`, `bbox_height_px`) |
| Facial keypoints | relative x/y for `right_eye`, `left_eye`, `nose_tip`, `mouth_center`, `right_ear_tragion`, `left_ear_tragion` |

All rows from the same run share the same `request_id`, making it easy to pull every
detection from a given execution:

```sql
SELECT * FROM face_detections WHERE request_id = '<guid-printed-at-the-end-of-the-run>';
```

## Requirements

- Python 3.11+
- A webcam accessible to the OS

Dependencies are listed in `requirements.txt`:

- `opencv-python` — webcam capture and frame display
- `mediapipe==0.10.14` — face detection (pinned: newer releases dropped the legacy
  `solutions` API used here)

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

```bash
.venv\Scripts\python.exe main.py
```

A window titled "Webcam" opens showing the live feed with detected faces outlined.
Press `q` at any time to stop manually. On exit, the run's `request_id` is printed to
the console.

The SQLite database file is `face_detection.db` in the project root. Its schema (the
`face_detections` table) is not created by the application code — it must already exist
before running `main.py`.
