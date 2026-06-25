import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "face_detection.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS face_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    detected_at TEXT NOT NULL,
    camera_index INTEGER NOT NULL,
    frame_width INTEGER NOT NULL,
    frame_height INTEGER NOT NULL,
    confidence REAL NOT NULL,
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_width REAL NOT NULL,
    bbox_height REAL NOT NULL,
    bbox_x_px INTEGER NOT NULL,
    bbox_y_px INTEGER NOT NULL,
    bbox_width_px INTEGER NOT NULL,
    bbox_height_px INTEGER NOT NULL,
    right_eye_x REAL,
    right_eye_y REAL,
    left_eye_x REAL,
    left_eye_y REAL,
    nose_tip_x REAL,
    nose_tip_y REAL,
    mouth_center_x REAL,
    mouth_center_y REAL,
    right_ear_tragion_x REAL,
    right_ear_tragion_y REAL,
    left_ear_tragion_x REAL,
    left_ear_tragion_y REAL
)
"""


def create_table() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(CREATE_TABLE_SQL)


if __name__ == "__main__":
    create_table()
