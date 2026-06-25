import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp

from create_table import create_table

DB_PATH = Path(__file__).parent / "face_detection.db"
ACCEPTABLE_CONFIDENCE = 0.9
MIN_FACE_SIZE_RATIO = 0.15

mp_face_detection = mp.solutions.face_detection


def insert_detection(
    request_id: str,
    frame_number: int,
    detected_at: str,
    camera_index: int,
    frame_width: int,
    frame_height: int,
    confidence: float,
    bbox: tuple[float, float, float, float],
    bbox_px: tuple[int, int, int, int],
    keypoints: dict[str, tuple[float, float]],
) -> None:
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    bbox_x_px, bbox_y_px, bbox_width_px, bbox_height_px = bbox_px

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO face_detections (
                request_id, frame_number, detected_at, camera_index,
                frame_width, frame_height, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height,
                bbox_x_px, bbox_y_px, bbox_width_px, bbox_height_px,
                right_eye_x, right_eye_y, left_eye_x, left_eye_y,
                nose_tip_x, nose_tip_y, mouth_center_x, mouth_center_y,
                right_ear_tragion_x, right_ear_tragion_y,
                left_ear_tragion_x, left_ear_tragion_y
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id, frame_number, detected_at, camera_index,
                frame_width, frame_height, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height,
                bbox_x_px, bbox_y_px, bbox_width_px, bbox_height_px,
                *keypoints["right_eye"], *keypoints["left_eye"],
                *keypoints["nose_tip"], *keypoints["mouth_center"],
                *keypoints["right_ear_tragion"], *keypoints["left_ear_tragion"],
            ),
        )


def draw_face_overlay(
    frame, bbox_px: tuple[int, int, int, int], keypoints_px, confidence: float, is_complete: bool
) -> None:
    x, y, w, h = bbox_px
    color = (0, 220, 0) if is_complete else (0, 165, 255)
    corner_len = max(int(min(w, h) * 0.2), 8)
    thickness = 2

    corners = ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1))
    for corner_x, corner_y, dx, dy in corners:
        cv2.line(frame, (corner_x, corner_y), (corner_x + dx * corner_len, corner_y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (corner_x, corner_y), (corner_x, corner_y + dy * corner_len), color, thickness, cv2.LINE_AA)

    for point_x, point_y in keypoints_px:
        cv2.circle(frame, (point_x, point_y), 2, color, -1, lineType=cv2.LINE_AA)

    label = f"{confidence * 100:.0f}%"
    cv2.putText(
        frame, label, (x, max(y - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )


def is_face_complete(confidence: float, box) -> bool:
    within_frame = (
        box.xmin >= 0
        and box.ymin >= 0
        and box.xmin + box.width <= 1
        and box.ymin + box.height <= 1
    )
    big_enough = box.width >= MIN_FACE_SIZE_RATIO and box.height >= MIN_FACE_SIZE_RATIO

    return confidence >= ACCEPTABLE_CONFIDENCE and within_frame and big_enough


def detect_faces(
    face_detection, frame, request_id: str, frame_number: int, camera_index: int
) -> bool:
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if not results.detections:
        return False

    detected_at = datetime.now(timezone.utc).isoformat()
    found_complete_face = False
    keypoint_names = (
        "right_eye", "left_eye", "nose_tip",
        "mouth_center", "right_ear_tragion", "left_ear_tragion",
    )

    for detection in results.detections:
        confidence = detection.score[0]
        box = detection.location_data.relative_bounding_box
        is_complete = is_face_complete(confidence, box)

        if is_complete:
            found_complete_face = True

        keypoints = {
            name: (point.x, point.y)
            for name, point in zip(keypoint_names, detection.location_data.relative_keypoints)
        }
        bbox_px = (
            int(box.xmin * frame_width),
            int(box.ymin * frame_height),
            int(box.width * frame_width),
            int(box.height * frame_height),
        )
        keypoints_px = [
            (int(px * frame_width), int(py * frame_height)) for px, py in keypoints.values()
        ]

        draw_face_overlay(frame, bbox_px, keypoints_px, confidence, is_complete)

        insert_detection(
            request_id=request_id,
            frame_number=frame_number,
            detected_at=detected_at,
            camera_index=camera_index,
            frame_width=frame_width,
            frame_height=frame_height,
            confidence=confidence,
            bbox=(box.xmin, box.ymin, box.width, box.height),
            bbox_px=bbox_px,
            keypoints=keypoints,
        )

    return found_complete_face


def capture_frames(camera_index: int = 0) -> str:
    create_table()

    request_id = str(uuid.uuid4())
    frame_number = 0

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível acessar a webcam (índice {camera_index})")

    try:
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as face_detection:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                found_complete_face = detect_faces(
                    face_detection, frame, request_id, frame_number, camera_index
                )
                frame_number += 1

                cv2.imshow("Webcam", frame)
                key_pressed = cv2.waitKey(1) & 0xFF

                if found_complete_face or key_pressed == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return request_id


def main() -> None:
    request_id = capture_frames()
    print(f"request_id: {request_id}")


if __name__ == "__main__":
    main()
