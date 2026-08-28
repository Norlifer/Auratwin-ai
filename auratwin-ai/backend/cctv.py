"""CCTV snapshot person counting for AuraTwin AI.

The detector intentionally counts only the COCO ``person`` class.  It does not
perform face recognition or store identity information.  The latest annotated
snapshot is kept in memory so that a newly uploaded image can immediately drive
the zone occupancy and HVAC recommendation pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0
MAX_SNAPSHOT_BYTES = 10 * 1024 * 1024


class CCTVPersonDetector:
    """Detect people in uploaded CCTV images using a lazy-loaded YOLO model."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model: Optional[YOLO] = None
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _get_model(self) -> YOLO:
        """Load YOLO on first use so server startup does not download a model."""
        with self._lock:
            if self._model is None:
                self._model = YOLO(self.model_name)
            return self._model

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("The uploaded snapshot is empty.")

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("The uploaded file is not a readable image.")
        return frame

    def _detect(self, image_bytes: bytes) -> Dict[str, Any]:
        """Return a person count and an annotated JPEG for one image."""
        frame = self._decode_image(image_bytes)
        result = self._get_model()(
            frame,
            conf=self.confidence_threshold,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]

        people_count = 0
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue

                people_count += 1
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    frame,
                    f"Person {confidence:.0%}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            f"People detected: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

        encoded_ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        if not encoded_ok:
            raise RuntimeError("OpenCV could not encode the annotated snapshot.")

        height, width = frame.shape[:2]
        return {
            "people_count": people_count,
            "head_count": people_count,
            "width": width,
            "height": height,
            "annotated_jpeg": encoded.tobytes(),
        }

    def process_snapshot(
        self,
        zone_id: str,
        image_bytes: bytes,
        filename: str = "snapshot.jpg",
    ) -> Dict[str, Any]:
        """Detect and remember the latest snapshot for a zone."""
        if not zone_id:
            raise ValueError("zone_id is required.")
        if len(image_bytes) > MAX_SNAPSHOT_BYTES:
            raise ValueError("The uploaded snapshot is larger than 10 MB.")

        detected = self._detect(image_bytes)
        snapshot = {
            "zone_id": zone_id,
            "filename": filename or "snapshot.jpg",
            "people_count": detected["people_count"],
            "head_count": detected["head_count"],
            "width": detected["width"],
            "height": detected["height"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "annotated_jpeg": detected["annotated_jpeg"],
        }
        with self._lock:
            self._latest[zone_id] = snapshot
        return self._public_metadata(snapshot)

    def get_people_count(self, zone_id: str) -> int:
        """Return the latest detected count, or zero before a snapshot is uploaded."""
        with self._lock:
            snapshot = self._latest.get(zone_id)
            return int(snapshot["people_count"]) if snapshot else 0

    def get_snapshot_metadata(self, zone_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            snapshot = self._latest.get(zone_id)
            return self._public_metadata(snapshot) if snapshot else None

    def get_annotated_image(self, zone_id: str) -> Optional[bytes]:
        with self._lock:
            snapshot = self._latest.get(zone_id)
            return snapshot["annotated_jpeg"] if snapshot else None

    @staticmethod
    def _public_metadata(snapshot: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if snapshot is None:
            return None
        return {
            key: value
            for key, value in snapshot.items()
            if key != "annotated_jpeg"
        }


cctv_detector = CCTVPersonDetector(
    model_name=os.getenv("CCTV_MODEL_NAME", "yolov8n.pt"),
    confidence_threshold=float(os.getenv("CCTV_CONFIDENCE", "0.40")),
)
