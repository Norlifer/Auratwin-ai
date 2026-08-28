"""Scheduled CCTV dataset ingestion for AuraTwin AI.

This module deliberately stops at the existing detector boundary.  It finds
the next dataset item, converts a video frame to JPEG when necessary, and
passes the resulting bytes to ``CCTVPersonDetector.process_snapshot``—the
same function used by the manual upload API.

The scheduler is intentionally implemented with the Python standard library
so the API does not need another scheduling dependency.  A background daemon
thread runs one job immediately at startup and then every configured interval.
State is persisted as JSON so a restart can continue from the next image or
video frame instead of starting over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import uuid

import cv2

from cctv import CCTVPersonDetector, cctv_detector


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(raw_value: str, project_root: Path) -> Path:
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class CCTVAutomationConfig:
    """Configuration for dataset discovery, sampling, state, and output."""

    dataset_path: Optional[Path]
    output_path: Path
    state_path: Path
    interval_minutes: float = 20.0
    enabled: bool = True
    default_zone_id: str = "zone_1"
    video_frame_step: int = 1
    recursive: bool = True
    reprocess_completed: bool = False

    @classmethod
    def from_env(cls, project_root: Optional[Path] = None) -> "CCTVAutomationConfig":
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()

        dataset_raw = os.getenv("CCTV_DATASET_PATH", "").strip()
        dataset_path = _resolve_path(dataset_raw, root) if dataset_raw else None

        output_raw = os.getenv("CCTV_AUTOMATION_OUTPUT_PATH", "data/cctv_results").strip()
        state_raw = os.getenv(
            "CCTV_AUTOMATION_STATE_PATH", "data/cctv_automation_state.json"
        ).strip()
        output_raw = output_raw or "data/cctv_results"
        state_raw = state_raw or "data/cctv_automation_state.json"

        try:
            interval_minutes = float(os.getenv("CCTV_AUTOMATION_INTERVAL_MINUTES", "20"))
            if interval_minutes <= 0:
                raise ValueError
        except ValueError:
            logger.warning(
                "Invalid CCTV_AUTOMATION_INTERVAL_MINUTES; using the 20 minute default."
            )
            interval_minutes = 20.0

        try:
            video_frame_step = int(os.getenv("CCTV_VIDEO_FRAME_STEP", "1"))
            if video_frame_step < 1:
                raise ValueError
        except ValueError:
            logger.warning("Invalid CCTV_VIDEO_FRAME_STEP; using 1 frame.")
            video_frame_step = 1

        return cls(
            dataset_path=dataset_path,
            output_path=_resolve_path(output_raw, root),
            state_path=_resolve_path(state_raw, root),
            interval_minutes=interval_minutes,
            enabled=_env_bool("CCTV_AUTOMATION_ENABLED", True),
            default_zone_id=os.getenv("CCTV_AUTOMATION_DEFAULT_ZONE", "zone_1").strip()
            or "zone_1",
            video_frame_step=video_frame_step,
            recursive=_env_bool("CCTV_DATASET_RECURSIVE", True),
            reprocess_completed=_env_bool("CCTV_REPROCESS_COMPLETED", False),
        )


@dataclass(frozen=True)
class DatasetItem:
    """One supported image or video file discovered in the dataset."""

    path: Path
    relative_path: str
    kind: str  # image or video


class CCTVAutomationProcessor:
    """Selects, processes, stores, and checkpoints sequential dataset items."""

    def __init__(
        self,
        config: CCTVAutomationConfig,
        detector: CCTVPersonDetector = cctv_detector,
        zone_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.zone_ids: Set[str] = set(zone_ids or [])
        self._state_lock = threading.RLock()
        self._state = self._load_state()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "last_processed_file": None,
            "last_processed_frame": None,
            "next_item": None,
            "last_processed_zone": None,
            "last_processed_at": None,
            "last_result_path": None,
            "last_metadata_path": None,
            "last_error": None,
            "items": {},
        }

    def _load_state(self) -> Dict[str, Any]:
        path = self.config.state_path
        if not path.exists():
            return self._default_state()
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            state = self._default_state()
            if isinstance(loaded, dict):
                state.update(loaded)
            if not isinstance(state.get("items"), dict):
                state["items"] = {}
            return state
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Could not read CCTV automation state %s: %s", path, error)
            return self._default_state()

    def _save_state(self) -> None:
        path = self.config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self._state, handle, indent=2, sort_keys=True)
            temporary.replace(path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _discover_items(self) -> List[DatasetItem]:
        dataset = self.config.dataset_path
        if dataset is None:
            logger.warning(
                "CCTV automation is enabled but CCTV_DATASET_PATH is not configured."
            )
            return []
        if not dataset.exists():
            logger.error("CCTV dataset path does not exist: %s", dataset)
            return []

        candidates: Iterable[Path]
        if dataset.is_file():
            candidates = [dataset]
        elif self.config.recursive:
            candidates = (path for path in dataset.rglob("*") if path.is_file())
        else:
            candidates = (path for path in dataset.iterdir() if path.is_file())

        items: List[DatasetItem] = []
        for path in candidates:
            # Do not rediscover generated annotated JPGs when a user points
            # CCTV_DATASET_PATH at a broad parent directory such as ``data``.
            try:
                path.resolve().relative_to(self.config.output_path.resolve())
                continue
            except ValueError:
                pass
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            try:
                relative_path = path.relative_to(dataset if dataset.is_dir() else dataset.parent)
            except ValueError:
                relative_path = Path(path.name)
            items.append(
                DatasetItem(
                    path=path,
                    relative_path=relative_path.as_posix(),
                    kind="video" if suffix in VIDEO_EXTENSIONS else "image",
                )
            )

        items.sort(key=lambda item: item.relative_path.lower())
        if not items:
            logger.warning("No supported image/video files found under %s", dataset)
        return items

    def _item_state(self, item: DatasetItem) -> Dict[str, Any]:
        items = self._state.setdefault("items", {})
        value = items.setdefault(
            item.relative_path,
            {
                "kind": item.kind,
                "status": "pending",
                "next_frame_index": 0,
                "last_frame_index": None,
                "last_processed_at": None,
                "last_result_path": None,
                "last_error": None,
            },
        )
        value.setdefault("kind", item.kind)
        value.setdefault("status", "pending")
        value.setdefault("next_frame_index", 0)
        return value

    def _reset_completed_state(self) -> None:
        for value in self._state.setdefault("items", {}).values():
            if value.get("status") == "completed":
                value.update(
                    {
                        "status": "pending",
                        "next_frame_index": 0,
                        "last_frame_index": None,
                        "last_error": None,
                    }
                )

    def _select_next_item(self, items: List[DatasetItem]) -> Optional[DatasetItem]:
        for item in items:
            state = self._item_state(item)
            if state.get("status") == "completed":
                continue
            if state.get("status") == "failed":
                continue
            return item

        if self.config.reprocess_completed and items:
            logger.info("All dataset items are complete; resetting for configured replay.")
            self._reset_completed_state()
            self._save_state()
            return items[0]
        return None

    def _set_next_item(self, items: List[DatasetItem]) -> None:
        """Persist the next sequential item name for operators/monitoring."""
        tracked = self._state.setdefault("items", {})
        self._state["next_item"] = next(
            (
                item.relative_path
                for item in items
                if tracked.get(item.relative_path, {}).get("status", "pending")
                not in {"completed", "failed"}
            ),
            None,
        )

    def _zone_for_item(self, item: DatasetItem) -> str:
        # A dataset can use zone_1/zone_2/... subfolders. Files without a zone
        # folder use the configured default, preserving compatibility with a
        # flat image/video directory.
        for part in Path(item.relative_path).parts[:-1]:
            if part in self.zone_ids:
                return part
        return self.config.default_zone_id

    @staticmethod
    def _read_image_bytes(path: Path) -> bytes:
        return path.read_bytes()

    @staticmethod
    def _read_video_frame(path: Path, frame_index: int) -> Tuple[Optional[bytes], int, int]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open CCTV video: {path}")

        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > 0 and frame_index >= total_frames:
                return None, total_frames, frame_index

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Could not extract frame {frame_index} from {path}")

            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if not encoded_ok:
                raise ValueError(f"Could not encode frame {frame_index} from {path}")
            return encoded.tobytes(), total_frames, frame_index
        finally:
            capture.release()

    def _source_bytes(self, item: DatasetItem, state: Dict[str, Any]) -> Tuple[bytes, Optional[int], int]:
        if item.kind == "image":
            return self._read_image_bytes(item.path), None, 0

        frame_index = max(0, int(state.get("next_frame_index", 0) or 0))
        image_bytes, total_frames, selected_frame = self._read_video_frame(
            item.path, frame_index
        )
        if image_bytes is None:
            # ``None`` means the cursor has reached the end of a valid video.
            state["status"] = "completed"
            state["last_error"] = None
            self._save_state()
            raise StopIteration
        return image_bytes, selected_frame, total_frames

    @staticmethod
    def _safe_name(value: str) -> str:
        value = value.replace("\\", "_").replace("/", "_")
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
        return value.strip("._") or "snapshot"

    def _save_result(
        self,
        item: DatasetItem,
        zone_id: str,
        detection: Dict[str, Any],
        annotated_jpeg: bytes,
        frame_index: Optional[int],
    ) -> Tuple[Path, Path]:
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc)
        stamp = timestamp.strftime("%Y%m%dT%H%M%S_%fZ")
        frame_suffix = f"_frame_{frame_index:06d}" if frame_index is not None else ""
        stem = self._safe_name(Path(item.relative_path).stem)
        base_name = f"{stamp}_{zone_id}_{stem}{frame_suffix}_{uuid.uuid4().hex[:8]}"
        image_path = self.config.output_path / f"{base_name}.jpg"
        metadata_path = self.config.output_path / f"{base_name}.json"

        image_path.write_bytes(annotated_jpeg)
        metadata = {
            "processed_at": timestamp.isoformat(),
            "zone_id": zone_id,
            "source_file": item.relative_path,
            "source_type": item.kind,
            "frame_index": frame_index,
            "people_count": detection.get("people_count", 0),
            "head_count": detection.get("head_count", 0),
            "width": detection.get("width"),
            "height": detection.get("height"),
            "annotated_image_path": str(image_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return image_path, metadata_path

    def _mark_failed(self, item: DatasetItem, error: Exception) -> None:
        state = self._item_state(item)
        state["status"] = "failed"
        state["last_error"] = str(error)
        self._state["last_error"] = {
            "file": item.relative_path,
            "error": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save_state()

    def process_next(self) -> Dict[str, Any]:
        """Process one next item/frame and return an operational summary."""
        started_at = datetime.now(timezone.utc)
        logger.info("CCTV automation job started at %s", started_at.isoformat())

        with self._state_lock:
            items = self._discover_items()
            if not items:
                result = {
                    "status": "no_dataset_item",
                    "message": "No supported CCTV dataset item is available.",
                    "started_at": started_at.isoformat(),
                }
                self._state["last_error"] = result["message"]
                self._state["next_item"] = None
                self._save_state()
                return result

            # Reaching the end of a video is not a failed run. Move on to the
            # next sequential item during this same scheduled invocation.
            attempts = 0
            while attempts < len(items):
                attempts += 1
                item = self._select_next_item(items)
                if item is None:
                    item_states = [self._item_state(candidate) for candidate in items]
                    all_completed = all(
                        value.get("status") == "completed" for value in item_states
                    )
                    result = {
                        "status": "complete" if all_completed else "no_processable_item",
                        "message": (
                            "All CCTV dataset items have been processed."
                            if all_completed
                            else "All discovered CCTV items are complete or failed."
                        ),
                        "started_at": started_at.isoformat(),
                    }
                    self._state["last_error"] = None
                    self._set_next_item(items)
                    self._save_state()
                    logger.info(result["message"])
                    return result

                item_state = self._item_state(item)
                zone_id = self._zone_for_item(item)
                if self.zone_ids and zone_id not in self.zone_ids:
                    error = ValueError(
                        f"Dataset item {item.relative_path} maps to unknown zone {zone_id}"
                    )
                    self._mark_failed(item, error)
                    logger.error("CCTV item skipped: %s", error)
                    continue

                frame_index: Optional[int] = None
                try:
                    image_bytes, frame_index, _ = self._source_bytes(item, item_state)
                except StopIteration:
                    logger.info("Finished video %s; selecting the next dataset item.", item.relative_path)
                    continue
                except (OSError, ValueError) as error:
                    self._mark_failed(item, error)
                    logger.exception("CCTV source failed; skipping %s", item.relative_path)
                    continue

                logger.info(
                    "Processing CCTV %s (type=%s, frame=%s, zone=%s)",
                    item.relative_path,
                    item.kind,
                    frame_index if frame_index is not None else "image",
                    zone_id,
                )

                try:
                    # This is the exact detector function used by the manual
                    # POST /snapshots/{zone_id} endpoint.
                    detection = self.detector.process_snapshot(
                        zone_id,
                        image_bytes,
                        filename=(
                            f"{item.path.name}#frame={frame_index}"
                            if frame_index is not None
                            else item.path.name
                        ),
                    )
                    annotated_jpeg = self.detector.get_annotated_image(zone_id)
                    if annotated_jpeg is None:
                        raise RuntimeError("Detector returned no annotated image")
                    image_path, metadata_path = self._save_result(
                        item, zone_id, detection, annotated_jpeg, frame_index
                    )
                except ValueError as error:
                    # Bad image bytes/model input are item-specific. Marking
                    # them failed prevents one corrupt file blocking the set.
                    self._mark_failed(item, error)
                    logger.exception("CCTV processing failed; skipping %s", item.relative_path)
                    continue
                except Exception as error:
                    # Infrastructure/model errors are retried on a future run
                    # rather than permanently marking the dataset item failed.
                    self._state["last_error"] = {
                        "file": item.relative_path,
                        "error": str(error),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self._save_state()
                    logger.exception("CCTV processing error; item will be retried: %s", item.relative_path)
                    return {
                        "status": "error",
                        "file": item.relative_path,
                        "error": str(error),
                        "started_at": started_at.isoformat(),
                    }

                processed_at = datetime.now(timezone.utc).isoformat()
                item_state["last_processed_at"] = processed_at
                item_state["last_frame_index"] = frame_index
                item_state["last_result_path"] = str(image_path)
                item_state["last_error"] = None
                if item.kind == "image":
                    item_state["status"] = "completed"
                else:
                    item_state["next_frame_index"] = int(frame_index or 0) + self.config.video_frame_step

                self._state.update(
                    {
                        "last_processed_file": item.relative_path,
                        "last_processed_frame": frame_index,
                        "last_processed_zone": zone_id,
                        "last_processed_at": processed_at,
                        "last_result_path": str(image_path),
                        "last_metadata_path": str(metadata_path),
                        "last_error": None,
                    }
                )
                self._set_next_item(items)
                self._save_state()
                logger.info(
                    "CCTV processing succeeded: file=%s frame=%s count=%s result=%s",
                    item.relative_path,
                    frame_index if frame_index is not None else "image",
                    detection.get("people_count", 0),
                    image_path,
                )
                return {
                    "status": "processed",
                    "file": item.relative_path,
                    "frame_index": frame_index,
                    "zone_id": zone_id,
                    "people_count": detection.get("people_count", 0),
                    "head_count": detection.get("head_count", 0),
                    "result_path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "processed_at": processed_at,
                    "started_at": started_at.isoformat(),
                }

            result = {
                "status": "no_processable_item",
                "message": "All discovered dataset items are complete or failed.",
                "started_at": started_at.isoformat(),
            }
            self._set_next_item(items)
            self._save_state()
            return result

    def status(self) -> Dict[str, Any]:
        with self._state_lock:
            items = self._state.get("items", {})
            counts = {"pending": 0, "completed": 0, "failed": 0}
            for value in items.values():
                status = value.get("status", "pending")
                counts[status] = counts.get(status, 0) + 1
            return {
                "enabled": self.config.enabled,
                "dataset_path": str(self.config.dataset_path) if self.config.dataset_path else None,
                "output_path": str(self.config.output_path),
                "state_path": str(self.config.state_path),
                "interval_minutes": self.config.interval_minutes,
                "default_zone_id": self.config.default_zone_id,
                "video_frame_step": self.config.video_frame_step,
                "recursive": self.config.recursive,
                "reprocess_completed": self.config.reprocess_completed,
                "tracked_items": len(items),
                "item_status_counts": counts,
                "last_processed_file": self._state.get("last_processed_file"),
                "last_processed_frame": self._state.get("last_processed_frame"),
                "next_item": self._state.get("next_item"),
                "last_processed_zone": self._state.get("last_processed_zone"),
                "last_processed_at": self._state.get("last_processed_at"),
                "last_result_path": self._state.get("last_result_path"),
                "last_error": self._state.get("last_error"),
            }


class CCTVAutomationScheduler:
    """Non-overlapping background scheduler for the dataset processor."""

    def __init__(self, processor: CCTVAutomationProcessor) -> None:
        self.processor = processor
        self.interval_seconds = max(1.0, processor.config.interval_minutes * 60.0)
        self._stop_event = threading.Event()
        self._job_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._lifecycle_lock = threading.Lock()
        self._last_job: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        if not self.processor.config.enabled:
            logger.info("CCTV automation is disabled by CCTV_AUTOMATION_ENABLED.")
            return
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="cctv-automation-scheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "CCTV automation scheduler started (interval=%s minutes, dataset=%s)",
                self.processor.config.interval_minutes,
                self.processor.config.dataset_path,
            )

    def stop(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
        thread.join(timeout=10.0)
        if thread.is_alive():
            logger.warning("CCTV automation scheduler did not stop within 10 seconds")
        else:
            logger.info("CCTV automation scheduler stopped")

    def run_once(self) -> Dict[str, Any]:
        if not self.processor.config.enabled:
            result = {"status": "disabled"}
            self._last_job = result
            return result
        if not self._job_lock.acquire(blocking=False):
            result = {"status": "skipped_overlap", "message": "A CCTV job is already running."}
            logger.warning(result["message"])
            self._last_job = result
            return result
        try:
            result = self.processor.process_next()
            self._last_job = result
            return result
        except Exception as error:  # defensive guard so the scheduler thread survives
            logger.exception("Unexpected CCTV automation job failure")
            result = {"status": "error", "error": str(error)}
            self._last_job = result
            return result
        finally:
            self._job_lock.release()

    def _run_loop(self) -> None:
        # Run once immediately after application startup, then keep a stable
        # monotonic schedule. A long job delays the next run but never overlaps.
        next_run = time.monotonic()
        while not self._stop_event.is_set():
            self.run_once()
            next_run += self.interval_seconds
            wait_seconds = max(0.0, next_run - time.monotonic())
            if self._stop_event.wait(wait_seconds):
                break

    def status(self) -> Dict[str, Any]:
        result = self.processor.status()
        result.update(
            {
                "running": bool(self._thread and self._thread.is_alive()),
                "job_running": self._job_lock.locked(),
                "last_job": self._last_job,
            }
        )
        return result
