"""
Netra AI — YOLO detector + OcSort tracker + RTMPose-m / YOLOv8-pose.

Phase-2 upgrades:
  2.1 — YOLO11s (drop-in, same API as YOLOv8n)
  2.2 — RTMPose-m ONNX with SimCC decode; falls back to YOLOv8-pose if absent
  2.3 — OcSort (boxmot) per-camera tracker; falls back to ultralytics ByteTrack
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from netra.types import (
    BBox, ItemDetection, Keypoint, PersonDetection,
)
from backend.core.settings import get_settings
from backend.core.logging import get_logger

log      = get_logger("inference.detector")
settings = get_settings()

# COCO classes that are "retail items" (index subset from 80-class COCO)
_RETAIL_ITEM_CLASSES: set[str] = {
    "backpack", "handbag", "suitcase",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "cell phone", "laptop", "mouse", "remote", "keyboard",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush", "umbrella",
}


# ---------------------------------------------------------------------------
# RTMPose-m ONNX estimator (phase-2.2)
# ---------------------------------------------------------------------------

class RTMPoseEstimator:
    """
    RTMPose-m top-down pose estimator via ONNX Runtime.

    Input : list of BGR person crops (any size)
    Output: list of dicts — {'keypoints': (17,2), 'scores': (17,)}
            Keypoint order: COCO-17, same as YOLOv8-pose.
            Coordinates are in the CROP space (caller maps to full frame).

    SimCC decode:
        simcc_x: (N, 17, W*2)  → argmax / 2 = pixel x in (0, 192)
        simcc_y: (N, 17, H*2)  → argmax / 2 = pixel y in (0, 256)
        score   = (softmax_max_x + softmax_max_y) / 2
    """

    INPUT_W = 192
    INPUT_H = 256
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._session    = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        active = self._session.get_providers()[0]
        log.info(f"RTMPose using provider: {active}")
        if active != "CUDAExecutionProvider":
            log.warning("RTMPose falling back to CPU — check onnxruntime-gpu install")

    def estimate(
        self,
        person_crops: list[np.ndarray],
        crop_origins: list[tuple[int, int]],   # (x1, y1) of each crop in full frame
        crop_sizes:   list[tuple[int, int]],   # (w, h) of each original crop
    ) -> list[dict]:
        """
        Returns list of {'keypoints': (17,2), 'scores': (17,)} in full-frame coords.
        Keypoints with confidence < keypoint_confidence_min are NOT zeroed here —
        the caller checks scores before using each keypoint.
        """
        if not person_crops:
            return []

        batch = np.stack([self._preprocess(c) for c in person_crops])  # (N,3,256,192)
        sim_x, sim_y = self._session.run(None, {self._input_name: batch})
        # sim_x: (N,17,384)  sim_y: (N,17,512)

        kp_x = np.argmax(sim_x, axis=2).astype(np.float32) / 2.0   # (N,17) in [0,192)
        kp_y = np.argmax(sim_y, axis=2).astype(np.float32) / 2.0   # (N,17) in [0,256)

        scores_x = _softmax_max(sim_x)   # (N,17)
        scores_y = _softmax_max(sim_y)   # (N,17)
        scores   = (scores_x + scores_y) * 0.5  # (N,17)

        results = []
        for i in range(len(person_crops)):
            ox, oy   = crop_origins[i]
            cw, ch   = crop_sizes[i]
            # scale from resized (192×256) back to original crop size, then add offset
            xs = kp_x[i] * (cw / self.INPUT_W) + ox
            ys = kp_y[i] * (ch / self.INPUT_H) + oy
            kps = np.stack([xs, ys], axis=1)   # (17, 2)
            results.append({'keypoints': kps, 'scores': scores[i]})
        return results

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (self.INPUT_W, self.INPUT_H))   # (H=256, W=192)
        img = img[:, :, ::-1].astype(np.float32) / 255.0     # BGR→RGB, normalise
        img = (img - self.MEAN) / self.STD
        return img.transpose(2, 0, 1)   # HWC → CHW


def _softmax_max(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax max over last axis. Returns (N, K)."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).max(axis=-1)


# ---------------------------------------------------------------------------
# OcSort tracker manager (phase-2.3) — one instance per camera
# ---------------------------------------------------------------------------

class TrackerManager:
    """
    Wraps OcSort (boxmot) for one camera.
    Falls back to None if boxmot is not installed — caller uses YOLO tracking.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._tracker  = None

        try:
            from boxmot import OcSort
            reid_path = Path(settings.reid_weights_path)
            reid_active = settings.tracker_use_reid and reid_path.exists()
            if settings.tracker_use_reid and not reid_path.exists():
                log.warning(
                    f"ReID weights not found at {reid_path} — "
                    "OcSort running without ReID. "
                    "Download osnet_x0_25_msmt17.pt and place it at that path."
                )
            elif reid_active:
                log.info(
                    f"OcSort ReID active: {reid_path} "
                    f"({reid_path.stat().st_size:,} bytes)"
                )
            self._tracker = OcSort(
                det_thresh  = settings.detection_conf,
                max_age     = settings.tracker_max_age,
                min_hits    = settings.tracker_min_hits,
                iou_threshold = settings.tracker_iou_threshold,
                delta_t     = 3,
                asso_func   = "iou",
                inertia     = 0.2,
                use_byte    = True,
                reid_weights = reid_path if reid_active else None,
                device      = 0 if settings.detector_device not in ("cpu",) else "cpu",
                half        = False,   # FP32 on CPU
            )
            log.info(f"OcSort tracker initialised for camera={camera_id} max_age={settings.tracker_max_age} reid={reid_active}")
        except Exception as exc:
            log.warning(f"OcSort unavailable ({exc}); will use ultralytics ByteTrack fallback")

    @property
    def available(self) -> bool:
        return self._tracker is not None

    def update(
        self,
        detections: np.ndarray,   # (N, 6): [x1,y1,x2,y2,conf,cls]
        frame: np.ndarray,
    ) -> np.ndarray:
        """Returns (M, 8): [x1,y1,x2,y2,track_id,conf,cls,det_idx]"""
        if detections is None or len(detections) == 0:
            empty = np.empty((0, 6), dtype=np.float32)
            result = self._tracker.update(empty, frame)
        else:
            result = self._tracker.update(detections.astype(np.float32), frame)
        return result if result is not None and len(result) > 0 else np.empty((0, 8))


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    Wraps YOLO11s detection + OcSort tracking + RTMPose-m pose estimation.
    Single call to detect() returns persons and retail items together.
    Falls back gracefully: OcSort→ByteTrack, RTMPose→YOLOv8-pose.
    """

    def __init__(self) -> None:
        from ultralytics import YOLO

        self._model      = YOLO(settings.detector_model_path)
        self._device     = settings.detector_device
        self._conf       = settings.detection_conf
        self._iou        = settings.detection_iou
        try:
            _yolo_device = str(next(self._model.model.parameters()).device)
            log.info(f"YOLO device: {_yolo_device}")
        except Exception:
            pass

        # Pose estimator — try RTMPose first, fall back to YOLOv8-pose
        self._pose_model   = None  # YOLOv8-pose fallback
        self._rtmpose      = None  # RTMPoseEstimator (preferred)

        if settings.pose_use_rtmpose:
            onnx_path = settings.pose_onnx_path
            if Path(onnx_path).exists():
                try:
                    self._rtmpose = RTMPoseEstimator(onnx_path)
                except Exception as exc:
                    log.warning(f"RTMPose load failed ({exc}); falling back to YOLOv8-pose")
            else:
                log.warning(f"RTMPose ONNX not found at {onnx_path}; falling back to YOLOv8-pose")

        if self._rtmpose is None:
            self._pose_model = YOLO(settings.pose_model_path)
            log.info(f"Using YOLOv8-pose fallback: {settings.pose_model_path}")

        # Per-camera OcSort trackers (created on demand)
        self._trackers: dict[str, TrackerManager] = {}

        log.info(
            f"YOLODetector loaded: det={settings.detector_model_path} "
            f"pose={'RTMPose-m' if self._rtmpose else settings.pose_model_path} "
            f"device={self._device}"
        )

    def _get_tracker(self, camera_id: str) -> TrackerManager:
        if camera_id not in self._trackers:
            self._trackers[camera_id] = TrackerManager(camera_id)
        return self._trackers[camera_id]

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_seq: int = 0,
        run_pose: bool = True,
    ) -> tuple[list[PersonDetection], list[ItemDetection]]:
        """
        Run detection + tracking on a frame.
        Returns (persons, items).
        """
        ts = time.time()
        tracker = self._get_tracker(camera_id)

        if tracker.available:
            # OcSort path: detect only (no built-in track), then OcSort update
            results = self._model(
                frame,
                conf=self._conf,
                iou=self._iou,
                device=self._device,
                verbose=False,
            )
            persons, items, raw_dets = self._parse_detections(results, camera_id, ts, frame_seq)
            persons = self._apply_ocsort(tracker, raw_dets, frame, persons, camera_id, ts, frame_seq)
        else:
            # Fallback: ultralytics ByteTrack
            results = self._model.track(
                frame,
                persist=True,
                conf=self._conf,
                iou=self._iou,
                device=self._device,
                verbose=False,
            )
            persons, items, _ = self._parse_detections(results, camera_id, ts, frame_seq, use_track_ids=True)

        if run_pose and persons:
            self._run_pose(frame, persons)

        return persons, items

    def _parse_detections(
        self,
        results,
        camera_id: str,
        ts: float,
        frame_seq: int,
        use_track_ids: bool = False,
    ) -> tuple[list[PersonDetection], list[ItemDetection], np.ndarray]:
        """
        Parse ultralytics Results into persons, items, and raw det array (N,6).
        raw_dets: [x1, y1, x2, y2, conf, cls] — used by OcSort.
        """
        persons: list[PersonDetection] = []
        items:   list[ItemDetection]   = []
        raw_rows: list[list[float]]    = []
        item_id_counter = 0

        if not results or results[0].boxes is None:
            return persons, items, np.empty((0, 6), dtype=np.float32)

        res   = results[0]
        boxes = res.boxes

        for i, box in enumerate(boxes):
            cls_idx    = int(box.cls[0])
            class_name = res.names[cls_idx]
            conf       = float(box.conf[0])
            xyxy       = box.xyxy[0].tolist()
            bbox       = BBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3])

            # Raw detection row for OcSort
            raw_rows.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls_idx])

            if class_name == "person":
                track_id = int(box.id[0]) if (use_track_ids and box.id is not None) else -i
                persons.append(PersonDetection(
                    track_id   = track_id,
                    camera_id  = camera_id,
                    bbox       = bbox,
                    confidence = conf,
                    keypoints  = [],
                    timestamp  = ts,
                    frame_seq  = frame_seq,
                ))
            elif class_name in _RETAIL_ITEM_CLASSES:
                items.append(ItemDetection(
                    item_id    = item_id_counter,
                    camera_id  = camera_id,
                    bbox       = bbox,
                    class_name = class_name,
                    confidence = conf,
                    timestamp  = ts,
                ))
                item_id_counter += 1

        raw_dets = np.array(raw_rows, dtype=np.float32) if raw_rows else np.empty((0, 6), dtype=np.float32)
        return persons, items, raw_dets

    def _apply_ocsort(
        self,
        tracker: TrackerManager,
        raw_dets: np.ndarray,
        frame: np.ndarray,
        persons: list[PersonDetection],
        camera_id: str,
        ts: float,
        frame_seq: int,
    ) -> list[PersonDetection]:
        """
        Run OcSort update and re-build persons list with stable track IDs.
        OcSort output cols: [x1,y1,x2,y2,track_id,conf,cls,det_idx]
        """
        track_out = tracker.update(raw_dets, frame)
        if len(track_out) == 0:
            return []

        tracked_persons: list[PersonDetection] = []
        for row in track_out:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            conf     = float(row[5])
            cls      = int(row[6])
            if cls != 0:  # COCO class 0 = person
                continue
            tracked_persons.append(PersonDetection(
                track_id   = track_id,
                camera_id  = camera_id,
                bbox       = BBox(x1, y1, x2, y2),
                confidence = conf,
                keypoints  = [],
                timestamp  = ts,
                frame_seq  = frame_seq,
            ))
        return tracked_persons

    def _run_pose(self, frame: np.ndarray, persons: list[PersonDetection]) -> None:
        """
        Batch pose estimation — fills PersonDetection.keypoints in-place.
        Uses RTMPose-m ONNX if available, falls back to YOLOv8-pose.
        """
        h, w = frame.shape[:2]
        crops:        list[np.ndarray]        = []
        crop_origins: list[tuple[int, int]]   = []
        crop_sizes:   list[tuple[int, int]]   = []
        crop_indices: list[int]               = []

        for idx, p in enumerate(persons):
            x1 = max(0, int(p.bbox.x1))
            y1 = max(0, int(p.bbox.y1))
            x2 = min(w, int(p.bbox.x2))
            y2 = min(h, int(p.bbox.y2))
            cw, ch = x2 - x1, y2 - y1
            if cw < 16 or ch < 16:
                continue
            crops.append(frame[y1:y2, x1:x2])
            crop_origins.append((x1, y1))
            crop_sizes.append((cw, ch))
            crop_indices.append(idx)

        if not crops:
            return

        if self._rtmpose is not None:
            self._run_rtmpose(crops, crop_origins, crop_sizes, crop_indices, persons)
        else:
            self._run_yolo_pose(crops, crop_origins, crop_indices, persons)

    def _run_rtmpose(
        self,
        crops: list[np.ndarray],
        crop_origins: list[tuple[int, int]],
        crop_sizes: list[tuple[int, int]],
        crop_indices: list[int],
        persons: list[PersonDetection],
    ) -> None:
        try:
            pose_results = self._rtmpose.estimate(crops, crop_origins, crop_sizes)
            min_conf = settings.keypoint_confidence_min
            for result_idx, pr in enumerate(pose_results):
                person_idx = crop_indices[result_idx]
                kps    = pr['keypoints']   # (17, 2) full-frame coords
                scores = pr['scores']      # (17,)
                persons[person_idx].keypoints = [
                    Keypoint(x=float(kps[k, 0]), y=float(kps[k, 1]), conf=float(scores[k]))
                    for k in range(min(17, len(kps)))
                ]
        except Exception as exc:
            log.warning(f"RTMPose inference failed: {exc}")

    def _run_yolo_pose(
        self,
        crops: list[np.ndarray],
        crop_origins: list[tuple[int, int]],
        crop_indices: list[int],
        persons: list[PersonDetection],
    ) -> None:
        try:
            pose_results = self._pose_model(crops, device=self._device, verbose=False)
            for result_idx, pose_res in enumerate(pose_results):
                person_idx = crop_indices[result_idx]
                p = persons[person_idx]
                if pose_res.keypoints is None or len(pose_res.keypoints) == 0:
                    continue
                kp_data = pose_res.keypoints.data[0]   # (17, 3): x, y, conf
                x1, y1 = crop_origins[result_idx]
                p.keypoints = [
                    Keypoint(
                        x    = float(kp_data[k, 0]) + x1,
                        y    = float(kp_data[k, 1]) + y1,
                        conf = float(kp_data[k, 2]),
                    )
                    for k in range(min(17, len(kp_data)))
                ]
        except Exception as exc:
            log.warning(f"YOLOv8-pose inference failed: {exc}")
