"""
Netra AI — Offline video evaluation script.

Runs the detection/pose/FSM pipeline on a video file and reports:
  - Total frames processed
  - Persons detected per frame (avg, peak)
  - FSM events fired (by type + count)
  - Risk score distribution
  - Hypothetical incidents (score > threshold)
  - Processing FPS

Usage:
    python scripts/evaluate_video.py --video /data/grocery_store_clip.mp4
    python scripts/evaluate_video.py --video /data/clip.mp4 --threshold 0.65 --max-frames 500
    python scripts/evaluate_video.py --video /data/clip.mp4 --output /data/results.json

Requires: ultralytics, opencv-python
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def evaluate(
    video_path: str,
    threshold: float = 0.65,
    max_frames: int | None = None,
    output_path: str | None = None,
) -> dict:
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required. pip install opencv-python")
        return {}

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics required. pip install ultralytics")
        return {}

    # Import pipeline components
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from backend.core.settings import get_settings
    settings = get_settings()

    # Load models
    print(f"Loading detector: {settings.detector_model_path}")
    det_model = YOLO(settings.detector_model_path)

    print(f"Loading pose model: {settings.pose_model_path}")
    pose_model = YOLO(settings.pose_model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return {}

    fps_video  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_vid  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path} ({total_vid} frames @ {fps_video:.1f} fps)")

    # Simple inline tracker (IOU-based for offline eval)
    class _SimpleTracker:
        def __init__(self):
            self._tracks = {}
            self._next_id = 1

        def update(self, boxes):
            # Assign track IDs by IoU matching (greedy)
            import numpy as np
            new_tracks = {}
            assigned   = set()
            for box in boxes:
                x1, y1, x2, y2, conf = box[:5]
                best_iou  = 0.3
                best_tid  = None
                for tid, (px1, py1, px2, py2, _) in self._tracks.items():
                    ix1 = max(x1, px1); iy1 = max(y1, py1)
                    ix2 = min(x2, px2); iy2 = min(y2, py2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = ((x2-x1)*(y2-y1) + (px2-px1)*(py2-py1) - inter)
                        iou   = inter / union if union > 0 else 0
                        if iou > best_iou and tid not in assigned:
                            best_iou = iou
                            best_tid = tid
                if best_tid is None:
                    best_tid = self._next_id
                    self._next_id += 1
                new_tracks[best_tid] = (x1, y1, x2, y2, conf)
                assigned.add(best_tid)
            self._tracks = new_tracks
            return list(assigned)

    tracker = _SimpleTracker()

    # FSM per track
    from backend.services.behavior.fsm import TheftRiskFSM
    from netra.types import PersonDetection, BBox, Keypoint

    fsms: dict[str, TheftRiskFSM] = {}
    camera_id = "eval-camera"

    # Stats accumulators
    frame_count      = 0
    person_counts: list[int] = []
    event_counts     = defaultdict(int)
    risk_scores: list[float] = []
    incidents        = 0
    t_start          = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        # Detection
        results = det_model(frame, verbose=False, conf=settings.detection_conf)
        boxes   = results[0].boxes.data.cpu().numpy() if results[0].boxes else []

        persons: list[PersonDetection] = []
        if len(boxes):
            track_ids = tracker.update(boxes)
            for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2, conf = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
                persons.append(PersonDetection(
                    track_id=tid,
                    camera_id=camera_id,
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=conf,
                    keypoints=[],
                    timestamp=time.time(),
                    frame_seq=frame_count,
                ))

        person_counts.append(len(persons))

        # Run FSM per person
        for p in persons:
            fsm = fsms.setdefault(str(p.track_id), TheftRiskFSM(camera_id))
            events = fsm.process([p], [], time.time())
            for ev in events:
                sig = ev.fsm_state.value if hasattr(ev.fsm_state, "value") else str(ev.fsm_state)
                event_counts[sig] += 1
                if hasattr(ev, "risk_score"):
                    rs = float(ev.risk_score)
                    risk_scores.append(rs)
                    if rs >= threshold:
                        incidents += 1

        if frame_count % 100 == 0:
            elapsed = time.time() - t_start
            pfps    = frame_count / elapsed
            print(f"  Frame {frame_count}/{total_vid or '?'} | {pfps:.1f} fps | persons/frame: {sum(person_counts[-100:])/100:.1f}")

    cap.release()
    elapsed   = time.time() - t_start
    proc_fps  = frame_count / elapsed if elapsed > 0 else 0

    results = {
        "video":             video_path,
        "frames_processed":  frame_count,
        "processing_fps":    round(proc_fps, 2),
        "video_fps":         round(fps_video, 2),
        "realtime_ratio":    round(proc_fps / fps_video, 2),
        "avg_persons_per_frame": round(sum(person_counts) / max(1, len(person_counts)), 2),
        "peak_persons":      max(person_counts) if person_counts else 0,
        "fsm_events":        dict(event_counts),
        "total_fsm_events":  sum(event_counts.values()),
        "hypothetical_incidents": incidents,
        "threshold_used":    threshold,
        "risk_score_stats":  _stats(risk_scores),
    }

    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2))

    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to: {output_path}")

    return results


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    import statistics
    return {
        "count": len(values),
        "mean":  round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min":   round(min(values), 4),
        "max":   round(max(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Netra AI offline video evaluator")
    parser.add_argument("--video",      required=True,              help="Path to video file")
    parser.add_argument("--threshold",  type=float, default=0.65,   help="Risk threshold for incidents")
    parser.add_argument("--max-frames", type=int,   default=None,   help="Max frames to process")
    parser.add_argument("--output",     type=str,   default=None,   help="Write results JSON to path")
    args = parser.parse_args()
    evaluate(args.video, args.threshold, args.max_frames, args.output)


if __name__ == "__main__":
    main()
