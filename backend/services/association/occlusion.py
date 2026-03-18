"""
OcclusionTracker — per-camera, per-track occlusion state.

Detects occlusion by:
  1. Significant IoU overlap between two person bounding boxes (person-person occlusion)
  2. Track disappearing from frame (covered by shelf/fixture)

Provides: is_occluded(camera_id, track_id) → bool
Used by association/main.py to reduce confidence on occluded tracks.
"""
from __future__ import annotations
from collections import defaultdict
from netra.types import BBox, PersonDetection

_PERSON_OCCLUSION_IOU = 0.35   # bbox IoU threshold for person-person occlusion
_DISAPPEAR_FRAMES     = 3      # frames without detection before "disappeared"


class OcclusionTracker:
    """
    Per-camera occlusion state tracker.
    Call update() each frame; query is_occluded() before emitting events.
    """

    def __init__(self) -> None:
        # last_seen[cam][track_id] = frame_seq
        self._last_seen: defaultdict[str, dict[int, int]] = defaultdict(dict)
        # occluded[cam][track_id] = True/False
        self._occluded: defaultdict[str, dict[int, bool]]  = defaultdict(dict)

    def update(self, camera_id: str, persons: list[PersonDetection], frame_seq: int) -> None:
        """Update occlusion state for all tracks in this frame."""
        seen_ids = {p.track_id for p in persons}

        # Update last-seen counter
        for p in persons:
            self._last_seen[camera_id][p.track_id] = frame_seq

        # Detect person-person occlusion via pairwise IoU
        for i, pa in enumerate(persons):
            occ = False
            for j, pb in enumerate(persons):
                if i == j:
                    continue
                if pa.bbox.iou(pb.bbox) >= _PERSON_OCCLUSION_IOU:
                    occ = True
                    break
            self._occluded[camera_id][pa.track_id] = occ

        # Detect disappeared tracks (not seen for N frames)
        for track_id, last_frame in list(self._last_seen[camera_id].items()):
            if track_id not in seen_ids:
                gap = frame_seq - last_frame
                if gap >= _DISAPPEAR_FRAMES:
                    self._occluded[camera_id][track_id] = True

    def is_occluded(self, camera_id: str, track_id: int) -> bool:
        return self._occluded.get(camera_id, {}).get(track_id, False)

    def confidence_factor(self, camera_id: str, track_id: int) -> float:
        """Returns 0.5 if occluded, 1.0 if clear."""
        return 0.5 if self.is_occluded(camera_id, track_id) else 1.0
