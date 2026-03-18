"""
TrackConfidenceScorer — composite per-track reliability score.

Score = detection_confidence * age_factor * (1 - occlusion_penalty)

  detection_confidence : rolling average of person.confidence over last N frames
  age_factor           : sigmoid(age / 30) — young tracks (< 10 frames) are unreliable
  occlusion_penalty    : 0.4 if currently occluded, 0.0 if clear

Score range: 0.0 – 1.0
Reliable threshold:    settings.track_confidence_min (default 0.4)

Usage:
    scorer = TrackConfidenceScorer()
    scorer.update(camera_id, person, is_occluded=occ_tracker.is_occluded(cam, pid))
    if not scorer.is_reliable(camera_id, person.track_id):
        continue  # skip this track for incident generation
"""
from __future__ import annotations
import math
from collections import defaultdict, deque
from netra.types import PersonDetection
from backend.core.settings import get_settings

settings = get_settings()

_CONF_WINDOW      = 30    # rolling window size
_AGE_SCALE        = 30.0  # sigmoid scale factor (half-max at 30 frames)
_OCCLUSION_PENALTY = 0.40


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _age_factor(frame_count: int) -> float:
    """Smooth ramp from ~0.5 at frame 0 to ~1.0 at frame 60+."""
    return _sigmoid((frame_count - _AGE_SCALE) / (_AGE_SCALE / 4))


class TrackConfidenceScorer:
    """Per-camera, per-track composite confidence scorer."""

    def __init__(self) -> None:
        # conf_history[cam][track_id] = deque of float
        self._conf_history: defaultdict[str, defaultdict[int, deque[float]]] = \
            defaultdict(lambda: defaultdict(lambda: deque(maxlen=_CONF_WINDOW)))
        # frame_count[cam][track_id] = int (total frames observed)
        self._frame_count: defaultdict[str, defaultdict[int, int]] = \
            defaultdict(lambda: defaultdict(int))
        # currently_occluded[cam][track_id] = bool
        self._occluded: defaultdict[str, defaultdict[int, bool]] = \
            defaultdict(lambda: defaultdict(bool))

    def update(
        self,
        camera_id: str,
        person: PersonDetection,
        is_occluded: bool = False,
    ) -> None:
        """Record one frame of detection for a track."""
        tid = person.track_id
        self._conf_history[camera_id][tid].append(person.confidence)
        self._frame_count[camera_id][tid] += 1
        self._occluded[camera_id][tid] = is_occluded

    def score(self, camera_id: str, track_id: int) -> float:
        """Compute composite confidence score [0.0, 1.0]."""
        history = self._conf_history[camera_id][track_id]
        if not history:
            return 0.0
        det_conf    = sum(history) / len(history)
        age         = self._frame_count[camera_id][track_id]
        age_f       = _age_factor(age)
        occ_penalty = _OCCLUSION_PENALTY if self._occluded[camera_id][track_id] else 0.0
        return det_conf * age_f * (1.0 - occ_penalty)

    def is_reliable(self, camera_id: str, track_id: int, threshold: float | None = None) -> bool:
        """True if track score meets reliability threshold."""
        min_score = threshold if threshold is not None else settings.track_confidence_min
        return self.score(camera_id, track_id) >= min_score

    def clear_track(self, camera_id: str, track_id: int) -> None:
        for d in (self._conf_history, self._frame_count, self._occluded):
            if camera_id in d:
                d[camera_id].pop(track_id, None)
