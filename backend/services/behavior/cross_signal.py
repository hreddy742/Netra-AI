"""
CrossSignalValidator — requires multiple distinct signal types per track before
high-confidence incident confirmation.

Prevents false positives from a single signal type firing repeatedly
(e.g., spurious SPEED_SPIKE × 10, or jitter in SHELF_INTERACTION).

A track is "cross-validated" when it has accumulated events from at least
`min_distinct` different AssocEventType values.

Usage in behavior service (alongside SignalConfirmationBuffer):
    validator = CrossSignalValidator()
    validator.record(camera_id, ev.track_id, ev.event_type)
    if not validator.is_valid(camera_id, track_id):
        # suppress low-confidence incident
"""
from __future__ import annotations
from collections import defaultdict
from netra.types import AssocEventType
from backend.core.settings import get_settings

settings = get_settings()

# Signals that don't count toward cross-validation (too noisy / always present)
_EXCLUDED_FROM_CROSS = {AssocEventType.SPEED_SPIKE}


class CrossSignalValidator:
    """
    Per-camera, per-track distinct signal accumulator.
    A track must show >= min_distinct distinct signal types to be considered valid.
    """

    def __init__(self, min_distinct: int | None = None) -> None:
        self._min = min_distinct if min_distinct is not None else settings.cross_signal_min_distinct
        # seen[camera_id][track_id] = set of AssocEventType values
        self._seen: defaultdict[str, defaultdict[int, set[str]]] = \
            defaultdict(lambda: defaultdict(set))

    def record(self, camera_id: str, track_id: int, event_type: AssocEventType) -> None:
        """Record an observed signal type for this track."""
        if event_type not in _EXCLUDED_FROM_CROSS:
            self._seen[camera_id][track_id].add(event_type.value)

    def record_many(self, camera_id: str, events: list) -> None:
        """Record all events from an association event list."""
        for ev in events:
            self.record(camera_id, ev.track_id, ev.event_type)

    def is_valid(self, camera_id: str, track_id: int) -> bool:
        """True if track has seen >= min_distinct distinct qualifying signal types."""
        if self._min <= 1:
            return True  # fast path: validation disabled
        return len(self._seen[camera_id][track_id]) >= self._min

    def distinct_count(self, camera_id: str, track_id: int) -> int:
        return len(self._seen[camera_id][track_id])

    def seen_types(self, camera_id: str, track_id: int) -> set[str]:
        return set(self._seen[camera_id][track_id])

    def clear_track(self, camera_id: str, track_id: int) -> None:
        if camera_id in self._seen:
            self._seen[camera_id].pop(track_id, None)

    def clear_camera(self, camera_id: str) -> None:
        self._seen.pop(camera_id, None)
