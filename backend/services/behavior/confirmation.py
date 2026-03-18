"""
SignalConfirmationBuffer — majority-vote temporal smoothing.

A signal is confirmed when >= ceil(N/2) of the last N frames contain it.
  N=3 (default): 2 of 3 frames must agree.
  N=1          : passthrough (no filtering).

Age eviction: if a signal hasn't been seen for > age_threshold_ms (event time),
its buffer is cleared. This prevents stale half-gestures from carrying over.

Uses event_time for age comparison — correct under Redis queue backlog where
wall clock would falsely evict all signals.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque

from netra.types import AssociationEvent, AssocEventType
from backend.core.settings import get_settings

settings = get_settings()


class SignalConfirmationBuffer:
    """
    Per-camera, per-track, per-signal-type majority-vote confirmation buffer.
    Confirmed when sum(deque) >= ceil(N/2) and deque has >= ceil(N/2) entries.
    """

    def __init__(
        self,
        required: int | None = None,
        age_threshold_ms: float | None = None,
    ) -> None:
        self._required = required if required is not None \
                         else settings.signal_confirmation_frames
        age_ms = age_threshold_ms if age_threshold_ms is not None \
                 else settings.signal_confirmation_age_ms
        self._age_thresh_sec = age_ms / 1000.0
        self._majority = math.ceil(self._required / 2)

        # _bufs[camera_id][track_id][signal_type] = deque[bool]
        self._bufs: defaultdict = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: deque(maxlen=self._required))
            )
        )
        # _last_ts[camera_id][track_id][signal_type] = last event_time seen
        self._last_ts: defaultdict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

    def filter(
        self,
        events: list[AssociationEvent],
        camera_id: str,
        event_time: float | None = None,
    ) -> list[AssociationEvent]:
        """
        Return events that pass the majority-vote threshold.
        event_time: frame capture timestamp (used for age eviction).
        """
        if self._required <= 1:
            return events  # fast path

        now = event_time if event_time is not None else time.time()
        confirmed: list[AssociationEvent] = []
        seen: set[tuple] = set()

        for ev in events:
            sig = (ev.track_id, ev.event_type.value)
            seen.add(sig)
            buf = self._bufs[camera_id][ev.track_id][ev.event_type.value]
            last = self._last_ts[camera_id][ev.track_id][ev.event_type.value]

            # Age eviction: clear on gap > threshold
            if last > 0 and now - last > self._age_thresh_sec:
                buf.clear()

            buf.append(True)
            self._last_ts[camera_id][ev.track_id][ev.event_type.value] = now

            # Majority vote
            if len(buf) >= self._majority and sum(buf) >= self._majority:
                confirmed.append(ev)

        # Signals NOT seen this frame: push False (streak broken) unless expired
        for track_id, sig_map in self._bufs[camera_id].items():
            for sig_type, buf in sig_map.items():
                if (track_id, sig_type) not in seen:
                    last = self._last_ts[camera_id][track_id][sig_type]
                    if last > 0 and now - last <= self._age_thresh_sec:
                        buf.append(False)

        return confirmed

    def clear_track(self, camera_id: str, track_id: int) -> None:
        if camera_id in self._bufs:
            self._bufs[camera_id].pop(track_id, None)
        if camera_id in self._last_ts:
            self._last_ts[camera_id].pop(track_id, None)
