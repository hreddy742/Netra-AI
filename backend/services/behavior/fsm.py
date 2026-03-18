"""
Netra AI — TheftRiskFSM.

Direct port of edgeguard/src/rules/theft_fsm.py, adapted to Netra types.
Signal weights are identical (edgeguard-validated, DO NOT CHANGE without
evaluation data — see MEMORY.md).

Key differences from edgeguard version:
- Uses Netra types (PersonDetection, BehaviorSignal, BehaviorEvent)
- Accepts AssociationEvent list instead of inline zone checks
- time-based decay (stable across variable FPS)
- Emits BehaviorEvent per track per frame (not just on threshold breach)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from netra.types import (
    AssocEventType, AssociationEvent, BehaviorEvent, BehaviorSignal,
    ConcealmentType, Keypoint, PersonDetection, TheftStage,
    KP_LEFT_WRIST, KP_RIGHT_WRIST, KP_LEFT_HIP, KP_RIGHT_HIP,
    KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
)
from backend.core.settings import get_settings
from backend.core.logging import get_logger

log      = get_logger("behavior.fsm")
settings = get_settings()

# ------------------------------------------------------------------
# Conceal type classifier from skeleton keypoints
# Ported from edgeguard/src/rules/association.py
# ------------------------------------------------------------------

def _kp(keypoints: list[Keypoint], idx: int) -> Optional[tuple[float, float]]:
    if idx < len(keypoints) and keypoints[idx].conf >= settings.keypoint_confidence_min:
        return (keypoints[idx].x, keypoints[idx].y)
    return None


def _dist(a: Optional[tuple], b: Optional[tuple]) -> float:
    if a is None or b is None:
        return float("inf")
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def classify_concealment(
    keypoints: list[Keypoint],
    hand_to_hip_threshold_px: float,
) -> tuple[ConcealmentType, float]:
    """
    Returns (ConcealmentType, hand_to_hip_distance_px).
    Uses wrist / hip / shoulder geometry.
    """
    lw = _kp(keypoints, KP_LEFT_WRIST)
    rw = _kp(keypoints, KP_RIGHT_WRIST)
    lh = _kp(keypoints, KP_LEFT_HIP)
    rh = _kp(keypoints, KP_RIGHT_HIP)
    ls = _kp(keypoints, KP_LEFT_SHOULDER)
    rs = _kp(keypoints, KP_RIGHT_SHOULDER)

    hip_centre    = _midpoint(lh, rh)
    shoulder_centre = _midpoint(ls, rs)

    best_wrist_hip = min(_dist(lw, hip_centre), _dist(rw, hip_centre))
    best_wrist_shoulder = min(_dist(lw, shoulder_centre), _dist(rw, shoulder_centre))

    if best_wrist_hip > hand_to_hip_threshold_px:
        return ConcealmentType.NONE, best_wrist_hip

    # Classify region
    if hip_centre is not None:
        # Is wrist above shoulder? → hoodie / shirt
        if shoulder_centre is not None:
            for wrist in [lw, rw]:
                if wrist is not None and shoulder_centre is not None:
                    if wrist[1] < shoulder_centre[1]:          # above shoulders (y decreases upward)
                        return ConcealmentType.HAND_TO_HOODIE, best_wrist_hip
                    if wrist[1] < hip_centre[1] - 20:           # torso region
                        return ConcealmentType.HAND_TO_SHIRT, best_wrist_hip

        # Below hip → pants pocket
        for wrist in [lw, rw]:
            if wrist is not None and hip_centre is not None:
                if wrist[1] > hip_centre[1] + 20:
                    return ConcealmentType.HAND_TO_PANTS, best_wrist_hip

        return ConcealmentType.HAND_TO_POCKET, best_wrist_hip

    return ConcealmentType.NONE, best_wrist_hip


def _midpoint(a: Optional[tuple], b: Optional[tuple]) -> Optional[tuple]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ------------------------------------------------------------------
# Per-track state
# ------------------------------------------------------------------

@dataclass
class TrackContext:
    state: TheftStage    = TheftStage.BROWSING
    risk_score: float    = 0.0

    last_ts: float        = field(default_factory=time.time)
    last_decay_ts: float  = field(default_factory=time.time)
    first_seen: float     = field(default_factory=time.time)

    last_pick_ts: float   = 0.0
    last_conceal_ts: float = 0.0
    last_scan_ts: float   = 0.0
    last_bag_ts: float    = 0.0
    last_speed_spike_ts: float = 0.0
    last_nonscan_ts: float = 0.0

    hand_in_shelf_frames: int  = 0
    repeated_shelf: int        = 0
    pick_visual: bool          = False

    active_signals: list[str]  = field(default_factory=list)
    speed_history: deque       = field(default_factory=lambda: deque(maxlen=10))


# ------------------------------------------------------------------
# FSM
# ------------------------------------------------------------------

class TheftRiskFSM:
    """
    Stateful per-camera FSM.
    Call process() with each frame's BehaviorInputs → BehaviorEvent list.
    """

    # Signal weights — edgeguard-validated, do NOT change without evaluation
    _WEIGHTS: dict[str, float] = {
        "SHELF_INTERACTION":          2.0,
        "VISUAL_PICK_CONFIRMED":      4.0,
        "HAND_TO_POCKET":             4.0,
        "HAND_TO_BAG":                3.0,
        "HAND_TO_PANTS":              4.5,
        "HAND_TO_SHIRT":              3.5,
        "HAND_TO_HOODIE":             4.0,
        "EXIT_AFTER_CONCEALMENT":     6.0,
        "REPEATED_SHELF_INTERACTION": 2.0,
        "NONSCAN_BAGGING":            5.0,
        "SPEED_SPIKE":                1.5,
    }

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._tracks: dict[int, TrackContext] = {}

    def process(
        self,
        persons:   list[PersonDetection],
        assoc_events: list[AssociationEvent],
        timestamp: float,
    ) -> list[BehaviorEvent]:
        """Process one frame. Returns BehaviorEvent per person track."""
        # Group association events by track_id for quick lookup
        assoc_by_track: dict[int, list[AssociationEvent]] = {}
        for ev in assoc_events:
            assoc_by_track.setdefault(ev.track_id, []).append(ev)

        result: list[BehaviorEvent] = []
        seen_ids = {p.track_id for p in persons}

        for person in persons:
            ctx     = self._get_or_create(person.track_id, timestamp)
            signals = self._update(ctx, person, assoc_by_track.get(person.track_id, []), timestamp)

            suspicious = ctx.risk_score >= settings.fsm_shopformer_trigger_score

            result.append(BehaviorEvent(
                camera_id        = self.camera_id,
                track_id         = person.track_id,
                fsm_state        = ctx.state,
                fsm_score        = ctx.risk_score,
                signals          = signals,
                concealment_type = self._dominant_conceal(signals),
                suspicious       = suspicious,
                timestamp        = timestamp,
            ))

        # Prune stale tracks
        for tid in list(self._tracks):
            if tid not in seen_ids and timestamp - self._tracks[tid].last_ts > 10.0:
                del self._tracks[tid]

        return result

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _get_or_create(self, track_id: int, ts: float) -> TrackContext:
        if track_id not in self._tracks:
            ctx = TrackContext(first_seen=ts, last_ts=ts, last_decay_ts=ts)
            self._tracks[track_id] = ctx
        return self._tracks[track_id]

    def _update(
        self,
        ctx:          TrackContext,
        person:       PersonDetection,
        assoc:        list[AssociationEvent],
        timestamp:    float,
    ) -> list[BehaviorSignal]:
        self._decay(ctx, timestamp)
        signals: list[BehaviorSignal] = []

        assoc_types = {ev.event_type for ev in assoc}

        # ---- Shelf interaction ----
        if AssocEventType.SHELF_INTERACTION in assoc_types:
            ctx.state = TheftStage.SHELF_INTERACTION
            ctx.repeated_shelf += 1
            self._emit("SHELF_INTERACTION", ctx, 1.0, signals, timestamp, person.track_id)
            if ctx.repeated_shelf > 1:
                self._emit("REPEATED_SHELF_INTERACTION", ctx, 1.0, signals, timestamp, person.track_id)

        # ---- Visual item pickup ----
        if AssocEventType.ITEM_PICKUP in assoc_types:
            ctx.last_pick_ts   = timestamp
            ctx.pick_visual    = True
            ctx.state          = TheftStage.ITEM_PICKED
            self._emit("VISUAL_PICK_CONFIRMED", ctx, 1.0, signals, timestamp, person.track_id)

        # ---- Concealment from pose ----
        recent_pick = (ctx.last_pick_ts > 0
                       and timestamp - ctx.last_pick_ts <= settings.conceal_window_sec)

        if person.keypoints and recent_pick:
            conceal_type, dist_px = classify_concealment(
                person.keypoints,
                settings.hand_to_hip_distance_px,
            )
            if conceal_type != ConcealmentType.NONE:
                ctx.state         = TheftStage.CONCEALMENT
                ctx.last_conceal_ts = timestamp
                signal_name = {
                    ConcealmentType.HAND_TO_POCKET:  "HAND_TO_POCKET",
                    ConcealmentType.HAND_TO_BAG:     "HAND_TO_BAG",
                    ConcealmentType.HAND_TO_PANTS:   "HAND_TO_PANTS",
                    ConcealmentType.HAND_TO_SHIRT:   "HAND_TO_SHIRT",
                    ConcealmentType.HAND_TO_HOODIE:  "HAND_TO_HOODIE",
                }.get(conceal_type, "HAND_TO_POCKET")
                self._emit(signal_name, ctx, 1.0, signals, timestamp, person.track_id,
                           conceal_type=conceal_type)

        # ---- Exit after concealment ----
        if AssocEventType.ZONE_EXIT in assoc_types:
            for ev in assoc:
                if ev.event_type == AssocEventType.ZONE_EXIT and ev.zone_name == "exit_zone":
                    if ctx.last_conceal_ts > 0:
                        ctx.state = TheftStage.HIGH_RISK_EXIT
                        self._emit("EXIT_AFTER_CONCEALMENT", ctx, 1.0, signals, timestamp, person.track_id)

        # ---- Speed spike ----
        if AssocEventType.SPEED_SPIKE in assoc_types:
            if timestamp - ctx.last_speed_spike_ts > 3.0:
                ctx.last_speed_spike_ts = timestamp
                self._emit("SPEED_SPIKE", ctx, 1.0, signals, timestamp, person.track_id)

        ctx.last_ts = timestamp
        return signals

    def _decay(self, ctx: TrackContext, now: float) -> None:
        dt = max(0.0, now - ctx.last_decay_ts)
        ctx.risk_score = max(0.0, ctx.risk_score - settings.fsm_decay_rate * dt)
        ctx.last_decay_ts = now

    def _emit(
        self,
        signal_name: str,
        ctx:         TrackContext,
        value:       float,
        out:         list[BehaviorSignal],
        timestamp:   float,
        track_id:    int,
        conceal_type: ConcealmentType = ConcealmentType.NONE,
    ) -> None:
        weight = self._WEIGHTS.get(signal_name, 0.0)
        ctx.risk_score += weight * value
        ctx.active_signals.append(signal_name)
        out.append(BehaviorSignal(
            camera_id            = self.camera_id,
            track_id             = track_id,
            signal_name          = signal_name,
            weight               = weight * value,
            fsm_state            = ctx.state,
            cumulative_fsm_score = ctx.risk_score,
            concealment_type     = conceal_type,
            timestamp            = timestamp,
        ))

    @staticmethod
    def _dominant_conceal(signals: list[BehaviorSignal]) -> ConcealmentType:
        for s in reversed(signals):
            if s.concealment_type != ConcealmentType.NONE:
                return s.concealment_type
        return ConcealmentType.NONE
