"""
Netra AI — Association Engine.

Tracks Person ↔ Item ↔ Zone relationships over time.
Detects: shelf approach, shelf interaction, item pickup, item disappear,
         zone enter/exit, speed spike.

Ported from edgeguard logic + zone geometry.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from netra.types import (
    AssocEventType, AssociationEvent, BBox,
    ItemDetection, PersonDetection,
)
from backend.core.settings import get_settings
from backend.core.logging import get_logger

log      = get_logger("association.engine")
settings = get_settings()

try:
    from shapely.geometry import Point, Polygon
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False
    log.warning("shapely not installed — zone intersection disabled")


@dataclass
class ZoneConfig:
    name: str
    polygon: list[tuple[float, float]]   # list of (x, y) vertices
    _poly: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if _HAS_SHAPELY:
            self._poly = Polygon(self.polygon)

    def contains_point(self, x: float, y: float) -> bool:
        if not _HAS_SHAPELY or self._poly is None:
            return False
        return self._poly.contains(Point(x, y))

    def contains_bbox(self, bbox: BBox) -> bool:
        return self.contains_point(bbox.cx, bbox.cy)


def load_zones(path: str) -> list[ZoneConfig]:
    try:
        with open(path) as f:
            raw = json.load(f)
        zones = []
        for z in raw.get("zones", []):
            zones.append(ZoneConfig(name=z["name"], polygon=z["polygon"]))
        return zones
    except FileNotFoundError:
        log.warning(f"Zone config not found at {path}; zone detection disabled")
        return []


# ---------------------------------------------------------------------------
# Per-track state
# ---------------------------------------------------------------------------

@dataclass
class PersonState:
    track_id:     int
    camera_id:    str
    last_bbox:    Optional[BBox]       = None
    last_ts:      float                = field(default_factory=time.time)
    active_zones: set[str]             = field(default_factory=set)
    held_items:   set[int]             = field(default_factory=set)   # item_ids near hand
    speed_history: deque               = field(default_factory=lambda: deque(maxlen=10))
    baseline_speed: float              = 0.0
    frames_near_shelf: int             = 0
    last_speed_spike_ts: float         = 0.0


class AssociationEngine:
    """
    Stateful per-camera association engine.
    Call process_frame() for each DetectionEvent.
    Returns list[AssociationEvent] to publish.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.zones     = load_zones(settings.zone_config_path)
        self._persons: dict[int, PersonState] = {}
        # item_id → last seen bbox (items have no persistent tracker)
        self._last_items: dict[int, tuple[BBox, float]] = {}

    def process_frame(
        self,
        persons: list[PersonDetection],
        items:   list[ItemDetection],
        timestamp: float,
    ) -> list[AssociationEvent]:
        events: list[AssociationEvent] = []

        # Update item registry
        current_item_ids = {it.item_id for it in items}
        disappeared      = set(self._last_items.keys()) - current_item_ids
        for it in items:
            self._last_items[it.item_id] = (it.bbox, timestamp)

        # Prune stale items older than 2 s
        self._last_items = {
            k: v for k, v in self._last_items.items()
            if timestamp - v[1] < 2.0
        }

        seen_track_ids = {p.track_id for p in persons}

        for person in persons:
            tid   = person.track_id
            state = self._persons.setdefault(
                tid,
                PersonState(track_id=tid, camera_id=self.camera_id),
            )
            evts = self._update_person(state, person, items, disappeared, timestamp)
            events.extend(evts)

        # Clean up tracks not seen in last 5 s
        stale = [tid for tid, st in self._persons.items()
                 if tid not in seen_track_ids and timestamp - st.last_ts > 5.0]
        for tid in stale:
            del self._persons[tid]

        return events

    def _update_person(
        self,
        state:      PersonState,
        person:     PersonDetection,
        items:      list[ItemDetection],
        disappeared_items: set[int],
        timestamp:  float,
    ) -> list[AssociationEvent]:
        events: list[AssociationEvent] = []
        bbox   = person.bbox

        # ------------------------------------------------------------------
        # Zone intersection
        # ------------------------------------------------------------------
        current_zones: set[str] = set()
        for zone in self.zones:
            if zone.contains_bbox(bbox):
                current_zones.add(zone.name)
                if zone.name not in state.active_zones:
                    events.append(AssociationEvent(
                        camera_id  = self.camera_id,
                        track_id   = person.track_id,
                        event_type = AssocEventType.ZONE_ENTER,
                        zone_name  = zone.name,
                        timestamp  = timestamp,
                    ))
            elif zone.name in state.active_zones:
                events.append(AssociationEvent(
                    camera_id  = self.camera_id,
                    track_id   = person.track_id,
                    event_type = AssocEventType.ZONE_EXIT,
                    zone_name  = zone.name,
                    timestamp  = timestamp,
                ))
        state.active_zones = current_zones

        # ------------------------------------------------------------------
        # Shelf interaction (hand proximity to shelf zone items)
        # ------------------------------------------------------------------
        nearby_items = _items_near_bbox(bbox, items, threshold_px=80.0)
        if nearby_items:
            state.frames_near_shelf += 1
            if state.frames_near_shelf == 1:
                events.append(AssociationEvent(
                    camera_id  = self.camera_id,
                    track_id   = person.track_id,
                    event_type = AssocEventType.SHELF_APPROACH,
                    timestamp  = timestamp,
                ))
            if state.frames_near_shelf >= settings.n_frames_hand_in_shelf:
                events.append(AssociationEvent(
                    camera_id  = self.camera_id,
                    track_id   = person.track_id,
                    event_type = AssocEventType.SHELF_INTERACTION,
                    timestamp  = timestamp,
                    confidence = 0.8,
                ))
        else:
            state.frames_near_shelf = 0

        # ------------------------------------------------------------------
        # Item pickup (item was near person, now disappeared)
        # ------------------------------------------------------------------
        for item_id in disappeared_items:
            last_bbox, last_ts = self._last_items.get(item_id, (None, 0))
            if last_bbox is not None and last_ts > timestamp - 1.5:
                if _bbox_overlap(bbox, last_bbox, threshold=0.3):
                    events.append(AssociationEvent(
                        camera_id  = self.camera_id,
                        track_id   = person.track_id,
                        event_type = AssocEventType.ITEM_PICKUP,
                        item_id    = item_id,
                        confidence = 0.75,
                        timestamp  = timestamp,
                    ))
                    state.held_items.add(item_id)

        # ------------------------------------------------------------------
        # Speed spike
        # ------------------------------------------------------------------
        if state.last_bbox is not None:
            dt = max(timestamp - state.last_ts, 1e-6)
            dx = bbox.cx - state.last_bbox.cx
            dy = bbox.cy - state.last_bbox.cy
            speed = np.sqrt(dx * dx + dy * dy) / dt
            state.speed_history.append(speed)
            if len(state.speed_history) >= 5:
                baseline = float(np.median(list(state.speed_history)[:-1]))
                state.baseline_speed = baseline
                if (baseline > 1.0
                        and speed > baseline * settings.speed_spike_multiplier
                        and timestamp - state.last_speed_spike_ts > 2.0):
                    events.append(AssociationEvent(
                        camera_id  = self.camera_id,
                        track_id   = person.track_id,
                        event_type = AssocEventType.SPEED_SPIKE,
                        confidence = min(1.0, speed / (baseline * settings.speed_spike_multiplier)),
                        metadata   = {"speed": speed, "baseline": baseline},
                        timestamp  = timestamp,
                    ))
                    state.last_speed_spike_ts = timestamp

        state.last_bbox = bbox
        state.last_ts   = timestamp
        return events


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _items_near_bbox(
    person_bbox: BBox,
    items: list[ItemDetection],
    threshold_px: float,
) -> list[ItemDetection]:
    """Return items whose centre is within threshold_px of the person bbox centre."""
    result = []
    for it in items:
        dx = person_bbox.cx - it.bbox.cx
        dy = person_bbox.cy - it.bbox.cy
        if dx * dx + dy * dy <= threshold_px * threshold_px:
            result.append(it)
    return result


def _bbox_overlap(a: BBox, b: BBox, threshold: float) -> bool:
    return a.iou(b) >= threshold
