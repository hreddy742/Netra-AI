"""Unit tests for AssociationEngine."""
from __future__ import annotations

import time
import pytest

from netra.types import AssocEventType, BBox, ItemDetection, PersonDetection, Keypoint
from backend.services.association.engine import AssociationEngine


def _person(track_id: int = 1, cx: float = 200, cy: float = 300) -> PersonDetection:
    half = 50
    return PersonDetection(
        track_id=track_id, camera_id="cam-01",
        bbox=BBox(cx - half, cy - 100, cx + half, cy + 100),
        confidence=0.9, keypoints=[], timestamp=time.time(),
    )


def _item(item_id: int, cx: float, cy: float, cls: str = "bottle") -> ItemDetection:
    half = 20
    return ItemDetection(
        item_id=item_id, camera_id="cam-01",
        bbox=BBox(cx - half, cy - half, cx + half, cy + half),
        class_name=cls, confidence=0.8, timestamp=time.time(),
    )


class TestAssociationEngine:
    def setup_method(self) -> None:
        self.engine = AssociationEngine("cam-01")

    def test_shelf_approach_detected(self) -> None:
        p    = _person(cx=200, cy=300)
        item = _item(0, cx=220, cy=310)   # item within 80px of person centre
        ts   = time.time()
        evts = self.engine.process_frame([p], [item], ts)
        types = {e.event_type for e in evts}
        assert AssocEventType.SHELF_APPROACH in types

    def test_item_pickup_when_item_disappears(self) -> None:
        p    = _person(cx=200, cy=300)
        item = _item(0, cx=210, cy=305)
        ts   = time.time()
        # Frame 1: item present
        self.engine.process_frame([p], [item], ts)
        # Frame 2: item gone
        evts = self.engine.process_frame([p], [], ts + 0.5)
        types = {e.event_type for e in evts}
        assert AssocEventType.ITEM_PICKUP in types

    def test_no_pickup_when_item_far_from_person(self) -> None:
        p    = _person(cx=100, cy=100)
        item = _item(0, cx=1000, cy=600)  # far away
        ts   = time.time()
        self.engine.process_frame([p], [item], ts)
        evts = self.engine.process_frame([p], [], ts + 0.5)
        types = {e.event_type for e in evts}
        assert AssocEventType.ITEM_PICKUP not in types

    def test_speed_spike_detection(self) -> None:
        ts = time.time()
        # Move person slowly for baseline
        for i in range(6):
            p = _person(cx=200 + i * 2, cy=300)   # 2px per frame
            self.engine.process_frame([p], [], ts + i * 0.1)
        # Sudden large jump
        p_fast = _person(cx=400, cy=300)           # +200px in 0.1s
        evts = self.engine.process_frame([p_fast], [], ts + 0.7)
        types = {e.event_type for e in evts}
        assert AssocEventType.SPEED_SPIKE in types
