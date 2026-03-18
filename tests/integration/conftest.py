"""
Integration test fixtures.

These tests do NOT require a running Docker stack.
They wire the Python service logic directly (no Redis, no SHM).
Redis-dependent services are tested with fakeredis.
"""
from __future__ import annotations

import time
import pytest

from netra.types import (
    AssocEventType, AssociationEvent, BBox, BehaviorEvent,
    ConcealmentType, IncidentStatus, ItemDetection, Keypoint,
    PersonDetection, ShopFormerScore, TheftStage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_person(
    track_id: int = 1,
    cx: float = 200,
    cy: float = 300,
    camera_id: str = "test-cam",
    keypoints: list[Keypoint] | None = None,
) -> PersonDetection:
    half = 60
    return PersonDetection(
        track_id   = track_id,
        camera_id  = camera_id,
        bbox       = BBox(cx - half, cy - 120, cx + half, cy + 120),
        confidence = 0.92,
        keypoints  = keypoints or [],
        timestamp  = time.time(),
    )


def make_item(item_id: int, cx: float, cy: float, cls: str = "bottle") -> ItemDetection:
    h = 20
    return ItemDetection(
        item_id    = item_id,
        camera_id  = "test-cam",
        bbox       = BBox(cx - h, cy - h, cx + h, cy + h),
        class_name = cls,
        confidence = 0.85,
        timestamp  = time.time(),
    )


def make_assoc(
    event_type: AssocEventType,
    track_id: int = 1,
    item_id: int | None = None,
    zone: str | None = None,
) -> AssociationEvent:
    return AssociationEvent(
        camera_id  = "test-cam",
        track_id   = track_id,
        event_type = event_type,
        item_id    = item_id,
        zone_name  = zone,
        timestamp  = time.time(),
    )


def make_concealment_keypoints(
    hip_y: float = 370,
    wrist_y: float = 375,
) -> list[Keypoint]:
    """17 COCO keypoints with wrists near hips."""
    kps = [Keypoint(0, 0, 0)] * 17
    kps[5]  = Keypoint(140, 200, 0.9)  # left shoulder
    kps[6]  = Keypoint(160, 200, 0.9)  # right shoulder
    kps[9]  = Keypoint(142, wrist_y, 0.9)  # left wrist (near hip)
    kps[10] = Keypoint(158, wrist_y, 0.9)  # right wrist
    kps[11] = Keypoint(140, hip_y, 0.9)    # left hip
    kps[12] = Keypoint(160, hip_y, 0.9)    # right hip
    return kps


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def camera_id() -> str:
    return "test-cam"


@pytest.fixture
def base_ts() -> float:
    return time.time()
