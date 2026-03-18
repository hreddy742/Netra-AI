"""Tests for ZoneHeatmap (using fakeredis or mock)."""
import asyncio
import json
import pytest
import numpy as np
from netra.types import PersonDetection, BBox, Keypoint
import time


def _person(cx, cy):
    return PersonDetection(
        track_id=1, camera_id="cam", confidence=0.9,
        bbox=BBox(cx-30, cy-60, cx+30, cy+60),
        keypoints=[], timestamp=time.time(),
    )


@pytest.mark.asyncio
async def test_heatmap_update_increments_cell():
    pytest.importorskip("fakeredis")
    import fakeredis.aioredis as fakeredis
    from backend.services.analytics.heatmap import ZoneHeatmap

    r = fakeredis.FakeRedis(decode_responses=True)
    hm = ZoneHeatmap(r)

    # Person at center of 1280x720 frame → cell (16, 9)
    await hm.update("cam", [_person(640, 360)], frame_width=1280, frame_height=720)
    result = await hm.get("cam")

    grid = np.array(result["grid"])
    assert grid[9, 16] == 1.0


@pytest.mark.asyncio
async def test_heatmap_accumulates():
    pytest.importorskip("fakeredis")
    import fakeredis.aioredis as fakeredis
    from backend.services.analytics.heatmap import ZoneHeatmap

    r = fakeredis.FakeRedis(decode_responses=True)
    hm = ZoneHeatmap(r)
    p = _person(100, 100)
    await hm.update("cam", [p])
    await hm.update("cam", [p])
    result = await hm.get("cam")
    grid = np.array(result["grid"])
    assert grid.max() == 2.0


@pytest.mark.asyncio
async def test_heatmap_reset():
    pytest.importorskip("fakeredis")
    import fakeredis.aioredis as fakeredis
    from backend.services.analytics.heatmap import ZoneHeatmap

    r = fakeredis.FakeRedis(decode_responses=True)
    hm = ZoneHeatmap(r)
    await hm.update("cam", [_person(640, 360)])
    await hm.reset("cam")
    result = await hm.get("cam")
    grid = np.array(result["grid"])
    assert grid.sum() == 0.0
