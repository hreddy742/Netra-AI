"""
ZoneHeatmap — accumulates person track positions into a 2D density grid per camera.

Grid is stored in Redis as a JSON array (rows×cols of float counts) with configurable TTL.
The gateway exposes it via GET /api/cameras/{id}/heatmap.

Resolution: 32×18 grid (maps to 1280×720 px; each cell = 40×40 px)
"""
from __future__ import annotations
import json
import numpy as np
import redis.asyncio as aioredis

from netra.types import PersonDetection

_GRID_COLS = 32
_GRID_ROWS = 18
_HEATMAP_TTL = 3600  # 1 hour before Redis expires the key


def _heatmap_key(camera_id: str) -> str:
    return f"netra:heatmap:{camera_id}"


class ZoneHeatmap:
    """
    Accumulates track centroids into a density grid.
    Persists in Redis; each call to update() increments cells.
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    async def update(
        self,
        camera_id: str,
        persons: list[PersonDetection],
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> None:
        if not persons:
            return
        # Load existing grid
        raw = await self._redis.get(_heatmap_key(camera_id))
        if raw:
            grid = np.array(json.loads(raw), dtype=np.float32)
        else:
            grid = np.zeros((_GRID_ROWS, _GRID_COLS), dtype=np.float32)

        for p in persons:
            cx = int(p.bbox.cx / frame_width * _GRID_COLS)
            cy = int(p.bbox.cy / frame_height * _GRID_ROWS)
            cx = max(0, min(_GRID_COLS - 1, cx))
            cy = max(0, min(_GRID_ROWS - 1, cy))
            grid[cy, cx] += 1.0

        await self._redis.set(
            _heatmap_key(camera_id),
            json.dumps(grid.tolist()),
            ex=_HEATMAP_TTL,
        )

    async def get(self, camera_id: str) -> dict:
        """Return grid as dict for API response."""
        raw = await self._redis.get(_heatmap_key(camera_id))
        if not raw:
            grid = np.zeros((_GRID_ROWS, _GRID_COLS)).tolist()
        else:
            grid = json.loads(raw)
        return {
            "camera_id": camera_id,
            "grid_rows": _GRID_ROWS,
            "grid_cols": _GRID_COLS,
            "grid": grid,
        }

    async def reset(self, camera_id: str) -> None:
        await self._redis.delete(_heatmap_key(camera_id))
