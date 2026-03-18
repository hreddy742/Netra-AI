"""
Netra AI — Edge deployment mode.

Provides:
  - is_edge_mode() → bool — True when NETRA_EDGE=1
  - LocalAlertQueue — writes alerts to local JSONL file when Redis unavailable
  - EdgeSyncWorker — reads queue file and publishes to Redis when reconnected

Edge mode = pipeline runs fully local, no cloud dependencies.
Alerts stored in /data/edge_queue/alerts.jsonl, synced when Redis recovers.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from backend.core.logging import get_logger
from backend.core.settings import get_settings

log      = get_logger("edge_mode")
settings = get_settings()

_QUEUE_FILE = Path("/data/edge_queue/alerts.jsonl")


def is_edge_mode() -> bool:
    return os.environ.get("NETRA_EDGE", "0") == "1"


class LocalAlertQueue:
    """
    Appends alert dicts to a local JSONL file.
    Thread-safe via asyncio lock.
    """

    def __init__(self, path: Path = _QUEUE_FILE) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def push(self, alert: dict[str, Any]) -> None:
        alert["_queued_at"] = time.time()
        async with self._lock:
            with open(self._path, "a") as f:
                f.write(json.dumps(alert) + "\n")

    async def drain(self) -> list[dict[str, Any]]:
        """Read and clear the queue. Returns all pending alerts."""
        async with self._lock:
            if not self._path.exists():
                return []
            lines = self._path.read_text().strip().splitlines()
            self._path.write_text("")   # clear
            return [json.loads(l) for l in lines if l.strip()]

    @property
    def size(self) -> int:
        if not self._path.exists():
            return 0
        return sum(1 for _ in open(self._path))


class EdgeSyncWorker:
    """
    Periodically attempts to sync the local alert queue to Redis.
    Runs as a background task in edge deployments.
    """

    def __init__(self, queue: LocalAlertQueue, redis_url: str) -> None:
        self._queue     = queue
        self._redis_url = redis_url
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def start(self) -> None:
        self._task = asyncio.create_task(self._sync_loop(), name="edge-sync")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _sync_loop(self) -> None:
        import redis.asyncio as aioredis
        from netra.events import STREAM_INCIDENTS
        try:
            while True:
                await asyncio.sleep(30)
                if self._queue.size == 0:
                    continue
                try:
                    r = aioredis.Redis.from_url(self._redis_url, decode_responses=True)
                    await r.ping()
                    alerts = await self._queue.drain()
                    for alert in alerts:
                        await r.xadd(STREAM_INCIDENTS, {k: str(v) for k, v in alert.items()})
                    await r.aclose()
                    log.info("Edge sync: pushed %d queued alerts", len(alerts))
                except Exception as exc:
                    log.debug("Edge sync failed (offline): %s", exc)
        except asyncio.CancelledError:
            pass
