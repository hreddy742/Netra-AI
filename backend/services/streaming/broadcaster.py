"""
Netra AI — Frame broadcaster (Phase 8 — pattern pubsub, GPU encode).

Changes from Phase 7:
  - ONE pattern subscription netra:frame-notify:* instead of N per-camera subs
    -> reduces Redis connections from O(cameras) to O(1) at 50+ camera scale
  - GPU-accelerated JPEG via jpeg_encoder (NVJPEG -> libjpeg-turbo -> OpenCV)
  - Per-camera FPS tracking exposed via get_fps(camera_id)
  - Broadcaster FPS gauge published to Prometheus
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import redis.asyncio as aioredis

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from backend.core.metrics import broadcaster_fps  # new Phase 8 metric
from backend.services.streaming.jpeg_encoder import encode_jpeg
from netra.events import decode_frame_pointer
from netra.redis_keys import frame_notify, overlay_data
from netra.shm import ReaderCache

log      = get_logger("broadcaster")
settings = get_settings()

# Pattern covering all camera notify channels
_NOTIFY_PATTERN = "netra:frame-notify:*"


@dataclass
class CameraStreamState:
    camera_id: str
    queues: list[asyncio.Queue] = field(default_factory=list)
    # Rolling timestamps for FPS calculation (last 30 frames)
    _frame_ts: deque = field(default_factory=lambda: deque(maxlen=30))

    def record_frame(self) -> None:
        self._frame_ts.append(time.monotonic())

    def fps(self) -> float:
        if len(self._frame_ts) < 2:
            return 0.0
        span = self._frame_ts[-1] - self._frame_ts[0]
        return (len(self._frame_ts) - 1) / span if span > 0 else 0.0


class FrameBroadcaster:
    """
    Singleton stored on app.state.broadcaster.
    Single pubsub consumer for ALL cameras via pattern subscription.
    """

    def __init__(self) -> None:
        self._cameras: dict[str, CameraStreamState] = {}
        self._redis: aioredis.Redis | None = None
        self._producer_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    async def start(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._producer_task = asyncio.create_task(
            self._producer(), name="broadcaster-producer"
        )

    def add_client(self, camera_id: str) -> asyncio.Queue:
        state = self._cameras.setdefault(camera_id, CameraStreamState(camera_id))
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        state.queues.append(q)
        return q

    def remove_client(self, camera_id: str, q: asyncio.Queue) -> None:
        state = self._cameras.get(camera_id)
        if state:
            try:
                state.queues.remove(q)
            except ValueError:
                pass

    def get_fps(self, camera_id: str) -> float:
        state = self._cameras.get(camera_id)
        return state.fps() if state else 0.0

    async def _producer(self) -> None:
        """
        Single task: pattern-subscribe to all camera notify channels.
        Demux by channel name -> encode JPEG once -> fan-out.
        """
        from backend.services.streaming.overlay import draw_overlay

        assert self._redis is not None

        pubsub = self._redis.pubsub()
        await pubsub.psubscribe(_NOTIFY_PATTERN)
        log.info("Broadcaster producer started (pattern: %s)", _NOTIFY_PATTERN)

        try:
            async for msg in pubsub.listen():
                if msg["type"] != "pmessage":
                    continue

                # Extract camera_id from channel name
                channel: str = msg["channel"]
                cam_id = channel.removeprefix("netra:frame-notify:")

                state = self._cameras.get(cam_id)
                if not state or not state.queues:
                    continue  # zero-encode optimisation

                # Decode FramePointer
                try:
                    fp = decode_frame_pointer(json.loads(msg["data"]))
                except Exception as exc:
                    log.debug("Bad FramePointer %s: %s", cam_id, exc)
                    continue

                # Read from SHM
                try:
                    reader = ReaderCache.get(cam_id, settings.shm_num_slots, fp.height, fp.width)
                    frame  = reader.read(fp.slot_index, fp.generation)
                except (RuntimeError, FileNotFoundError) as exc:
                    log.debug("SHM read %s: %s", cam_id, exc)
                    continue

                # Overlay + JPEG encode (once, shared across all clients)
                try:
                    ov_json = await self._redis.get(overlay_data(cam_id))
                    if ov_json:
                        frame = draw_overlay(frame, ov_json)
                    out_bytes: bytes = encode_jpeg(frame, quality=settings.mjpeg_jpeg_quality)
                except Exception as exc:
                    log.debug("Encode error %s: %s", cam_id, exc)
                    continue

                # Update FPS tracker
                state.record_frame()
                try:
                    broadcaster_fps.labels(camera_id=cam_id).set(state.fps())
                except Exception:
                    pass

                # Fan-out: drop oldest if client is slow
                for q in list(state.queues):
                    if q.full():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        q.put_nowait(out_bytes)
                    except asyncio.QueueFull:
                        pass

        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.punsubscribe(_NOTIFY_PATTERN)
            await pubsub.aclose()
            log.info("Broadcaster producer stopped")

    async def stop(self) -> None:
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass
        self._cameras.clear()
        log.info("FrameBroadcaster stopped")
