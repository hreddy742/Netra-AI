"""
Netra AI — mediabridge service.

Responsibility:
  - Open RTSP / file sources (one thread per camera)
  - Write frames to SHM ring buffer
  - Push FramePointer to Redis per-camera queue
  - Maintain /health and heartbeat key
"""
from __future__ import annotations

import asyncio
import json
import signal
import threading
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from backend.core.metrics import frames_ingested, queue_depth, service_up
from netra.shm import RingBufferWriter
from netra.events import encode_frame_pointer
from netra.redis_keys import frame_queue, ACTIVE_CAMERAS, service_heartbeat
from netra.types import FramePointer
from backend.services.mediabridge.source import RTSPSource, VideoFileSource

log     = get_logger("mediabridge")
settings = get_settings()


class CameraWorker(threading.Thread):
    """
    One thread per camera.
    Opens video source → writes to SHM → pushes FramePointer to Redis queue.
    """

    def __init__(
        self,
        camera_id: str,
        source_url: str,
        source_type: str,
        redis_sync: Any,
    ) -> None:
        super().__init__(name=f"mediabridge-{camera_id}", daemon=True)
        self.camera_id   = camera_id
        self.source_url  = source_url
        self.source_type = source_type
        self.redis       = redis_sync
        self._stop       = threading.Event()
        self._frame_seq  = 0

        h, w = settings.shm_frame_height, settings.shm_frame_width
        self._writer = RingBufferWriter(camera_id, settings.shm_num_slots, h, w)
        log.info(f"CameraWorker initialised for {camera_id} ({source_type}:{source_url})")

    def run(self) -> None:
        if self.source_type == "rtsp":
            src = RTSPSource(self.source_url, self.camera_id)
        else:
            src = VideoFileSource(self.source_url, self.camera_id)

        try:
            for raw_frame in src.frames():
                if self._stop.is_set():
                    break
                # Resize to configured resolution
                h, w = settings.shm_frame_height, settings.shm_frame_width
                frame = _resize(raw_frame, w, h)

                slot, gen = self._writer.write(frame)
                self._frame_seq += 1

                fp = FramePointer(
                    camera_id  = self.camera_id,
                    shm_name   = self._writer.shm_name,
                    slot_index = slot,
                    generation = gen,
                    width      = w,
                    height     = h,
                    channels   = 3,
                    timestamp  = time.time(),
                    frame_seq  = self._frame_seq,
                )
                payload = encode_frame_pointer(fp)
                # Check queue depth before pushing — enforce backpressure limit
                current_depth = self.redis.llen(frame_queue(self.camera_id))
                if current_depth >= settings.max_queue_depth_per_camera:
                    # Drop oldest frame (lpop) to make room — newest wins
                    self.redis.rpop(frame_queue(self.camera_id))

                self.redis.lpush(frame_queue(self.camera_id), json.dumps(payload))
                # Hard trim as safety net
                self.redis.ltrim(frame_queue(self.camera_id), 0, settings.max_queue_depth_per_camera - 1)
                frames_ingested.labels(camera_id=self.camera_id).inc()
                queue_depth.labels(camera_id=self.camera_id).set(
                    self.redis.llen(frame_queue(self.camera_id))
                )
        except Exception as e:
            log.error(f"CameraWorker {self.camera_id} crashed: {e}")
        finally:
            src.close()
            self._writer.close()

    def stop(self) -> None:
        self._stop.set()


def _resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


async def heartbeat_loop(redis: aioredis.Redis) -> None:
    key = service_heartbeat("mediabridge")
    while True:
        await redis.set(key, "ok", ex=15)
        await asyncio.sleep(5)


async def main() -> None:
    import redis as sync_redis_lib

    r_async = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=True,
    )
    r_sync = sync_redis_lib.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        decode_responses=True,
    )

    # Discover cameras from Redis active set
    cameras_raw = await r_async.smembers(ACTIVE_CAMERAS)
    if not cameras_raw:
        log.warning("No cameras in ACTIVE_CAMERAS set — using defaults from .env")
        cameras_raw = {settings.redis_host}  # fallback for single-camera mode

    workers: list[CameraWorker] = []

    for cam_id in cameras_raw:
        cfg_raw = await r_async.hgetall(f"netra:config:camera:{cam_id}")
        url         = cfg_raw.get("url", "")
        source_type = cfg_raw.get("source_type", "rtsp")
        if not url:
            log.warning(f"Camera {cam_id} has no URL configured; skipping")
            continue
        w = CameraWorker(cam_id, url, source_type, r_sync)
        w.start()
        workers.append(w)
        service_up.labels(service="mediabridge").set(1)
    log.info(f"Started worker for camera {cam_id}")

    # Heartbeat task
    asyncio.create_task(heartbeat_loop(r_async))

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _handle_sig(*_: Any) -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGTERM, _handle_sig)
    loop.add_signal_handler(signal.SIGINT, _handle_sig)

    await stop_event.wait()

    log.info("Stopping camera workers…")
    for w in workers:
        w.stop()
    for w in workers:
        w.join(timeout=5)

    await r_async.aclose()
    log.info("mediabridge stopped")


if __name__ == "__main__":
    asyncio.run(main())
