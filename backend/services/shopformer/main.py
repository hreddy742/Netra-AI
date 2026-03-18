"""
Netra AI — ShopFormer service.

Consumes: netra:behavior (only suspicious=True tracks)
Maintains: per-track pose sequence buffer (Redis sorted set)
Produces: netra:shopformer scores
"""
from __future__ import annotations

import asyncio
import json
import signal
import time
from collections import defaultdict, deque
from typing import Any

import redis.asyncio as aioredis

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from netra.events import (
    STREAM_BEHAVIOR, STREAM_SHOPFORMER,
    decode_behavior_event, encode_shopformer_score,
)
from netra.redis_keys import service_heartbeat
from netra.types import ShopFormerScore
from backend.services.shopformer.model import ShopFormerInference

log      = get_logger("shopformer")
settings = get_settings()

_GROUP    = "shopformer-group"
_CONSUMER = "shopformer-worker-0"


class PoseBuffer:
    """
    Maintains a sliding window of pose keypoints per track.
    Triggers ShopFormer when the buffer is full (seq_len frames).
    """

    def __init__(self, seq_len: int, stride: int) -> None:
        self.seq_len = seq_len
        self.stride  = stride
        # track_key → deque of (timestamp, [[x,y], ...])
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=seq_len))
        self._last_scored: dict[str, float] = {}

    def push(self, camera_id: str, track_id: int, keypoints: list, timestamp: float) -> None:
        key = f"{camera_id}:{track_id}"
        # keypoints: list of Keypoint-encoded [[x, y, conf], ...]
        # We extract only (x, y) for ShopFormer
        xy = [[kp[0], kp[1]] for kp in keypoints]
        self._buffers[key].append((timestamp, xy))

    def ready(self, camera_id: str, track_id: int) -> bool:
        key = f"{camera_id}:{track_id}"
        buf = self._buffers[key]
        if len(buf) < self.seq_len:
            return False
        # Rate-limit: don't re-score more often than every stride/fps seconds
        last = self._last_scored.get(key, 0.0)
        return (time.time() - last) >= (self.stride / 10.0)   # assume ~10 fps

    def get_sequence(self, camera_id: str, track_id: int) -> list[list[list[float]]]:
        key = f"{camera_id}:{track_id}"
        self._last_scored[key] = time.time()
        return [frame_xy for _, frame_xy in self._buffers[key]]

    def evict_stale(self, max_age_sec: float = 30.0) -> None:
        now = time.time()
        stale = [k for k, buf in self._buffers.items()
                 if buf and now - buf[-1][0] > max_age_sec]
        for k in stale:
            del self._buffers[k]
            self._last_scored.pop(k, None)


async def create_group(redis: aioredis.Redis) -> None:
    try:
        await redis.xgroup_create(STREAM_BEHAVIOR, _GROUP, id="0", mkstream=True)
    except Exception:
        pass


async def run(redis: aioredis.Redis, model: ShopFormerInference) -> None:
    pose_buffer = PoseBuffer(settings.shopformer_seq_len, settings.shopformer_stride)
    await create_group(redis)
    log.info("ShopFormer consumer started")

    last_evict = time.time()

    while True:
        msgs = await redis.xreadgroup(
            groupname=_GROUP, consumername=_CONSUMER,
            streams={STREAM_BEHAVIOR: ">"}, count=20, block=200,
        )
        if not msgs:
            # Periodic buffer eviction
            if time.time() - last_evict > 60:
                pose_buffer.evict_stale()
                last_evict = time.time()
            continue

        for _stream, entries in msgs:
            for msg_id, data in entries:
                try:
                    bev = decode_behavior_event(data)

                    # Only process suspicious tracks
                    if not bev.suspicious:
                        await redis.xack(STREAM_BEHAVIOR, _GROUP, msg_id)
                        continue

                    # We need keypoints from the behavior event (carried in signals metadata
                    # not directly stored — behavior service would need to pass them.
                    # For now we pull them from the detection stream via separate path.
                    # The behavior event carries fsm signals but not raw keypoints.
                    # ShopFormer only needs keypoints — we handle this via a separate
                    # Redis key written by inference service.
                    kp_raw = await redis.get(f"netra:kp:{bev.camera_id}:{bev.track_id}")
                    if kp_raw:
                        keypoints = json.loads(kp_raw)
                        pose_buffer.push(bev.camera_id, bev.track_id, keypoints, bev.timestamp)

                    if pose_buffer.ready(bev.camera_id, bev.track_id):
                        sequence = pose_buffer.get_sequence(bev.camera_id, bev.track_id)

                        loop = asyncio.get_event_loop()
                        anomaly_score, recon_error, embedding = await loop.run_in_executor(
                            None, model.score, sequence
                        )

                        score_obj = ShopFormerScore(
                            camera_id            = bev.camera_id,
                            track_id             = bev.track_id,
                            anomaly_score        = anomaly_score,
                            reconstruction_error = recon_error,
                            embedding            = embedding,
                            pose_sequence_len    = len(sequence),
                            timestamp            = bev.timestamp,
                        )
                        payload = encode_shopformer_score(score_obj)
                        await redis.xadd(STREAM_SHOPFORMER, payload, maxlen=500, approximate=True)

                        log.debug(
                            f"ShopFormer scored {bev.camera_id}:{bev.track_id} "
                            f"anomaly={anomaly_score:.3f}"
                        )

                    await redis.xack(STREAM_BEHAVIOR, _GROUP, msg_id)

                except Exception as e:
                    log.error(f"ShopFormer error: {e}")
                    await redis.xack(STREAM_BEHAVIOR, _GROUP, msg_id)


async def heartbeat_loop(redis: aioredis.Redis) -> None:
    key = service_heartbeat("shopformer")
    while True:
        await redis.set(key, "ok", ex=15)
        await asyncio.sleep(5)


async def main() -> None:
    r = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=True,
    )
    model = ShopFormerInference()
    model.load()

    stop_event = asyncio.Event()
    loop       = asyncio.get_event_loop()

    def _sig(*_: Any) -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGTERM, _sig)
    loop.add_signal_handler(signal.SIGINT, _sig)

    asyncio.create_task(heartbeat_loop(r))
    asyncio.create_task(run(r, model))

    await stop_event.wait()
    await r.aclose()
    log.info("shopformer stopped")


if __name__ == "__main__":
    asyncio.run(main())
