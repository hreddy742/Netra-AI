"""
Netra AI — Camera load simulator.

Pushes synthetic FramePointers to Redis at a configurable FPS per camera.
Used for load testing without real RTSP cameras.

Usage:
    python -m tests.load.camera_simulator --cameras 50 --fps 10 --duration 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import redis.asyncio as aioredis


async def simulate(n_cameras: int, fps: float, duration: float, redis_url: str) -> None:
    r = aioredis.Redis.from_url(redis_url, decode_responses=True)
    interval = 1.0 / fps
    deadline = time.monotonic() + duration
    camera_ids = [f"sim-cam-{i:03d}" for i in range(n_cameras)]

    # Register cameras
    await r.sadd("netra:cameras:active", *camera_ids)

    frame_seq: dict[str, int] = {c: 0 for c in camera_ids}
    frames_sent = 0
    print(f"Simulating {n_cameras} cameras @ {fps} FPS for {duration}s")

    while time.monotonic() < deadline:
        t0 = time.monotonic()
        for cam_id in camera_ids:
            frame_seq[cam_id] += 1
            fp = {
                "camera_id":  cam_id,
                "shm_name":   f"netra_cam_{cam_id}_ring",
                "slot_index": str(frame_seq[cam_id] % 8),
                "generation": str(frame_seq[cam_id]),
                "width":      "1280",
                "height":     "720",
                "channels":   "3",
                "timestamp":  str(time.time()),
                "frame_seq":  str(frame_seq[cam_id]),
            }
            await r.lpush(f"netra:queue:frames:{cam_id}", json.dumps(fp))
            await r.ltrim(f"netra:queue:frames:{cam_id}", 0, 29)
            frames_sent += 1

        elapsed = time.monotonic() - t0
        sleep_for = max(0.0, interval - elapsed)
        await asyncio.sleep(sleep_for)

    # Cleanup
    await r.srem("netra:cameras:active", *camera_ids)
    await r.aclose()
    print(f"Done. Sent {frames_sent} frame pointers total.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Netra camera load simulator")
    parser.add_argument("--cameras",  type=int,   default=10)
    parser.add_argument("--fps",      type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--redis",    type=str,   default="redis://localhost:6379/0")
    args = parser.parse_args()
    asyncio.run(simulate(args.cameras, args.fps, args.duration, args.redis))


if __name__ == "__main__":
    main()
