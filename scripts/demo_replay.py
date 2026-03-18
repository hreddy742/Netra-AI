"""
Netra AI — Demo replay script.

Pushes synthetic incidents + behavior events into Redis for investor demos.
Simulates a busy retail store with 3 cameras, realistic theft patterns.

Usage:
    python scripts/demo_replay.py --duration 300 --rate 4
    python scripts/demo_replay.py --once       # single burst

Options:
    --duration  seconds to run (default: 300)
    --rate      incidents per minute (default: 3)
    --cameras   comma-separated camera IDs (default: cam-demo-01,cam-demo-02,cam-demo-03)
    --once      push one incident then exit
    --redis-url Redis URL (default: redis://localhost:6379/0)
"""
from __future__ import annotations
import argparse
import asyncio
import json
import random
import time
import uuid

import redis.asyncio as aioredis

STREAM_INCIDENTS = "netra:incidents"
STREAM_BEHAVIOR  = "netra:behavior"

_SEVERITIES   = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
_STAGES       = ["SHELF_INTERACTION", "VISUAL_PICK", "CONCEALMENT",
                 "EXIT_AFTER_CONCEALMENT", "NONSCAN_BAGGING"]
_CONCEALMENT  = ["POCKET", "BAG", "UNDER_SHIRT", "JACKET", "PANTS"]
_FSM_STATES   = ["MONITORING", "SUSPICIOUS", "HIGH_RISK"]

# Weighted severity — makes demos more dramatic
_SEV_WEIGHTS = [5, 25, 45, 25]


def _incident(camera_id: str) -> dict:
    severity = random.choices(_SEVERITIES, weights=_SEV_WEIGHTS, k=1)[0]
    stage    = random.choice(_STAGES)
    risk     = round(random.uniform(0.65, 0.98), 3)
    return {
        "incident_id":      str(uuid.uuid4()),
        "camera_id":        camera_id,
        "track_id":         str(random.randint(1, 30)),
        "risk_score":       str(risk),
        "fsm_score":        str(round(random.uniform(0.5, 1.0), 3)),
        "shopformer_score": str(round(random.uniform(0.4, 0.9), 3)),
        "severity":         severity,
        "theft_stage":      stage,
        "concealment_type": random.choice(_CONCEALMENT),
        "timestamp":        str(time.time()),
        "store_id":         "demo-store",
        "org_id":           "demo-org",
        "_demo":            "true",
    }


def _behavior(camera_id: str) -> dict:
    return {
        "camera_id":   camera_id,
        "track_id":    str(random.randint(1, 30)),
        "fsm_state":   random.choice(_FSM_STATES),
        "risk_score":  str(round(random.uniform(0.0, 1.0), 3)),
        "timestamp":   str(time.time()),
        "_demo":       "true",
    }


async def replay(
    redis_url: str,
    duration_sec: int,
    rate_per_minute: float,
    cameras: list[str],
    once: bool = False,
) -> None:
    r = aioredis.Redis.from_url(redis_url, decode_responses=True)
    await r.ping()
    print(f"Connected to Redis: {redis_url}")

    if once:
        cam = random.choice(cameras)
        inc = _incident(cam)
        await r.xadd(STREAM_INCIDENTS, inc, maxlen=500, approximate=True)
        for _ in range(5):
            await r.xadd(STREAM_BEHAVIOR, _behavior(cam), maxlen=2000, approximate=True)
        print(f"Pushed 1 incident ({inc['severity']}) on {cam}")
        await r.aclose()
        return

    interval    = 60.0 / max(0.1, rate_per_minute)
    deadline    = time.time() + duration_sec
    total_inc   = 0
    total_bev   = 0

    print(f"Starting demo replay: {rate_per_minute}/min, {duration_sec}s, cameras={cameras}")

    while time.time() < deadline:
        await asyncio.sleep(interval)
        cam = random.choice(cameras)

        # Incident
        inc = _incident(cam)
        await r.xadd(STREAM_INCIDENTS, inc, maxlen=500, approximate=True)
        total_inc += 1

        # Behavior events (make the WS feed lively)
        bev_count = random.randint(3, 8)
        for _ in range(bev_count):
            await r.xadd(STREAM_BEHAVIOR, _behavior(cam), maxlen=2000, approximate=True)
        total_bev += bev_count

        elapsed = int(time.time() - (deadline - duration_sec))
        print(
            f"  [{elapsed:>4}s] incident={inc['severity']:<8} cam={cam}  "
            f"total: {total_inc} incidents, {total_bev} behavior"
        )

    await r.aclose()
    print(f"\nDone. Pushed {total_inc} incidents, {total_bev} behavior events in {duration_sec}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Netra AI demo replay")
    parser.add_argument("--duration",  type=int,   default=300,    help="Seconds to run")
    parser.add_argument("--rate",      type=float, default=3.0,    help="Incidents per minute")
    parser.add_argument("--cameras",   type=str,
                        default="cam-demo-01,cam-demo-02,cam-demo-03",
                        help="Comma-separated camera IDs")
    parser.add_argument("--once",      action="store_true",         help="Push single burst")
    parser.add_argument("--redis-url", type=str,
                        default="redis://localhost:6379/0",
                        help="Redis URL")
    args = parser.parse_args()
    cameras = [c.strip() for c in args.cameras.split(",")]
    asyncio.run(replay(args.redis_url, args.duration, args.rate, cameras, args.once))


if __name__ == "__main__":
    main()
