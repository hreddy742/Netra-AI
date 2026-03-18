"""
Netra AI — Demo mode.

Provides:
  - is_demo_mode() → bool — True when settings.demo_mode=True or NETRA_DEMO=1
  - SyntheticEventGenerator — pushes realistic synthetic incidents into Redis
    at configured rate for investor/sales demos

Synthetic events are clearly tagged with {"_demo": "true"} to prevent them
from appearing in real analytics or DB.

Usage:
    gen = SyntheticEventGenerator(redis)
    await gen.start()   # begins background loop
    await gen.stop()
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

from backend.core.logging import get_logger
from backend.core.settings import get_settings
from netra.events import STREAM_INCIDENTS, STREAM_BEHAVIOR

log      = get_logger("demo_mode")
settings = get_settings()

_DEMO_CAMERAS = ["cam-demo-01", "cam-demo-02", "cam-demo-03"]
_DEMO_SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
_DEMO_STAGES = [
    "SHELF_INTERACTION", "VISUAL_PICK", "CONCEALMENT",
    "EXIT_AFTER_CONCEALMENT", "NONSCAN_BAGGING",
]
_DEMO_CONCEALMENT = [
    "POCKET", "BAG", "UNDER_SHIRT", "JACKET", "PANTS",
]


def is_demo_mode() -> bool:
    return settings.demo_mode or os.environ.get("NETRA_DEMO", "0") == "1"


def _synthetic_incident() -> dict[str, Any]:
    cam      = random.choice(_DEMO_CAMERAS)
    severity = random.choice(_DEMO_SEVERITIES)
    stage    = random.choice(_DEMO_STAGES)
    conceal  = random.choice(_DEMO_CONCEALMENT)
    risk     = round(random.uniform(0.65, 0.98), 3)
    fsm_sc   = round(random.uniform(0.5, 1.0), 3)
    sf_sc    = round(random.uniform(0.4, 0.9), 3)
    return {
        "incident_id":    str(uuid.uuid4()),
        "camera_id":      cam,
        "track_id":       str(random.randint(1, 50)),
        "risk_score":     str(risk),
        "fsm_score":      str(fsm_sc),
        "shopformer_score": str(sf_sc),
        "severity":       severity,
        "theft_stage":    stage,
        "concealment_type": conceal,
        "timestamp":      str(time.time()),
        "store_id":       "demo-store",
        "org_id":         "demo-org",
        "_demo":          "true",
    }


def _synthetic_behavior(camera_id: str) -> dict[str, Any]:
    return {
        "camera_id":   camera_id,
        "track_id":    str(random.randint(1, 50)),
        "fsm_state":   random.choice(["MONITORING", "SUSPICIOUS", "HIGH_RISK"]),
        "risk_score":  str(round(random.uniform(0.0, 1.0), 3)),
        "timestamp":   str(time.time()),
        "_demo":       "true",
    }


class SyntheticEventGenerator:
    """
    Pushes synthetic incidents + behavior events to Redis at configured rate.
    All events are tagged with "_demo": "true".
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._running = False

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._generator_loop(), name="demo-generator")
        log.info("SyntheticEventGenerator started (%d events/min)", settings.demo_events_per_minute)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("SyntheticEventGenerator stopped")

    @property
    def is_running(self) -> bool:
        return self._running and bool(self._task) and not self._task.done()

    async def _generator_loop(self) -> None:
        interval = 60.0 / max(1, settings.demo_events_per_minute)
        try:
            while self._running:
                await asyncio.sleep(interval)
                try:
                    incident = _synthetic_incident()
                    await self._redis.xadd(
                        STREAM_INCIDENTS, incident, maxlen=500, approximate=True
                    )
                    # Also push 3-5 behavior events to make the WS feed lively
                    cam = incident["camera_id"]
                    for _ in range(random.randint(3, 5)):
                        bev = _synthetic_behavior(cam)
                        await self._redis.xadd(
                            STREAM_BEHAVIOR, bev, maxlen=2000, approximate=True
                        )
                except Exception as exc:
                    log.debug("Demo generator error: %s", exc)
        except asyncio.CancelledError:
            pass
