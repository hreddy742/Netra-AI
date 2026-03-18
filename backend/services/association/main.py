"""
Netra AI — association service.

Consumes: netra:detection stream
Produces: netra:association stream
"""
from __future__ import annotations

import asyncio
import signal
from collections import defaultdict
from typing import Any

import redis.asyncio as aioredis

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from netra.events import (
    STREAM_DETECTION, STREAM_ASSOCIATION,
    decode_detection_event, encode_association_event,
)
from netra.redis_keys import service_heartbeat
from backend.services.association.engine import AssociationEngine

log      = get_logger("association")
settings = get_settings()

_GROUP   = "association-group"
_CONSUMER = "association-worker-0"


async def create_group_if_needed(redis: aioredis.Redis) -> None:
    try:
        await redis.xgroup_create(STREAM_DETECTION, _GROUP, id="0", mkstream=True)
    except Exception:
        pass  # group already exists


async def run_consumer(redis: aioredis.Redis) -> None:
    engines: dict[str, AssociationEngine] = {}

    await create_group_if_needed(redis)
    log.info("Association consumer started")

    while True:
        msgs = await redis.xreadgroup(
            groupname  = _GROUP,
            consumername = _CONSUMER,
            streams    = {STREAM_DETECTION: ">"},
            count      = 20,
            block      = 100,
        )
        if not msgs:
            continue

        for _stream, entries in msgs:
            for msg_id, data in entries:
                try:
                    camera_id, persons, items, frame_seq, timestamp = decode_detection_event(data)
                    engine = engines.setdefault(camera_id, AssociationEngine(camera_id))
                    assoc_events = engine.process_frame(persons, items, timestamp)

                    for ev in assoc_events:
                        payload = encode_association_event(ev)
                        await redis.xadd(STREAM_ASSOCIATION, payload, maxlen=1000, approximate=True)

                    await redis.xack(STREAM_DETECTION, _GROUP, msg_id)
                except Exception as e:
                    log.error(f"Association processing error: {e}")
                    await redis.xack(STREAM_DETECTION, _GROUP, msg_id)


async def heartbeat_loop(redis: aioredis.Redis) -> None:
    key = service_heartbeat("association")
    while True:
        await redis.set(key, "ok", ex=15)
        await asyncio.sleep(5)


async def main() -> None:
    r = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=True,
    )
    stop_event = asyncio.Event()
    loop       = asyncio.get_event_loop()

    def _sig(*_: Any) -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGTERM, _sig)
    loop.add_signal_handler(signal.SIGINT, _sig)

    asyncio.create_task(heartbeat_loop(r))
    asyncio.create_task(run_consumer(r))

    await stop_event.wait()
    await r.aclose()
    log.info("association stopped")


if __name__ == "__main__":
    asyncio.run(main())
