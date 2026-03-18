"""
Netra AI — behavior service.

Consumes: netra:detection (persons+poses) + netra:association (events)
Produces: netra:behavior stream
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
    STREAM_DETECTION, STREAM_ASSOCIATION, STREAM_BEHAVIOR,
    decode_detection_event, decode_association_event, encode_behavior_event,
)
from netra.redis_keys import service_heartbeat
from backend.services.behavior.fsm import TheftRiskFSM
from backend.services.behavior.confirmation import SignalConfirmationBuffer

log      = get_logger("behavior")
settings = get_settings()

_GROUP_DET   = "behavior-det-group"
_GROUP_ASSOC = "behavior-assoc-group"
_CONSUMER    = "behavior-worker-0"


async def create_groups(redis: aioredis.Redis) -> None:
    for stream, group in [
        (STREAM_DETECTION,   _GROUP_DET),
        (STREAM_ASSOCIATION, _GROUP_ASSOC),
    ]:
        try:
            await redis.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass


async def run(redis: aioredis.Redis) -> None:
    fsms: dict[str, TheftRiskFSM] = {}
    confirmers: dict[str, SignalConfirmationBuffer] = {}
    # Buffer recent association events per camera (last 2 seconds of event time)
    assoc_buffer: dict[str, list] = defaultdict(list)
    # Track latest event_time seen per camera (used for stale-buffer cleanup)
    latest_event_ts: dict[str, float] = {}

    await create_groups(redis)
    log.info("Behavior consumer started")

    while True:
        # Drain any pending association events first (they're lightweight)
        assoc_msgs = await redis.xreadgroup(
            groupname=_GROUP_ASSOC, consumername=_CONSUMER,
            streams={STREAM_ASSOCIATION: ">"}, count=50, block=0,
        )
        if assoc_msgs:
            for _stream, entries in assoc_msgs:
                for msg_id, data in entries:
                    try:
                        ev = decode_association_event(data)
                        assoc_buffer[ev.camera_id].append(ev)
                        await redis.xack(STREAM_ASSOCIATION, _GROUP_ASSOC, msg_id)
                    except Exception as e:
                        log.warning(f"Bad association event: {e}")

        # Process detection events (which drive the FSM)
        det_msgs = await redis.xreadgroup(
            groupname=_GROUP_DET, consumername=_CONSUMER,
            streams={STREAM_DETECTION: ">"}, count=10, block=100,
        )
        if not det_msgs:
            # Clean stale assoc buffer entries using event time (not wall clock).
            # Under Redis queue backlog, wall clock would be ~10s ahead of event
            # timestamps and would discard ALL buffered events — blinding the FSM.
            for cam, buf in list(assoc_buffer.items()):
                ref_ts = latest_event_ts.get(cam)
                if ref_ts is not None:
                    assoc_buffer[cam] = [e for e in buf if ref_ts - e.timestamp < 2.0]
            continue

        for _stream, entries in det_msgs:
            for msg_id, data in entries:
                try:
                    camera_id, persons, items, frame_seq, timestamp = decode_detection_event(data)

                    # Update per-camera watermark (latest event time seen)
                    if timestamp > latest_event_ts.get(camera_id, 0.0):
                        latest_event_ts[camera_id] = timestamp
                    elif timestamp < latest_event_ts.get(camera_id, 0.0) - 5.0:
                        log.warning("Out-of-order event cam=%s lag=%.1fs",
                                    camera_id, latest_event_ts[camera_id] - timestamp)

                    fsm = fsms.setdefault(camera_id, TheftRiskFSM(camera_id))

                    # Take buffered assoc events for this camera, flush buffer
                    assoc = assoc_buffer.pop(camera_id, [])
                    confirmer = confirmers.setdefault(camera_id, SignalConfirmationBuffer())
                    confirmed_assoc = confirmer.filter(assoc, camera_id, event_time=timestamp)

                    behavior_events = fsm.process(persons, confirmed_assoc, timestamp)

                    for bev in behavior_events:
                        payload = encode_behavior_event(bev)
                        await redis.xadd(STREAM_BEHAVIOR, payload, maxlen=2000, approximate=True)

                    await redis.xack(STREAM_DETECTION, _GROUP_DET, msg_id)
                except Exception as e:
                    log.error(f"Behavior processing error: {e}")
                    await redis.xack(STREAM_DETECTION, _GROUP_DET, msg_id)


async def heartbeat_loop(redis: aioredis.Redis) -> None:
    key = service_heartbeat("behavior")
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
    asyncio.create_task(run(r))

    await stop_event.wait()
    await r.aclose()
    log.info("behavior stopped")


if __name__ == "__main__":
    asyncio.run(main())
