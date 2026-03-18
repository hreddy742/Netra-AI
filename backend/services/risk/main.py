"""
Netra AI — risk service.

Consumes: netra:behavior + netra:shopformer
Produces: netra:incidents
"""
from __future__ import annotations

import asyncio
import signal
from typing import Any

import redis.asyncio as aioredis

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from netra.events import (
    STREAM_BEHAVIOR, STREAM_SHOPFORMER, STREAM_INCIDENTS,
    decode_behavior_event, decode_shopformer_score, encode_incident_event,
)
from netra.redis_keys import service_heartbeat, track_risk
from backend.services.risk.engine import RiskEngine

log      = get_logger("risk")
settings = get_settings()

_GROUP_BEH  = "risk-beh-group"
_GROUP_SF   = "risk-sf-group"
_CONSUMER   = "risk-worker-0"


async def create_groups(redis: aioredis.Redis) -> None:
    for stream, group in [
        (STREAM_BEHAVIOR,   _GROUP_BEH),
        (STREAM_SHOPFORMER, _GROUP_SF),
    ]:
        try:
            await redis.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass


async def run(redis: aioredis.Redis) -> None:
    engines: dict[str, RiskEngine] = {}

    await create_groups(redis)
    log.info("Risk consumer started")
    import time
    last_prune = time.time()

    while True:
        # Drain ShopFormer scores first (update cached state)
        sf_msgs = await redis.xreadgroup(
            groupname=_GROUP_SF, consumername=_CONSUMER,
            streams={STREAM_SHOPFORMER: ">"}, count=20, block=0,
        )
        if sf_msgs:
            for _s, entries in sf_msgs:
                for mid, data in entries:
                    try:
                        score = decode_shopformer_score(data)
                        eng   = engines.setdefault(score.camera_id, RiskEngine(score.camera_id))
                        eng.update_shopformer(score)
                        await redis.xack(STREAM_SHOPFORMER, _GROUP_SF, mid)
                    except Exception as e:
                        log.warning(f"Bad shopformer score: {e}")

        # Process behavior events
        beh_msgs = await redis.xreadgroup(
            groupname=_GROUP_BEH, consumername=_CONSUMER,
            streams={STREAM_BEHAVIOR: ">"}, count=20, block=100,
        )
        if not beh_msgs:
            # Periodic prune
            now = time.time()
            if now - last_prune > 60:
                for eng in engines.values():
                    eng.prune_stale()
                last_prune = now
            continue

        for _s, entries in beh_msgs:
            for mid, data in entries:
                try:
                    bev = decode_behavior_event(data)
                    eng = engines.setdefault(bev.camera_id, RiskEngine(bev.camera_id))

                    incident = eng.update_behavior(bev)

                    # Publish live risk score for streaming overlay
                    await redis.set(
                        track_risk(bev.camera_id, bev.track_id),
                        str(round(eng._get(bev.track_id).last_risk_score, 4)),
                        ex=10,
                    )

                    if incident is not None:
                        payload = encode_incident_event(incident)
                        await redis.xadd(STREAM_INCIDENTS, payload, maxlen=2000, approximate=True)

                    await redis.xack(STREAM_BEHAVIOR, _GROUP_BEH, mid)
                except Exception as e:
                    log.error(f"Risk processing error: {e}")
                    await redis.xack(STREAM_BEHAVIOR, _GROUP_BEH, mid)


async def heartbeat_loop(redis: aioredis.Redis) -> None:
    key = service_heartbeat("risk")
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
    log.info("risk stopped")


if __name__ == "__main__":
    asyncio.run(main())
