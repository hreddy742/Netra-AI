"""
Netra AI — alerting service.

Consumes: netra:incidents
Dispatches: Telegram, Email, SMS, Webhook (based on .env config)
"""
from __future__ import annotations

import asyncio
import signal
from typing import Any

import redis.asyncio as aioredis

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from netra.events import STREAM_INCIDENTS, decode_incident_event
from netra.redis_keys import service_heartbeat
from backend.services.alerting.channels import AlertDispatcher
from backend.services.alerting.cooldown import CameraAlertCooldown
from backend.services.alerting.priority import score_incident, AlertPriority

log      = get_logger("alerting")
settings = get_settings()
_GROUP   = "alerting-group"
_CONSUMER = "alerting-worker-0"


async def create_group(redis: aioredis.Redis) -> None:
    try:
        await redis.xgroup_create(STREAM_INCIDENTS, _GROUP, id="0", mkstream=True)
    except Exception:
        pass


async def run(redis: aioredis.Redis, dispatcher: AlertDispatcher) -> None:
    await create_group(redis)
    # Instantiate once at startup — shares the Lua script registration
    cooldown = CameraAlertCooldown(redis)
    log.info("Alerting consumer started")

    while True:
        msgs = await redis.xreadgroup(
            groupname=_GROUP, consumername=_CONSUMER,
            streams={STREAM_INCIDENTS: ">"}, count=5, block=200,
        )
        if not msgs:
            continue

        for _s, entries in msgs:
            for mid, data in entries:
                try:
                    incident = decode_incident_event(data)
                    burst = incident.severity.value == "CRITICAL"
                    # Atomic check-and-record: suppressed if returns False
                    if not await cooldown.try_dispatch(incident.camera_id, burst=burst):
                        log.debug("Cooldown suppressed incident cam=%s", incident.camera_id)
                        await redis.xack(STREAM_INCIDENTS, _GROUP, mid)
                        continue
                    priority = score_incident(incident)
                    log.info("Alert priority=%s cam=%s risk=%.3f",
                             priority.name, incident.camera_id, incident.risk_score)
                    await dispatcher.dispatch(incident)
                    await redis.xack(STREAM_INCIDENTS, _GROUP, mid)
                except Exception as e:
                    log.error(f"Alert dispatch error: {e}")
                    await redis.xack(STREAM_INCIDENTS, _GROUP, mid)


async def heartbeat_loop(redis: aioredis.Redis) -> None:
    key = service_heartbeat("alerting")
    while True:
        await redis.set(key, "ok", ex=15)
        await asyncio.sleep(5)


async def main() -> None:
    r = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=True,
    )
    dispatcher = AlertDispatcher()
    stop_event = asyncio.Event()
    loop       = asyncio.get_event_loop()

    def _sig(*_: Any) -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGTERM, _sig)
    loop.add_signal_handler(signal.SIGINT, _sig)

    asyncio.create_task(heartbeat_loop(r))
    asyncio.create_task(run(r, dispatcher))

    await stop_event.wait()
    await r.aclose()
    log.info("alerting stopped")


if __name__ == "__main__":
    asyncio.run(main())
