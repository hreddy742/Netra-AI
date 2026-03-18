"""
Netra AI — persistence service.

Consumes: netra:incidents stream
Writes: PostgreSQL via asyncpg
"""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone
from typing import Any

import asyncpg
import redis.asyncio as aioredis

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from netra.events import STREAM_INCIDENTS, decode_incident_event
from netra.redis_keys import service_heartbeat, latest_frame_jpeg

log      = get_logger("persistence")
settings = get_settings()
_GROUP   = "persistence-group"
_CONSUMER = "persistence-worker-0"


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Apply schema.sql if tables don't exist."""
    import os
    schema_path = os.path.join(os.path.dirname(__file__), "../../../database/schema.sql")
    try:
        with open(schema_path) as f:
            sql = f.read()
        async with pool.acquire() as conn:
            await conn.execute(sql)
        log.info("Schema applied")
    except FileNotFoundError:
        log.warning(f"Schema file not found at {schema_path} — skipping")


async def create_group(redis: aioredis.Redis) -> None:
    try:
        await redis.xgroup_create(STREAM_INCIDENTS, _GROUP, id="0", mkstream=True)
    except Exception:
        pass


async def write_incident(conn: asyncpg.Connection, incident: Any) -> None:
    occurred = datetime.fromtimestamp(incident.timestamp, tz=timezone.utc)

    # Upsert (re-processing safety)
    await conn.execute(
        """
        INSERT INTO incidents
            (id, camera_id, track_id, risk_score, fsm_score, shopformer_score,
             theft_stage, concealment_type, severity, status,
             model_version, logic_version, occurred_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
        ON CONFLICT (id) DO NOTHING
        """,
        incident.incident_id,
        incident.camera_id,
        incident.track_id,
        incident.risk_score,
        incident.fsm_score,
        incident.shopformer_score,
        incident.theft_stage.value,
        incident.concealment_type.value,
        incident.severity.value,
        incident.status.value,
        incident.model_version,
        incident.logic_version,
        occurred,
    )

    # Ensure camera row exists (from config; insert placeholder if not yet configured)
    await conn.execute(
        """
        INSERT INTO cameras (id, display_name)
        VALUES ($1, $1)
        ON CONFLICT (id) DO NOTHING
        """,
        incident.camera_id,
    )


async def capture_evidence(
    incident_id: str,
    camera_id: str,
    jpeg_bytes: bytes,
    base_path: str,
    conn: asyncpg.Connection,
) -> None:
    """Fire-and-forget evidence frame capture. Never awaited directly — use create_task()."""
    try:
        import cv2
        import numpy as np
        from pathlib import Path

        # Blur check: skip frames with Laplacian variance < threshold
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < settings.evidence_blur_threshold:
            log.debug("Skipping blurry evidence frame for %s", incident_id)
            return

        path = f"{base_path}/{incident_id}/frame_001.jpg"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: open(path, "wb").write(jpeg_bytes))

        occurred = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp(), tz=timezone.utc
        )
        await conn.execute(
            "INSERT INTO evidence_frames (incident_id, frame_path, frame_seq, occurred_at) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
            incident_id, path, 1, occurred,
        )
    except Exception as e:
        log.error("Evidence capture failed for %s: %s", incident_id, e)
        # Never re-raise — must not affect the main pipeline


async def run(redis: aioredis.Redis, redis_bin: aioredis.Redis, pool: asyncpg.Pool) -> None:
    await create_group(redis)
    log.info("Persistence consumer started")

    while True:
        msgs = await redis.xreadgroup(
            groupname=_GROUP, consumername=_CONSUMER,
            streams={STREAM_INCIDENTS: ">"}, count=20, block=200,
        )
        if not msgs:
            continue

        for _s, entries in msgs:
            for mid, data in entries:
                try:
                    incident = decode_incident_event(data)
                    async with pool.acquire() as conn:
                        await write_incident(conn, incident)
                        # Fire-and-forget evidence capture (never blocks main pipeline)
                        jpeg = await redis_bin.get(latest_frame_jpeg(incident.camera_id))
                        if jpeg:
                            asyncio.create_task(capture_evidence(
                                incident.incident_id, incident.camera_id,
                                jpeg, settings.evidence_dir, conn,
                            ))
                    await redis.xack(STREAM_INCIDENTS, _GROUP, mid)
                    log.debug(f"Persisted incident {incident.incident_id[:8]}")
                except Exception as e:
                    log.error(f"Persistence error: {e}")
                    await redis.xack(STREAM_INCIDENTS, _GROUP, mid)


async def heartbeat_loop(redis: aioredis.Redis) -> None:
    key = service_heartbeat("persistence")
    while True:
        await redis.set(key, "ok", ex=15)
        await asyncio.sleep(5)


async def main() -> None:
    r = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=True,
    )
    # Separate binary connection for reading JPEG frames
    r_bin = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=False,
    )

    pool = await asyncpg.create_pool(
        host     = settings.postgres_host,
        port     = settings.postgres_port,
        database = settings.postgres_db,
        user     = settings.postgres_user,
        password = settings.postgres_password,
        min_size = 2,
        max_size = 10,
    )
    await ensure_schema(pool)

    stop_event = asyncio.Event()
    loop       = asyncio.get_event_loop()

    def _sig(*_: Any) -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGTERM, _sig)
    loop.add_signal_handler(signal.SIGINT, _sig)

    asyncio.create_task(heartbeat_loop(r))
    asyncio.create_task(run(r, r_bin, pool))

    await stop_event.wait()
    await pool.close()
    await r.aclose()
    log.info("persistence stopped")


if __name__ == "__main__":
    asyncio.run(main())
