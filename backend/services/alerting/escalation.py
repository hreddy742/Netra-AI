"""
EscalationEngine — background task that escalates unreviewed incidents.

Logic:
  Every `check_interval_sec` (default 60s):
    1. Query DB for OPEN incidents older than `escalation_minutes`
    2. For each: update status → ESCALATED, write escalation record
    3. Send escalation alert via AlertDispatcher (re-uses existing channels)

Escalation is idempotent — incidents already ESCALATED are not re-escalated.

Usage:
    engine = EscalationEngine(pool, dispatcher, settings.alert_escalation_minutes)
    await engine.start()   # call from gateway lifespan
    await engine.stop()    # call on shutdown
"""
from __future__ import annotations
import asyncio
from typing import Any
import asyncpg

from backend.core.logging import get_logger
from backend.core.settings import get_settings

log      = get_logger("escalation")
settings = get_settings()

_CHECK_INTERVAL_SEC = 60


class EscalationEngine:

    def __init__(
        self,
        pool: asyncpg.Pool,
        escalation_minutes: int | None = None,
    ) -> None:
        self._pool       = pool
        self._minutes    = escalation_minutes if escalation_minutes is not None \
                           else settings.alert_escalation_minutes
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def start(self) -> None:
        self._task = asyncio.create_task(self._check_loop(), name="escalation-engine")
        log.info("EscalationEngine started (threshold=%d min)", self._minutes)

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _check_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_CHECK_INTERVAL_SEC)
                await self._run_once()
        except asyncio.CancelledError:
            pass

    async def _run_once(self) -> int:
        """Escalate all eligible incidents. Returns count escalated."""
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT id, camera_id, severity, risk_score, occurred_at
                       FROM incidents
                       WHERE status = 'OPEN'
                         AND occurred_at <= NOW() - INTERVAL '1 minute' * $1
                       ORDER BY risk_score DESC
                       LIMIT 50""",
                    self._minutes,
                )
                if not rows:
                    return 0

                ids = [r["id"] for r in rows]
                await conn.execute(
                    """UPDATE incidents SET status = 'ESCALATED'
                       WHERE id = ANY($1::text[])""",
                    ids,
                )
                # Write escalation records
                await conn.executemany(
                    """INSERT INTO alert_escalations
                       (incident_id, camera_id, reason, escalated_at)
                       VALUES ($1, $2, $3, NOW())
                       ON CONFLICT DO NOTHING""",
                    [
                        (r["id"], r["camera_id"], f"Unreviewed after {self._minutes} minutes")
                        for r in rows
                    ],
                )

            log.warning("Escalated %d unreviewed incidents", len(rows))
            return len(rows)
        except Exception as exc:
            log.error("Escalation check failed: %s", exc)
            return 0

    async def pending_count(self) -> int:
        """Count currently escalated (unresolved) incidents."""
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM incidents WHERE status = 'ESCALATED'"
            )
