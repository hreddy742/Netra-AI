"""
IncidentCorrelator — links incidents across adjacent cameras.

When incident fires on cam-A, checks if there is a recent high-risk track
on any adjacent camera (from camera adjacency config). Returns correlated
incidents as a group for the operator dashboard.

Reads adjacency from Redis camera config hash (key: "adjacent_cameras").
"""
from __future__ import annotations
import json
import time
from typing import Any
import asyncpg
import redis.asyncio as aioredis


_CORRELATION_WINDOW_SEC = 120   # incidents within 2 min of each other


class IncidentCorrelator:

    def __init__(self, pool: asyncpg.Pool, redis: aioredis.Redis) -> None:
        self._pool  = pool
        self._redis = redis

    async def correlated(self, incident_id: str) -> list[dict[str, Any]]:
        """
        Given an incident ID, return list of incidents on adjacent cameras
        within the correlation time window.
        """
        async with self._pool.acquire() as conn:
            inc = await conn.fetchrow(
                "SELECT * FROM incidents WHERE id=$1", incident_id
            )
        if not inc:
            return []

        adj_raw = await self._redis.hget(
            f"netra:config:camera:{inc['camera_id']}", "adjacent_cameras"
        )
        adjacent = json.loads(adj_raw) if adj_raw else []
        if not adjacent:
            return []

        ts = inc["occurred_at"]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM incidents
                   WHERE camera_id = ANY($1::text[])
                     AND ABS(EXTRACT(EPOCH FROM (occurred_at - $2))) < $3
                     AND id != $4
                   ORDER BY occurred_at""",
                adjacent, ts, _CORRELATION_WINDOW_SEC, incident_id,
            )
        return [dict(r) for r in rows]

    async def groups(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent correlated incident groups (multi-camera clusters)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT i.id, i.camera_id, i.risk_score, i.severity, i.occurred_at,
                          i.theft_stage, i.concealment_type
                   FROM incidents i
                   WHERE i.status IN ('OPEN','CONFIRMED')
                   ORDER BY i.occurred_at DESC
                   LIMIT $1""",
                limit * 5,
            )
        # Simple greedy grouping: cluster incidents within 2-min window
        groups: list[list[dict]] = []
        used: set[str] = set()
        incidents = [dict(r) for r in rows]
        for inc in incidents:
            if inc["id"] in used:
                continue
            group = [inc]
            used.add(inc["id"])
            for other in incidents:
                if other["id"] in used:
                    continue
                dt = abs((inc["occurred_at"] - other["occurred_at"]).total_seconds())
                if dt <= _CORRELATION_WINDOW_SEC and other["camera_id"] != inc["camera_id"]:
                    group.append(other)
                    used.add(other["id"])
            if len(group) > 1:
                groups.append(group)
            if len(groups) >= limit:
                break
        return [{"incidents": g, "camera_count": len({x["camera_id"] for x in g})} for g in groups]
