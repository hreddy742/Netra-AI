"""
ModelRegistry — versioned model tracking with DB-backed deployment history.

Tracks:
  - model_name     : "shopformer" | "detector" | "pose"
  - model_version  : semantic version e.g. "1.2.0"
  - logic_version  : FSM logic version
  - path           : local path to model weights
  - deployed_at    : timestamp
  - is_active      : only one active version per model_name
  - notes          : deployment notes

Usage:
    registry = ModelRegistry(pool)
    await registry.deploy("shopformer", "1.2.0", "/models/shopformer/v1.2.pth", notes="retrained on 500 samples")
    active = await registry.active("shopformer")
    history = await registry.history("shopformer")
    await registry.rollback("shopformer")  # reactivate previous version
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import asyncpg

from backend.core.logging import get_logger
from backend.core.settings import get_settings

log      = get_logger("model_registry")
settings = get_settings()


class ModelRegistry:

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def deploy(
        self,
        model_name: str,
        model_version: str,
        path: str,
        logic_version: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Register a new model deployment as the active version.
        Previous active version for this model_name is deactivated.
        """
        lv = logic_version or settings.logic_version
        async with self._pool.acquire() as conn:
            # Deactivate previous
            await conn.execute(
                """UPDATE model_deployments SET is_active = FALSE
                   WHERE model_name = $1 AND is_active = TRUE""",
                model_name,
            )
            row = await conn.fetchrow(
                """INSERT INTO model_deployments
                   (model_name, model_version, logic_version, path, notes, is_active, deployed_at)
                   VALUES ($1, $2, $3, $4, $5, TRUE, NOW())
                   RETURNING *""",
                model_name, model_version, lv, path, notes,
            )
        log.info("Deployed %s v%s → %s", model_name, model_version, path)
        return dict(row)

    async def active(self, model_name: str) -> dict[str, Any] | None:
        """Return the currently active deployment for a model."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM model_deployments WHERE model_name=$1 AND is_active=TRUE",
                model_name,
            )
        return dict(row) if row else None

    async def history(self, model_name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return deployment history for a model, newest first."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM model_deployments
                   WHERE model_name=$1
                   ORDER BY deployed_at DESC LIMIT $2""",
                model_name, limit,
            )
        return [dict(r) for r in rows]

    async def rollback(self, model_name: str) -> dict[str, Any] | None:
        """
        Rollback to the previous version of a model.
        Deactivates current active, activates most recent inactive.
        """
        async with self._pool.acquire() as conn:
            # Deactivate current
            await conn.execute(
                "UPDATE model_deployments SET is_active=FALSE WHERE model_name=$1 AND is_active=TRUE",
                model_name,
            )
            # Find previous
            prev = await conn.fetchrow(
                """SELECT * FROM model_deployments
                   WHERE model_name=$1 AND is_active=FALSE
                   ORDER BY deployed_at DESC LIMIT 1""",
                model_name,
            )
            if not prev:
                log.warning("No previous version found for %s rollback", model_name)
                return None
            await conn.execute(
                "UPDATE model_deployments SET is_active=TRUE WHERE id=$1",
                prev["id"],
            )
        log.info("Rolled back %s to v%s", model_name, prev["model_version"])
        return dict(prev)

    async def all_active(self) -> list[dict[str, Any]]:
        """Return all currently active model versions."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM model_deployments WHERE is_active=TRUE ORDER BY model_name"
            )
        return [dict(r) for r in rows]
