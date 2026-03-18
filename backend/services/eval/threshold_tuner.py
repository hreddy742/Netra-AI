"""
AdaptiveThresholdTuner — learns optimal alert threshold per camera from operator labels.

Algorithm:
  1. Pull all reviewed incidents for the camera from last N days
  2. Sweep thresholds from 0.40 → 0.90 in steps of 0.025
  3. For each threshold: count TP (CONFIRMED incidents above threshold),
     FP (DISMISSED above threshold), FN (CONFIRMED below threshold)
  4. Compute F-beta (beta=0.5 → precision-weighted)
  5. Store winning threshold in Redis with TTL=24h

Usage:
    tuner = AdaptiveThresholdTuner(pool, redis)
    new_threshold = await tuner.calibrate("cam-01", days=30)
    current = await tuner.get_threshold("cam-01")
"""
from __future__ import annotations
import json
from typing import Any
import asyncpg
import redis.asyncio as aioredis

from backend.core.logging import get_logger
from backend.core.settings import get_settings

log      = get_logger("threshold_tuner")
settings = get_settings()

_CALIBRATION_KEY_PREFIX = "netra:calibration:"
_CALIBRATION_TTL        = 86400   # 24 hours
_THRESHOLD_SWEEP_START  = 0.40
_THRESHOLD_SWEEP_END    = 0.90
_THRESHOLD_SWEEP_STEP   = 0.025
_F_BETA                 = 0.5    # beta < 1 → precision-weighted


def _f_beta(tp: int, fp: int, fn: int, beta: float = _F_BETA) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    return ((1 + b2) * precision * recall / denom) if denom > 0 else 0.0


class AdaptiveThresholdTuner:

    def __init__(self, pool: asyncpg.Pool, redis: aioredis.Redis) -> None:
        self._pool  = pool
        self._redis = redis

    async def calibrate(self, camera_id: str, days: int = 30) -> float | None:
        """
        Compute optimal threshold for camera from operator-labeled incidents.
        Returns new threshold, or None if insufficient samples.
        """
        rows = await self._fetch_labeled(camera_id, days)
        if len(rows) < settings.calibration_min_samples:
            log.info(
                "Camera %s: only %d labeled samples (need %d), skipping calibration",
                camera_id, len(rows), settings.calibration_min_samples,
            )
            return None

        best_threshold = settings.risk_alert_threshold
        best_score     = -1.0

        t = _THRESHOLD_SWEEP_START
        while t <= _THRESHOLD_SWEEP_END + 1e-9:
            tp = sum(1 for r in rows if r["risk_score"] >= t and r["status"] == "CONFIRMED")
            fp = sum(1 for r in rows if r["risk_score"] >= t and r["status"] == "DISMISSED")
            fn = sum(1 for r in rows if r["risk_score"] <  t and r["status"] == "CONFIRMED")
            score = _f_beta(tp, fp, fn)
            if score > best_score:
                best_score     = score
                best_threshold = round(t, 3)
            t += _THRESHOLD_SWEEP_STEP

        # Store in Redis
        await self._redis.set(
            _CALIBRATION_KEY_PREFIX + camera_id,
            json.dumps({"threshold": best_threshold, "f_score": round(best_score, 4), "samples": len(rows)}),
            ex=_CALIBRATION_TTL,
        )
        log.info(
            "Camera %s calibrated: threshold=%.3f f0.5=%.4f (n=%d)",
            camera_id, best_threshold, best_score, len(rows),
        )
        return best_threshold

    async def get_threshold(self, camera_id: str) -> float:
        """Return calibrated threshold for camera; fallback to settings default."""
        raw = await self._redis.get(_CALIBRATION_KEY_PREFIX + camera_id)
        if raw:
            return json.loads(raw)["threshold"]
        return settings.risk_alert_threshold

    async def get_calibration(self, camera_id: str) -> dict[str, Any] | None:
        """Return full calibration record for camera, or None."""
        raw = await self._redis.get(_CALIBRATION_KEY_PREFIX + camera_id)
        return json.loads(raw) if raw else None

    async def calibrate_all(self, days: int = 30) -> dict[str, float]:
        """Calibrate all cameras with labeled incidents. Returns {camera_id: threshold}."""
        async with self._pool.acquire() as conn:
            cameras = await conn.fetch(
                "SELECT DISTINCT camera_id FROM incidents "
                "WHERE status IN ('CONFIRMED','DISMISSED') "
                "AND occurred_at >= NOW() - INTERVAL '1 day' * $1",
                days,
            )
        results: dict[str, float] = {}
        for row in cameras:
            cam_id = row["camera_id"]
            t = await self.calibrate(cam_id, days)
            if t is not None:
                results[cam_id] = t
        return results

    async def _fetch_labeled(self, camera_id: str, days: int) -> list[Any]:
        async with self._pool.acquire() as conn:
            return await conn.fetch(
                """SELECT risk_score, status FROM incidents
                   WHERE camera_id = $1
                     AND status IN ('CONFIRMED','DISMISSED')
                     AND occurred_at >= NOW() - INTERVAL '1 day' * $2
                   ORDER BY occurred_at""",
                camera_id, days,
            )
