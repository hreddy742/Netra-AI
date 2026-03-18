"""
AccuracyMetrics — compute precision, recall, F1 from operator-reviewed incidents.

Primary use: measure model accuracy across a time window.
Called by: retraining pipeline, /api/analytics/accuracy endpoint.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import asyncpg


@dataclass
class MetricSet:
    tp: int
    fp: int
    fn: int   # estimated: incidents that should have fired but didn't
    precision: float
    recall: float | None   # None if fn unknown
    f1: float | None


class AccuracyMetrics:

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def overall(self, days: int = 30) -> MetricSet:
        async with self._pool.acquire() as conn:
            tp = await conn.fetchval(
                "SELECT COUNT(*) FROM incidents WHERE status='CONFIRMED' "
                "AND occurred_at >= NOW() - INTERVAL '1 day' * $1", days
            )
            fp = await conn.fetchval(
                "SELECT COUNT(*) FROM incidents WHERE status='DISMISSED' "
                "AND occurred_at >= NOW() - INTERVAL '1 day' * $1", days
            )
        return _compute(tp, fp)

    async def by_camera(self, days: int = 30) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT camera_id,
                          SUM(CASE WHEN status='CONFIRMED' THEN 1 ELSE 0 END) tp,
                          SUM(CASE WHEN status='DISMISSED' THEN 1 ELSE 0 END) fp
                   FROM incidents
                   WHERE occurred_at >= NOW() - INTERVAL '1 day' * $1
                   GROUP BY camera_id""",
                days,
            )
        result = []
        for r in rows:
            m = _compute(r["tp"], r["fp"])
            result.append({"camera_id": r["camera_id"], **_metric_dict(m)})
        return result

    async def by_severity(self, days: int = 30) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT severity,
                          SUM(CASE WHEN status='CONFIRMED' THEN 1 ELSE 0 END) tp,
                          SUM(CASE WHEN status='DISMISSED' THEN 1 ELSE 0 END) fp
                   FROM incidents
                   WHERE occurred_at >= NOW() - INTERVAL '1 day' * $1
                   GROUP BY severity""",
                days,
            )
        return [{"severity": r["severity"], **_metric_dict(_compute(r["tp"], r["fp"]))} for r in rows]

    async def trend(self, days: int = 30) -> list[dict[str, Any]]:
        """Daily precision trend over past N days."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT DATE_TRUNC('day', occurred_at) day,
                          SUM(CASE WHEN status='CONFIRMED' THEN 1 ELSE 0 END) tp,
                          SUM(CASE WHEN status='DISMISSED' THEN 1 ELSE 0 END) fp
                   FROM incidents
                   WHERE occurred_at >= NOW() - INTERVAL '1 day' * $1
                   GROUP BY 1 ORDER BY 1""",
                days,
            )
        return [{"date": str(r["day"].date()), **_metric_dict(_compute(r["tp"], r["fp"]))} for r in rows]


def _compute(tp: int, fp: int, fn: int = 0) -> MetricSet:
    reviewed = tp + fp
    precision = tp / reviewed if reviewed > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1: float | None = None
    if recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return MetricSet(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def _metric_dict(m: MetricSet) -> dict[str, Any]:
    return {
        "tp": m.tp, "fp": m.fp,
        "precision": round(m.precision, 3),
        "recall":    round(m.recall, 3) if m.recall is not None else None,
        "f1":        round(m.f1, 3) if m.f1 is not None else None,
    }
