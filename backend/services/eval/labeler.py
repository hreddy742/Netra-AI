"""
OperatorLabelExporter — queries DB for reviewed incidents, exports labeled dataset.

Output formats:
  - CSV: incident_id, camera_id, risk_score, fsm_score, shopformer_score,
         concealment_type, theft_stage, label (1=TP, 0=FP)
  - JSONL: one JSON object per line with full incident + label

Usage:
    exporter = OperatorLabelExporter(pool)
    df = await exporter.export_csv(output_path, since_days=30)
    counts = await exporter.label_stats()
"""
from __future__ import annotations
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import asyncpg

from backend.core.logging import get_logger

log = get_logger("labeler")


class OperatorLabelExporter:

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def export_csv(self, output_path: str, since_days: int = 90) -> int:
        """Export labeled incidents to CSV. Returns row count."""
        since = datetime.now(tz=timezone.utc) - timedelta(days=since_days)
        rows = await self._fetch_labeled(since)

        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "incident_id", "camera_id", "store_id", "occurred_at",
                "risk_score", "fsm_score", "shopformer_score",
                "theft_stage", "concealment_type", "severity",
                "label", "operator_id",
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "incident_id":      r["id"],
                    "camera_id":        r["camera_id"],
                    "store_id":         r.get("store_id", "default"),
                    "occurred_at":      r["occurred_at"].isoformat(),
                    "risk_score":       r["risk_score"],
                    "fsm_score":        r["fsm_score"],
                    "shopformer_score": r["shopformer_score"],
                    "theft_stage":      r["theft_stage"],
                    "concealment_type": r["concealment_type"],
                    "severity":         r["severity"],
                    "label":            1 if r["status"] == "CONFIRMED" else 0,
                    "operator_id":      r.get("reviewed_by", ""),
                })

        log.info("Exported %d labeled incidents to %s", len(rows), output_path)
        return len(rows)

    async def export_jsonl(self, output_path: str, since_days: int = 90) -> int:
        since = datetime.now(tz=timezone.utc) - timedelta(days=since_days)
        rows = await self._fetch_labeled(since)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for r in rows:
                obj = dict(r)
                # Convert datetime to ISO string for JSON serialization
                for k, v in obj.items():
                    if hasattr(v, "isoformat"):
                        obj[k] = v.isoformat()
                obj["label"] = 1 if obj["status"] == "CONFIRMED" else 0
                f.write(json.dumps(obj) + "\n")
        return len(rows)

    async def label_stats(self) -> dict[str, Any]:
        """Return TP/FP/unreviewed counts and precision estimate."""
        async with self._pool.acquire() as conn:
            total    = await conn.fetchval("SELECT COUNT(*) FROM incidents")
            tp       = await conn.fetchval("SELECT COUNT(*) FROM incidents WHERE status='CONFIRMED'")
            fp       = await conn.fetchval("SELECT COUNT(*) FROM incidents WHERE status='DISMISSED'")
            pending  = await conn.fetchval("SELECT COUNT(*) FROM incidents WHERE status='OPEN'")
        reviewed = tp + fp
        precision = tp / reviewed if reviewed > 0 else None
        return {
            "total":      total,
            "confirmed":  tp,
            "dismissed":  fp,
            "pending":    pending,
            "precision":  round(precision, 3) if precision is not None else None,
            "reviewed_pct": round(reviewed / total * 100, 1) if total > 0 else 0.0,
        }

    async def _fetch_labeled(self, since: datetime) -> list[Any]:
        async with self._pool.acquire() as conn:
            return await conn.fetch(
                """SELECT i.*, r.operator_id
                   FROM incidents i
                   LEFT JOIN reviews r ON r.incident_id = i.id
                   WHERE i.status IN ('CONFIRMED','DISMISSED')
                     AND i.occurred_at >= $1
                   ORDER BY i.occurred_at DESC""",
                since,
            )
