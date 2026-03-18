"""
DatasetCollector — builds a structured training dataset from production incidents.

Workflow:
  1. Query DB for labeled incidents (CONFIRMED + DISMISSED) since N days
  2. For each incident: copy evidence frames/clips to output_dir/{label}/
  3. Write manifest.jsonl with metadata per sample
  4. Write dataset_stats.json with summary

Output structure:
  {output_dir}/
    theft/       — CONFIRMED incident evidence
    normal/      — DISMISSED incident evidence (false positives)
    manifest.jsonl
    dataset_stats.json

Usage:
    collector = DatasetCollector(pool)
    stats = await collector.collect("/data/training/dataset_v2", since_days=90)
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import asyncpg

from backend.core.logging import get_logger
from backend.core.settings import get_settings

log      = get_logger("dataset_collector")
settings = get_settings()


class DatasetCollector:

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def collect(self, output_dir: str, since_days: int = 90) -> dict[str, Any]:
        """
        Build dataset from labeled incidents.
        Returns stats dict with sample counts.
        """
        out = Path(output_dir)
        theft_dir  = out / "theft"
        normal_dir = out / "normal"
        theft_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)

        since = datetime.now(tz=timezone.utc) - timedelta(days=since_days)
        rows  = await self._fetch_labeled(since)

        theft_count  = 0
        normal_count = 0
        manifest: list[dict[str, Any]] = []

        for r in rows:
            label    = "theft" if r["status"] == "CONFIRMED" else "normal"
            dest_dir = theft_dir if label == "theft" else normal_dir

            # Copy evidence clips
            clips_copied = []
            for clip in r["clips"]:
                src = Path(clip)
                if src.exists():
                    dst = dest_dir / f"{r['id']}_{src.name}"
                    shutil.copy2(src, dst)
                    clips_copied.append(str(dst))

            # Copy key frames
            frames_copied = []
            for frame in r["frames"][:5]:  # max 5 frames per incident
                src = Path(frame)
                if src.exists():
                    dst = dest_dir / f"{r['id']}_frame_{src.name}"
                    shutil.copy2(src, dst)
                    frames_copied.append(str(dst))

            manifest.append({
                "incident_id":      r["id"],
                "label":            label,
                "camera_id":        r["camera_id"],
                "store_id":         r.get("store_id", "default"),
                "risk_score":       r["risk_score"],
                "fsm_score":        r["fsm_score"],
                "shopformer_score": r["shopformer_score"],
                "theft_stage":      r["theft_stage"],
                "concealment_type": r["concealment_type"],
                "severity":         r["severity"],
                "occurred_at":      r["occurred_at"].isoformat(),
                "clips":            clips_copied,
                "frames":           frames_copied,
            })

            if label == "theft":
                theft_count += 1
            else:
                normal_count += 1

        # Write manifest
        manifest_path = out / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for entry in manifest:
                f.write(json.dumps(entry) + "\n")

        stats = {
            "total":   len(manifest),
            "theft":   theft_count,
            "normal":  normal_count,
            "since_days": since_days,
            "output_dir": str(out),
            "manifest": str(manifest_path),
        }
        (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
        log.info("Dataset collected: %d theft, %d normal → %s", theft_count, normal_count, out)
        return stats

    async def _fetch_labeled(self, since: datetime) -> list[Any]:
        async with self._pool.acquire() as conn:
            incidents = await conn.fetch(
                """SELECT i.id, i.camera_id, i.store_id, i.risk_score,
                          i.fsm_score, i.shopformer_score, i.theft_stage,
                          i.concealment_type, i.severity, i.status, i.occurred_at
                   FROM incidents i
                   WHERE i.status IN ('CONFIRMED','DISMISSED')
                     AND i.occurred_at >= $1
                   ORDER BY i.occurred_at DESC""",
                since,
            )
            result = []
            for inc in incidents:
                row = dict(inc)
                # Fetch associated clips
                clips = await conn.fetch(
                    "SELECT clip_path FROM evidence_clips WHERE incident_id=$1", inc["id"]
                )
                frames = await conn.fetch(
                    "SELECT frame_path FROM evidence_frames WHERE incident_id=$1 "
                    "ORDER BY frame_seq LIMIT 10",
                    inc["id"],
                )
                row["clips"]  = [c["clip_path"] for c in clips]
                row["frames"] = [f["frame_path"] for f in frames]
                result.append(row)
            return result
