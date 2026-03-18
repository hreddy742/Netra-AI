"""
Netra AI — Memory and storage safety utilities.

cleanup_stale_shm    — unlink orphaned SHM segments from previous runs
trim_redis_streams   — XTRIM all named streams to bounded length
cleanup_evidence     — delete evidence files older than retention_days
"""
from __future__ import annotations

import multiprocessing.shared_memory as mp_shm
import time
from pathlib import Path

import redis.asyncio as aioredis

from backend.core.logging import get_logger

log = get_logger("cleanup")

_SHM_SUFFIXES = ("ring", "idx", "gen")


def cleanup_stale_shm(camera_ids: list[str]) -> None:
    """
    Try to unlink SHM segments created by a previous mediabridge run.
    Call on mediabridge startup, before creating new RingBufferWriter instances.
    Silent on FileNotFoundError (segment already gone).
    """
    for cam_id in camera_ids:
        for suffix in _SHM_SUFFIXES:
            name = f"netra_cam_{cam_id}_{suffix}"
            try:
                shm = mp_shm.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                log.info("Unlinked stale SHM segment: %s", name)
            except FileNotFoundError:
                pass
            except Exception as exc:
                log.warning("Could not unlink SHM %s: %s", name, exc)


async def trim_redis_streams(
    redis: aioredis.Redis,
    max_len: int = 500,
) -> None:
    """
    XTRIM all Netra Redis streams to max_len entries (approximate).
    Call periodically (e.g. every 60s) from gateway or any long-running service.
    """
    from netra.events import (
        STREAM_FRAMES, STREAM_DETECTION, STREAM_ASSOCIATION,
        STREAM_BEHAVIOR, STREAM_SHOPFORMER, STREAM_INCIDENTS, STREAM_TELEMETRY,
    )
    streams = [
        STREAM_FRAMES, STREAM_DETECTION, STREAM_ASSOCIATION,
        STREAM_BEHAVIOR, STREAM_SHOPFORMER, STREAM_INCIDENTS, STREAM_TELEMETRY,
    ]
    for stream in streams:
        try:
            await redis.xtrim(stream, maxlen=max_len, approximate=True)
        except Exception as exc:
            log.debug("XTRIM failed for %s: %s", stream, exc)


def cleanup_evidence(evidence_dir: str, retention_days: int) -> int:
    """
    Delete evidence files (frames + clips) older than retention_days.
    Returns the number of files deleted.
    """
    cutoff = time.time() - retention_days * 86400
    deleted = 0
    root = Path(evidence_dir)
    if not root.exists():
        return 0
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        except OSError as exc:
            log.warning("Could not delete evidence file %s: %s", f, exc)
    if deleted:
        log.info("Evidence cleanup: deleted %d files from %s", deleted, evidence_dir)
    return deleted
