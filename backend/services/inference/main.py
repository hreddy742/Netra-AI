"""
Netra AI — inference service.

Responsibility:
  - Pull FramePointers from per-camera Redis queues (fair round-robin)
  - Read frame from SHM ring buffer
  - Run YOLO detection + ByteTrack + YOLOv8-pose
  - Publish DetectionEvent to netra:detection stream
  - Write overlay JSON for streaming service
  - Write keypoints per-track for ShopFormer
  - Backpressure: drop frames older than inference_max_frame_age_ms
  - Metrics: frames_processed, frames_dropped, inference_duration, queue_depth
"""
from __future__ import annotations

import asyncio
import json
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import redis.asyncio as aioredis

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from backend.core.metrics import (
    frames_processed, frames_dropped, inference_duration,
    queue_depth, service_up, make_metrics_app,
)
from netra.events import STREAM_DETECTION, encode_detection_event, decode_frame_pointer, encode_frame_pointer
from netra.redis_keys import (
    ACTIVE_CAMERAS, frame_queue, service_heartbeat,
    latest_frame_jpeg, overlay_data, frame_notify,
)
from netra.shm import ReaderCache
from backend.services.inference.ml.detector import YOLODetector

log      = get_logger("inference")
settings = get_settings()

# Semaphore: limit concurrent inference tasks to avoid GPU/CPU overload
_INFER_SEM: asyncio.Semaphore | None = None


def _build_overlay_json(
    persons: list,
    items:   list,
    risk_scores: dict[int, float],
) -> str:
    """Build JSON for streaming overlay: list of track annotations."""
    tracks = []
    for p in persons:
        severity = _risk_to_severity(risk_scores.get(p.track_id, 0.0))
        tracks.append({
            "track_id":  p.track_id,
            "bbox":      [p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2],
            "risk":      risk_scores.get(p.track_id, 0.0),
            "fsm_state": "BROWSING",  # updated by risk service; use cached here
            "severity":  severity,
        })
    return json.dumps(tracks)


def _risk_to_severity(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    if score >= 0.80:
        return "HIGH"
    if score >= 0.65:
        return "MEDIUM"
    return "LOW"


async def _get_cached_risks(
    redis: aioredis.Redis,
    cam_id: str,
    track_ids: list[int],
) -> dict[int, float]:
    """Fetch cached risk scores for all visible tracks (pipeline: redis.get per track)."""
    from netra.redis_keys import track_risk
    risks: dict[int, float] = {}
    if not track_ids:
        return risks
    pipe = redis.pipeline(transaction=False)
    for tid in track_ids:
        pipe.get(track_risk(cam_id, tid))
    results = await pipe.execute()
    for tid, val in zip(track_ids, results):
        risks[tid] = float(val) if val else 0.0
    return risks


async def run_worker(
    redis_text: aioredis.Redis,
    redis_bin:  aioredis.Redis,
    detector:   YOLODetector,
    executor:   ThreadPoolExecutor,
) -> None:
    """
    Main inference loop.
    Round-robins across all active camera queues.
    Drops frames older than inference_max_frame_age_ms.
    """
    global _INFER_SEM
    _INFER_SEM = asyncio.Semaphore(settings.inference_max_concurrent)

    log.info("Inference worker started")
    service_up.labels(service="inference").set(1)

    # Adaptive FPS state per camera
    _proc_times: dict[str, list[float]] = {}

    while True:
        camera_ids: set[str] = await redis_text.smembers(ACTIVE_CAMERAS)
        if not camera_ids:
            await asyncio.sleep(0.5)
            continue

        for cam_id in sorted(camera_ids):
            # Update queue depth metric
            depth = await redis_text.llen(frame_queue(cam_id))
            queue_depth.labels(camera_id=cam_id).set(depth)

            # Drain excess frames (backpressure: keep only newest)
            if depth > settings.max_queue_depth_per_camera:
                # Pop and discard all but the newest
                excess = depth - 1
                pipe = redis_text.pipeline(transaction=False)
                for _ in range(int(excess)):
                    pipe.rpop(frame_queue(cam_id))
                await pipe.execute()
                frames_dropped.labels(camera_id=cam_id, reason="queue_overflow").inc(excess)
                log.debug(f"Dropped {excess} frames from {cam_id} queue (backpressure)")

            raw = await redis_text.rpop(frame_queue(cam_id))
            if not raw:
                continue

            try:
                fp = decode_frame_pointer(json.loads(raw))
            except Exception as e:
                log.warning(f"Bad FramePointer from {cam_id}: {e}")
                continue

            # Frame age check — drop if too old
            age_ms = (time.time() - fp.timestamp) * 1000
            if age_ms > settings.inference_max_frame_age_ms:
                frames_dropped.labels(camera_id=cam_id, reason="stale").inc()
                log.debug(f"Stale frame dropped for {cam_id} (age={age_ms:.0f}ms)")
                continue

            # Adaptive FPS: if recent processing is slow, skip with probability
            if settings.adaptive_fps_enabled and cam_id in _proc_times:
                avg_ms = sum(_proc_times[cam_id][-5:]) / len(_proc_times[cam_id][-5:])
                if avg_ms > settings.adaptive_fps_target_latency_ms * 2:
                    # Drop ~50% of frames when severely overloaded
                    import random
                    if random.random() < 0.5:
                        frames_dropped.labels(camera_id=cam_id, reason="adaptive_fps").inc()
                        continue

            # Read from SHM
            try:
                reader = ReaderCache.get(cam_id, settings.shm_num_slots, fp.height, fp.width)
                frame  = reader.read(fp.slot_index, fp.generation)
            except RuntimeError as e:
                frames_dropped.labels(camera_id=cam_id, reason="stale_shm").inc()
                log.debug(f"Stale SHM frame for {cam_id}: {e}")
                continue
            except FileNotFoundError as e:
                log.warning(str(e))
                continue

            # Run inference (CPU/GPU — thread executor to avoid blocking event loop)
            t0 = time.monotonic()
            async with _INFER_SEM:
                loop    = asyncio.get_event_loop()
                persons, items = await loop.run_in_executor(
                    executor,
                    detector.detect,
                    frame, cam_id, fp.frame_seq, True,
                )

            elapsed = (time.monotonic() - t0) * 1000
            _proc_times.setdefault(cam_id, []).append(elapsed)
            if len(_proc_times[cam_id]) > 20:
                _proc_times[cam_id] = _proc_times[cam_id][-20:]

            inference_duration.labels(camera_id=cam_id).observe(elapsed / 1000)
            frames_processed.labels(camera_id=cam_id).inc()

            # Publish detection event
            event = encode_detection_event(
                camera_id = cam_id,
                persons   = persons,
                items     = items,
                frame_seq = fp.frame_seq,
                timestamp = fp.timestamp,
            )
            await redis_text.xadd(STREAM_DETECTION, event, maxlen=500, approximate=True)

            # Fetch cached risk scores for overlay
            track_ids = [p.track_id for p in persons]
            risks     = await _get_cached_risks(redis_text, cam_id, track_ids)

            # Write JPEG for streaming service + notify broadcaster via pub/sub
            try:
                import cv2
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.mjpeg_jpeg_quality])
                await redis_bin.set(latest_frame_jpeg(cam_id), jpeg.tobytes(), ex=5)
                await redis_text.publish(frame_notify(cam_id), json.dumps(encode_frame_pointer(fp)))
            except Exception:
                pass

            # Write overlay JSON for streaming service
            ov_json = _build_overlay_json(persons, items, risks)
            await redis_text.set(overlay_data(cam_id), ov_json, ex=2)

            # Write per-track keypoints for ShopFormer (used by shopformer service)
            for p in persons:
                if p.keypoints:
                    kp_data = [[kp.x, kp.y, kp.conf] for kp in p.keypoints]
                    await redis_text.set(
                        f"netra:kp:{cam_id}:{p.track_id}",
                        json.dumps(kp_data),
                        ex=5,
                    )


async def heartbeat_loop(redis: aioredis.Redis) -> None:
    key = service_heartbeat("inference")
    while True:
        await redis.set(key, "ok", ex=15)
        await asyncio.sleep(5)


async def main() -> None:
    r_text = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=True,
    )
    r_bin = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=False,   # bytes for JPEG
    )

    detector = YOLODetector()
    executor = ThreadPoolExecutor(max_workers=settings.inference_max_concurrent, thread_name_prefix="infer")

    stop_event = asyncio.Event()
    loop       = asyncio.get_event_loop()

    def _sig(*_: Any) -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGTERM, _sig)
    loop.add_signal_handler(signal.SIGINT, _sig)

    asyncio.create_task(heartbeat_loop(r_text))
    asyncio.create_task(run_worker(r_text, r_bin, detector, executor))

    await stop_event.wait()
    service_up.labels(service="inference").set(0)
    executor.shutdown(wait=False)
    await r_text.aclose()
    await r_bin.aclose()
    log.info("inference stopped")


if __name__ == "__main__":
    asyncio.run(main())
