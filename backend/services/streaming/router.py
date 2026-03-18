"""
Netra AI — MJPEG streaming router.

Mounts as a FastAPI router in the gateway.
Endpoints:
    GET /api/stream/{camera_id}       MJPEG continuous stream
    GET /api/snapshot/{camera_id}     Single JPEG frame

Streaming reads from FrameBroadcaster (SHM-direct, zero Redis JPEG polling).
Snapshot still reads latest_frame_jpeg from Redis (fine for thumbnails).
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from netra.redis_keys import latest_frame_jpeg, overlay_data

log      = get_logger("streaming")
settings = get_settings()

router = APIRouter(tags=["streaming"])

# Timeout waiting for next frame from broadcaster before sending placeholder
_FRAME_TIMEOUT = 3.0 / max(1.0, settings.mjpeg_target_fps)

# Multipart boundary for MJPEG
_BOUNDARY = b"--netra_frame"


async def _frame_generator(
    camera_id: str,
    request: Request,
) -> AsyncIterator[bytes]:
    """
    Yields MJPEG multipart chunks by pulling pre-encoded frames from
    the FrameBroadcaster queue.  The broadcaster handles SHM read +
    overlay draw + JPEG encode once per frame and fans out to all clients.
    """
    broadcaster = request.app.state.broadcaster
    q = broadcaster.add_client(camera_id)
    _placeholder = _make_placeholder(camera_id)

    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                out_bytes: bytes = await asyncio.wait_for(q.get(), timeout=_FRAME_TIMEOUT)
            except asyncio.TimeoutError:
                out_bytes = _placeholder
            yield _boundary_chunk(out_bytes)
    finally:
        broadcaster.remove_client(camera_id, q)


def _boundary_chunk(jpg_bytes: bytes) -> bytes:
    return (
        _BOUNDARY + b"\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpg_bytes)).encode() + b"\r\n"
        b"\r\n" + jpg_bytes + b"\r\n"
    )


def _make_placeholder(camera_id: str) -> bytes:
    """Generate a 320×180 grey placeholder JPEG with 'No Signal' text."""
    import cv2
    import numpy as np
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, f"No signal — {camera_id}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/stream/{camera_id}")
async def mjpeg_stream(
    camera_id: str,
    request:   Request,
) -> StreamingResponse:
    """
    MJPEG continuous stream for a camera.

    Use as:  <img src="/api/stream/cam-01" />
    """
    return StreamingResponse(
        _frame_generator(camera_id, request),
        media_type="multipart/x-mixed-replace; boundary=netra_frame",
        headers={"Cache-Control": "no-cache, no-store"},
    )


@router.get("/snapshot/{camera_id}")
async def snapshot(camera_id: str, request: Request) -> Response:
    """Single JPEG frame with overlay — for thumbnails / evidence."""
    from backend.services.streaming.overlay import draw_overlay, encode_jpeg
    import cv2
    import numpy as np

    redis: aioredis.Redis = request.app.state.redis  # binary client (decode_responses=False)
    jpg_bytes = await redis.get(latest_frame_jpeg(camera_id))

    if not jpg_bytes:
        raise HTTPException(status_code=503, detail=f"No frame available for {camera_id}")

    try:
        arr     = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ov_raw = await redis.get(overlay_data(camera_id))
        if ov_raw:
            ov_json = ov_raw.decode() if isinstance(ov_raw, bytes) else ov_raw
            frame = draw_overlay(frame, ov_json)
        out = encode_jpeg(frame, quality=90)
    except Exception:
        out = jpg_bytes

    return Response(content=out, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})
