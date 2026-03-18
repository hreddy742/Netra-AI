"""
Netra AI — WebSocket event hub (Phase 8 — sharded dispatch).

Architecture:
  - 1 stream reader task (fast, minimal work)
  - N dispatcher workers (ws_shard_count setting)
  - Per-client send queue (maxsize=20, drop on overflow — never block dispatcher)
  - Heartbeat task (5s)

This decouples Redis I/O from WebSocket send latency, allowing slow clients
to be dropped without stalling the reader.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.logging import get_logger
from backend.core.metrics import ws_clients_total  # new Phase 8 metric
from backend.core.settings import get_settings
from netra.events import (
    STREAM_BEHAVIOR, STREAM_INCIDENTS,
    decode_behavior_event, decode_incident_event,
)
from netra.redis_keys import track_risk

log      = get_logger("ws_router")
settings = get_settings()

router = APIRouter(tags=["websocket"])

_HEARTBEAT_INTERVAL = 5.0
_CLIENT_QUEUE_SIZE  = 20    # per-client backpressure buffer


@dataclass
class _WSClient:
    ws: WebSocket
    camera_id: Optional[str]           # None = all cameras
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=_CLIENT_QUEUE_SIZE))


class WebSocketManager:
    """
    Decoupled stream reader + N dispatcher workers for scalable WS fan-out.
    """

    def __init__(self) -> None:
        self._clients: list[_WSClient]  = []
        self._dispatch_q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._tasks: list[asyncio.Task] = []  # type: ignore[type-arg]
        self._redis: Optional[aioredis.Redis] = None

    async def start(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        shard_count = settings.ws_shard_count

        # 1 reader, N dispatchers, 1 heartbeat
        self._tasks.append(asyncio.create_task(self._reader_loop(), name="ws-reader"))
        for i in range(shard_count):
            self._tasks.append(
                asyncio.create_task(self._dispatcher(i), name=f"ws-dispatcher-{i}")
            )
        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="ws-heartbeat"))

    async def stop(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._clients.clear()
        self._tasks.clear()
        log.info("WebSocketManager stopped")

    async def connect(self, ws: WebSocket, camera_id: Optional[str] = None) -> _WSClient:
        await ws.accept()
        client = _WSClient(ws=ws, camera_id=camera_id)
        self._clients.append(client)
        try:
            ws_clients_total.inc()
        except Exception:
            pass
        log.info("WS connected (filter=%s, total=%d)", camera_id, len(self._clients))
        return client

    def disconnect(self, ws: WebSocket) -> None:
        self._clients = [c for c in self._clients if c.ws is not ws]
        try:
            ws_clients_total.dec()
        except Exception:
            pass
        log.info("WS disconnected (total=%d)", len(self._clients))

    # ------------------------------------------------------------------
    # Internal: stream reader -> dispatch queue
    # ------------------------------------------------------------------

    async def _reader_loop(self) -> None:
        assert self._redis is not None
        last_beh = "$"
        last_inc = "$"
        try:
            while True:
                # Behavior events
                try:
                    beh = await self._redis.xread({STREAM_BEHAVIOR: last_beh}, count=50, block=50)
                except Exception:
                    beh = []
                if beh:
                    for _s, entries in beh:
                        for mid, data in entries:
                            last_beh = mid
                            try:
                                ev   = decode_behavior_event(data)
                                risk = await self._redis.get(track_risk(ev.camera_id, ev.track_id))
                                msg  = {
                                    "type":      "behavior",
                                    "camera_id": ev.camera_id,
                                    "track_id":  ev.track_id,
                                    "risk":      float(risk) if risk else 0.0,
                                    "fsm_state": ev.fsm_state.value,
                                    "ts":        ev.timestamp,
                                }
                                await self._enqueue(msg, ev.camera_id)
                            except Exception:
                                pass

                # Incident events
                try:
                    inc = await self._redis.xread({STREAM_INCIDENTS: last_inc}, count=20, block=50)
                except Exception:
                    inc = []
                if inc:
                    for _s, entries in inc:
                        for mid, data in entries:
                            last_inc = mid
                            try:
                                ev  = decode_incident_event(data)
                                msg = {
                                    "type":        "incident",
                                    "camera_id":   ev.camera_id,
                                    "track_id":    ev.track_id,
                                    "risk":        ev.risk_score,
                                    "fsm_state":   ev.theft_stage.value,
                                    "ts":          ev.timestamp,
                                    "incident_id": ev.incident_id,
                                    "severity":    ev.severity.value,
                                    "theft_stage": ev.theft_stage.value,
                                    "concealment": ev.concealment_type.value,
                                }
                                await self._enqueue(msg, ev.camera_id)
                            except Exception:
                                pass

        except asyncio.CancelledError:
            pass

    async def _enqueue(self, msg: dict, camera_id: str) -> None:
        """Route message to matching client queues."""
        for client in list(self._clients):
            if client.camera_id is not None and client.camera_id != camera_id:
                continue
            if client.queue.full():
                try:
                    client.queue.get_nowait()  # drop oldest for slow client
                except asyncio.QueueEmpty:
                    pass
            try:
                client.queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Dispatcher workers (N parallel WS senders)
    # ------------------------------------------------------------------

    async def _dispatcher(self, shard_id: int) -> None:
        """
        Pull messages from per-client queues and send via WebSocket.
        Each dispatcher handles ALL clients in round-robin.
        """
        try:
            while True:
                dead: list[WebSocket] = []
                for client in list(self._clients):
                    if client.queue.empty():
                        continue
                    try:
                        msg = client.queue.get_nowait()
                        await client.ws.send_json(msg)
                    except Exception:
                        dead.append(client.ws)
                for ws in dead:
                    self.disconnect(ws)
                await asyncio.sleep(0.01)   # 10ms poll interval
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                dead: list[WebSocket] = []
                for client in list(self._clients):
                    try:
                        await client.ws.send_json({"type": "heartbeat"})
                    except Exception:
                        dead.append(client.ws)
                for ws in dead:
                    self.disconnect(ws)
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.websocket("/ws/events")
async def ws_all_events(websocket: WebSocket) -> None:
    manager: WebSocketManager = websocket.app.state.ws_manager
    client = await manager.connect(websocket, camera_id=None)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


@router.websocket("/ws/events/{camera_id}")
async def ws_camera_events(camera_id: str, websocket: WebSocket) -> None:
    manager: WebSocketManager = websocket.app.state.ws_manager
    client = await manager.connect(websocket, camera_id=camera_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
