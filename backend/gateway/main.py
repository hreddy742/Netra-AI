"""
Netra AI — API gateway.

Single FastAPI entry point for the frontend.
Handles: auth, camera config, incident history, SSE live events, system health.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import asyncpg
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from backend.core.settings import get_settings
from backend.core.logging import get_logger
from backend.core.metrics import make_metrics_app
from backend.services.auth.roles import check_role
from backend.services.streaming.broadcaster import FrameBroadcaster
from backend.services.streaming.router import router as streaming_router
from backend.gateway.ws_router import router as ws_router, WebSocketManager
from netra.events import (
    STREAM_INCIDENTS, decode_incident_event,
    STREAM_BEHAVIOR, decode_behavior_event,
)
from netra.redis_keys import service_heartbeat, latest_frame_jpeg, track_risk, ACTIVE_CAMERAS

log      = get_logger("gateway")
settings = get_settings()

# ---------------------------------------------------------------------------
# JWT auth (simple; replace with Keycloak / OAuth2 in production)
# ---------------------------------------------------------------------------

try:
    import jwt as pyjwt
    _HAS_JWT = True
except ImportError:
    _HAS_JWT = False
    log.warning("PyJWT not installed — auth disabled (dev only)")


def _create_token(user_id: str, role: str) -> str:
    if not _HAS_JWT:
        return "dev-token"
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.now(tz=timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return pyjwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _verify_token(token: str) -> dict:
    if not _HAS_JWT or token == "dev-token":
        return {"sub": "dev", "role": "admin"}
    try:
        return pyjwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    if creds is None:
        if settings.environment == "development":
            return {"sub": "dev", "role": "admin"}
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    return _verify_token(creds.credentials)


def require_role(minimum_role: str):
    """Dependency factory: verifies token AND enforces minimum role level."""
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        check_role(user, minimum_role)
        return user
    return _dep


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_redis: aioredis.Redis | None = None
_pool:  asyncpg.Pool   | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _redis, _pool

    # Fail fast on default JWT secret in non-dev environments
    if (settings.jwt_secret == "CHANGE_ME_IN_PRODUCTION"
            and settings.environment != "development"):
        raise RuntimeError(
            "JWT_SECRET environment variable is not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if settings.jwt_secret == "CHANGE_ME_IN_PRODUCTION":
        log.warning(
            "JWT_SECRET is using the default value — acceptable in development only. "
            "Set JWT_SECRET before deploying to staging or production."
        )

    _redis = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=True,
    )
    # Also expose binary Redis on app state for streaming service (JPEG bytes)
    _redis_bin = aioredis.Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        password=settings.redis_password or None,
        decode_responses=False,
    )
    _pool = await asyncpg.create_pool(
        host=settings.postgres_host, port=settings.postgres_port,
        database=settings.postgres_db, user=settings.postgres_user,
        password=settings.postgres_password, min_size=2, max_size=20,
    )
    # Store on app.state for streaming router access
    app.state.redis     = _redis_bin
    app.state.redis_dec = _redis

    # FrameBroadcaster: SHM-direct MJPEG fan-out
    broadcaster = FrameBroadcaster()
    await broadcaster.start(_redis)   # text client for pub/sub + overlay reads
    app.state.broadcaster = broadcaster

    # WebSocket manager: live event hub
    ws_manager = WebSocketManager()
    await ws_manager.start(_redis)
    app.state.ws_manager = ws_manager

    # Periodic Redis stream trim (every 60s)
    from backend.utils.cleanup import trim_redis_streams as _trim
    async def _stream_trimmer() -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await _trim(_redis)
            except Exception:
                pass
    _trim_task = asyncio.create_task(_stream_trimmer(), name="stream-trimmer")

    # Escalation engine (runs in gateway background)
    from backend.services.alerting.escalation import EscalationEngine
    _escalation = EscalationEngine(_pool)
    await _escalation.start()
    app.state.escalation = _escalation

    # Demo mode synthetic generator
    from backend.core.demo_mode import is_demo_mode, SyntheticEventGenerator
    _demo_gen = SyntheticEventGenerator(_redis)
    app.state.demo_gen = _demo_gen
    if is_demo_mode():
        await _demo_gen.start()
        log.info("Demo mode active — synthetic events enabled")

    log.info("Gateway started")
    yield

    _trim_task.cancel()
    try:
        await _trim_task
    except asyncio.CancelledError:
        pass
    await _escalation.stop()
    await _demo_gen.stop()
    await broadcaster.stop()
    await ws_manager.stop()
    if _redis:
        await _redis.aclose()
    await _redis_bin.aclose()
    if _pool:
        await _pool.close()
    log.info("Gateway stopped")


app = FastAPI(
    title="Netra AI Gateway",
    version=settings.model_version,
    docs_url="/docs" if settings.environment != "production" else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount streaming router (MJPEG + snapshots)
app.include_router(streaming_router, prefix="/api")

# Mount WebSocket event hub
app.include_router(ws_router)

# Mount Prometheus metrics
app.mount("/metrics", make_metrics_app())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def login(body: LoginRequest) -> dict:
    # Placeholder — in production verify against DB + bcrypt
    if body.username == "admin" and body.password == settings.jwt_secret:
        token = _create_token("admin", "admin")
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

@app.get("/api/cameras")
async def list_cameras(_user: dict = Depends(get_current_user)) -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM cameras ORDER BY id")
    return [dict(r) for r in rows]


class CameraCreate(BaseModel):
    id:           str
    display_name: str
    location:     str = ""
    rtsp_url:     str = ""
    source_type:  str = "rtsp"
    enabled:      bool = True


@app.post("/api/cameras", status_code=201)
async def create_camera(body: CameraCreate, _user: dict = Depends(get_current_user)) -> dict:
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO cameras (id, display_name, location, rtsp_url, source_type, enabled) "
            "VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (id) DO UPDATE SET "
            "display_name=$2, location=$3, rtsp_url=$4, source_type=$5, enabled=$6",
            body.id, body.display_name, body.location,
            body.rtsp_url, body.source_type, body.enabled,
        )
    # Register in Redis active set and config hash
    await _redis.sadd(ACTIVE_CAMERAS, body.id)
    await _redis.hset(f"netra:config:camera:{body.id}", mapping={
        "url": body.rtsp_url,
        "source_type": body.source_type,
        "enabled": "1" if body.enabled else "0",
    })
    return {"status": "created", "id": body.id}


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

@app.get("/api/incidents")
async def list_incidents(
    limit:    int = 50,
    offset:   int = 0,
    severity: str | None = None,
    camera_id: str | None = None,
    status:    str | None = None,
    _user:    dict = Depends(get_current_user),
) -> dict:
    conditions = []
    params: list[Any] = []
    idx = 1

    if severity:
        conditions.append(f"severity = ${idx}"); params.append(severity); idx += 1
    if camera_id:
        conditions.append(f"camera_id = ${idx}"); params.append(camera_id); idx += 1
    if status:
        conditions.append(f"status = ${idx}"); params.append(status); idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params += [limit, offset]

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM incidents {where} ORDER BY occurred_at DESC LIMIT ${idx} OFFSET ${idx+1}",
            *params,
        )
        total = await conn.fetchval(f"SELECT COUNT(*) FROM incidents {where}", *params[:-2])

    return {
        "total": total,
        "incidents": [dict(r) for r in rows],
    }


@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: str, _user: dict = Depends(get_current_user)) -> dict:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM incidents WHERE id = $1", incident_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return dict(row)


class ReviewRequest(BaseModel):
    verdict: str
    notes:   str = ""


@app.post("/api/incidents/{incident_id}/review")
async def review_incident(
    incident_id: str,
    body:        ReviewRequest,
    user:        dict = Depends(get_current_user),
) -> dict:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE incidents SET status=$1, reviewed_by=$2, reviewed_at=NOW(), operator_notes=$3 WHERE id=$4",
            body.verdict.upper(), user["sub"], body.notes, incident_id,
        )
        await conn.execute(
            "INSERT INTO reviews (incident_id, operator_id, verdict, notes) VALUES ($1,$2,$3,$4)",
            incident_id, user["sub"], body.verdict, body.notes,
        )
    return {"status": "reviewed"}


# ---------------------------------------------------------------------------
# System health
# ---------------------------------------------------------------------------

SERVICES = ["mediabridge", "inference", "association", "behavior", "shopformer", "risk", "alerting", "persistence"]


@app.get("/api/health")
async def health() -> dict:
    results: dict[str, str] = {}
    for svc in SERVICES:
        val = await _redis.get(service_heartbeat(svc))
        results[svc] = "ok" if val else "down"
    results["gateway"] = "ok"
    results["redis"]   = "ok"
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        results["postgres"] = "ok"
    except Exception:
        results["postgres"] = "down"
    return results


# ---------------------------------------------------------------------------
# SSE — live incident stream for frontend
# ---------------------------------------------------------------------------

@app.get("/api/stream/incidents")
async def stream_incidents(
    request: Request,
    _user:   dict = Depends(get_current_user),
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        last_id = "$"
        while True:
            if await request.is_disconnected():
                break
            msgs = await _redis.xread({STREAM_INCIDENTS: last_id}, count=10, block=500)
            if msgs:
                for _s, entries in msgs:
                    for msg_id, data in entries:
                        last_id = msg_id
                        try:
                            incident = decode_incident_event(data)
                            payload  = json.dumps({
                                "incident_id":   incident.incident_id,
                                "camera_id":     incident.camera_id,
                                "track_id":      incident.track_id,
                                "risk_score":    incident.risk_score,
                                "severity":      incident.severity.value,
                                "theft_stage":   incident.theft_stage.value,
                                "concealment":   incident.concealment_type.value,
                                "timestamp":     incident.timestamp,
                            })
                            yield f"data: {payload}\n\n"
                        except Exception:
                            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# SSE — live risk scores per camera (for dashboard overlay)
# ---------------------------------------------------------------------------

@app.get("/api/stream/risk/{camera_id}")
async def stream_risk(
    camera_id: str,
    request:   Request,
    _user:     dict = Depends(get_current_user),
) -> StreamingResponse:
    async def generator() -> AsyncIterator[str]:
        last_id = "$"
        while True:
            if await request.is_disconnected():
                break
            msgs = await _redis.xread({STREAM_BEHAVIOR: last_id}, count=20, block=200)
            if msgs:
                for _s, entries in msgs:
                    for msg_id, data in entries:
                        last_id = msg_id
                        try:
                            bev = decode_behavior_event(data)
                            if bev.camera_id != camera_id:
                                continue
                            risk = await _redis.get(track_risk(camera_id, bev.track_id)) or "0"
                            payload = json.dumps({
                                "track_id":  bev.track_id,
                                "risk":      float(risk),
                                "fsm_state": bev.fsm_state.value,
                                "ts":        bev.timestamp,
                            })
                            yield f"data: {payload}\n\n"
                        except Exception:
                            pass

    return StreamingResponse(generator(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Phase 9: Incident timeline
# ---------------------------------------------------------------------------

@app.get("/api/incidents/{incident_id}/timeline")
async def incident_timeline(incident_id: str, _user: dict = Depends(get_current_user)) -> dict:
    """Ordered sequence of behavior signals for an incident."""
    async with _pool.acquire() as conn:
        inc = await conn.fetchrow("SELECT * FROM incidents WHERE id=$1", incident_id)
        if not inc:
            raise HTTPException(404, "Not found")
        signals = await conn.fetch(
            "SELECT * FROM behavior_signals WHERE incident_id=$1 ORDER BY occurred_at",
            incident_id,
        )
        frames = await conn.fetch(
            "SELECT * FROM evidence_frames WHERE incident_id=$1 ORDER BY frame_seq",
            incident_id,
        )
    return {
        "incident":  dict(inc),
        "signals":   [dict(s) for s in signals],
        "frames":    [dict(f) for f in frames],
    }


# ---------------------------------------------------------------------------
# Phase 9: Evidence clip streaming
# ---------------------------------------------------------------------------

@app.get("/api/incidents/{incident_id}/clip")
async def stream_clip(incident_id: str, _user: dict = Depends(get_current_user)) -> Response:
    """Stream evidence video clip for an incident."""
    import os
    from fastapi.responses import FileResponse
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT clip_path FROM evidence_clips WHERE incident_id=$1 LIMIT 1",
            incident_id,
        )
    if not row or not row["clip_path"]:
        raise HTTPException(404, "No clip available")
    path = row["clip_path"]
    if not os.path.exists(path):
        raise HTTPException(404, "Clip file not found on disk")
    return FileResponse(path, media_type="video/mp4",
                        headers={"Content-Disposition": f'inline; filename="{incident_id}.mp4"'})


# ---------------------------------------------------------------------------
# Phase 9: Zone heatmap
# ---------------------------------------------------------------------------

@app.get("/api/cameras/{camera_id}/heatmap")
async def get_heatmap(camera_id: str, _user: dict = Depends(get_current_user)) -> dict:
    """2D density grid of person track positions for a camera."""
    from backend.services.analytics.heatmap import ZoneHeatmap
    hm = ZoneHeatmap(_redis)
    return await hm.get(camera_id)


# ---------------------------------------------------------------------------
# Phase 9: Multi-camera correlation
# ---------------------------------------------------------------------------

@app.get("/api/incidents/{incident_id}/correlated")
async def correlated_incidents(incident_id: str, _user: dict = Depends(get_current_user)) -> list:
    from backend.services.analytics.correlation import IncidentCorrelator
    correlator = IncidentCorrelator(_pool, _redis)
    return await correlator.correlated(incident_id)


@app.get("/api/analytics/incident-groups")
async def incident_groups(limit: int = 20, _user: dict = Depends(get_current_user)) -> list:
    """Recent multi-camera correlated incident clusters."""
    from backend.services.analytics.correlation import IncidentCorrelator
    correlator = IncidentCorrelator(_pool, _redis)
    return await correlator.groups(limit=limit)


# ---------------------------------------------------------------------------
# Phase 9: Accuracy metrics (from operator labels)
# ---------------------------------------------------------------------------

def _metric_dict(ms) -> dict:
    from backend.services.eval.metrics import _metric_dict as _md
    return _md(ms)


@app.get("/api/analytics/accuracy")
async def accuracy_metrics(days: int = 30, _user: dict = Depends(get_current_user)) -> dict:
    from backend.services.eval.metrics import AccuracyMetrics
    m = AccuracyMetrics(_pool)
    return {
        "overall":     _metric_dict(await m.overall(days)),
        "by_camera":   await m.by_camera(days),
        "by_severity": await m.by_severity(days),
        "trend":       await m.trend(days),
    }


# ---------------------------------------------------------------------------
# Phase 9: Label export (triggers CSV export for retraining)
# ---------------------------------------------------------------------------

@app.post("/api/analytics/export-labels")
async def export_labels(
    output_path: str = "/data/training/labels.csv",
    days: int = 90,
    _user: dict = Depends(get_current_user),
) -> dict:
    from backend.services.eval.labeler import OperatorLabelExporter
    exporter = OperatorLabelExporter(_pool)
    count    = await exporter.export_csv(output_path, since_days=days)
    stats    = await exporter.label_stats()
    return {"exported": count, "output": output_path, "stats": stats}


# ---------------------------------------------------------------------------
# Phase 9: Store-level analytics
# ---------------------------------------------------------------------------

@app.get("/api/analytics/stores")
async def store_analytics(days: int = 7, _user: dict = Depends(get_current_user)) -> list:
    """Per-store incident counts, severity breakdown, precision estimate."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT store_id,
                      COUNT(*) total,
                      SUM(CASE WHEN severity='CRITICAL' THEN 1 ELSE 0 END) critical,
                      SUM(CASE WHEN severity='HIGH'     THEN 1 ELSE 0 END) high,
                      SUM(CASE WHEN status='CONFIRMED'  THEN 1 ELSE 0 END) confirmed,
                      SUM(CASE WHEN status='DISMISSED'  THEN 1 ELSE 0 END) dismissed,
                      AVG(risk_score) avg_risk
               FROM incidents
               WHERE occurred_at >= NOW() - INTERVAL '1 day' * $1
               GROUP BY store_id
               ORDER BY total DESC""",
            days,
        )
    result = []
    for r in rows:
        reviewed = r["confirmed"] + r["dismissed"]
        result.append({
            "store_id":  r["store_id"],
            "total":     r["total"],
            "critical":  r["critical"],
            "high":      r["high"],
            "confirmed": r["confirmed"],
            "dismissed": r["dismissed"],
            "precision": round(r["confirmed"] / reviewed, 3) if reviewed else None,
            "avg_risk":  round(float(r["avg_risk"]), 3),
        })
    return result


@app.get("/api/analytics/overview")
async def analytics_overview(days: int = 7, _user: dict = Depends(get_current_user)) -> dict:
    """System-wide overview: total incidents, FP rate, top cameras, trend."""
    async with _pool.acquire() as conn:
        total  = await conn.fetchval(
            "SELECT COUNT(*) FROM incidents WHERE occurred_at >= NOW() - INTERVAL '1 day' * $1", days)
        by_sev = await conn.fetch(
            """SELECT severity, COUNT(*) n FROM incidents
               WHERE occurred_at >= NOW() - INTERVAL '1 day' * $1
               GROUP BY severity""", days)
        top_cams = await conn.fetch(
            """SELECT camera_id, COUNT(*) n FROM incidents
               WHERE occurred_at >= NOW() - INTERVAL '1 day' * $1
               GROUP BY camera_id ORDER BY n DESC LIMIT 5""", days)
    return {
        "total_incidents":   total,
        "by_severity":       {r["severity"]: r["n"] for r in by_sev},
        "top_cameras":       [{"camera_id": r["camera_id"], "count": r["n"]} for r in top_cams],
    }

# ===========================================================================
# Phase 10: RBAC — User management (admin only)
# ===========================================================================

class UserCreateBody(BaseModel):
    username: str
    password: str
    role: str = "operator"


@app.get("/api/users")
async def list_users(_user: dict = Depends(require_role("admin"))) -> list:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, username, role, created_at FROM users ORDER BY created_at DESC"
        )
    return [dict(r) for r in rows]


@app.post("/api/users", status_code=201)
async def create_user(body: UserCreateBody, _user: dict = Depends(require_role("admin"))) -> dict:
    try:
        import bcrypt
        pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        import hashlib
        pw_hash = hashlib.sha256(body.password.encode()).hexdigest()

    import uuid as _uuid
    user_id = str(_uuid.uuid4())
    async with _pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                "INSERT INTO users (id, username, password_hash, role) VALUES ($1,$2,$3,$4) RETURNING id, username, role",
                user_id, body.username, pw_hash, body.role,
            )
        except Exception as e:
            raise HTTPException(409, f"Username already exists: {e}")
    return dict(row)


@app.patch("/api/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role: str,
    _user: dict = Depends(require_role("admin")),
) -> dict:
    from backend.services.auth.roles import ROLE_HIERARCHY
    if role not in ROLE_HIERARCHY:
        raise HTTPException(400, f"Invalid role. Valid: {ROLE_HIERARCHY}")
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE users SET role=$1 WHERE id=$2 RETURNING id, username, role",
            role, user_id,
        )
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)


# ===========================================================================
# Phase 10: Store onboarding (admin only)
# ===========================================================================

class StoreCreateBody(BaseModel):
    store_id: str
    org_id: str
    display_name: str
    address: str = ""
    timezone: str = "UTC"
    plan: str = "standard"


@app.get("/api/stores")
async def list_stores(_user: dict = Depends(require_role("manager"))) -> list:
    async with _pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM stores ORDER BY created_at DESC")
    return [dict(r) for r in rows]


@app.post("/api/stores", status_code=201)
async def create_store(body: StoreCreateBody, _user: dict = Depends(require_role("admin"))) -> dict:
    async with _pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                "INSERT INTO stores (store_id, org_id, display_name, address, timezone, plan)"
                " VALUES ($1,$2,$3,$4,$5,$6) RETURNING *",
                body.store_id, body.org_id, body.display_name,
                body.address, body.timezone, body.plan,
            )
        except Exception as e:
            raise HTTPException(409, f"Store already exists: {e}")
    return dict(row)


@app.get("/api/stores/{store_id}")
async def get_store(store_id: str, _user: dict = Depends(require_role("manager"))) -> dict:
    async with _pool.acquire() as conn:
        store = await conn.fetchrow("SELECT * FROM stores WHERE store_id=$1", store_id)
        if not store:
            raise HTTPException(404, "Store not found")
        cameras = await conn.fetch(
            "SELECT id, display_name, enabled FROM cameras WHERE store_id=$1", store_id
        )
        recent = await conn.fetchval(
            "SELECT COUNT(*) FROM incidents WHERE store_id=$1 AND occurred_at >= NOW() - INTERVAL '24 hours'",
            store_id,
        )
    return {
        "store":         dict(store),
        "cameras":       [dict(c) for c in cameras],
        "incidents_24h": recent,
    }


# ===========================================================================
# Phase 10: Alert management
# ===========================================================================

@app.get("/api/alerts/pending")
async def pending_alerts(
    limit: int = 50,
    _user: dict = Depends(require_role("operator")),
) -> list:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM incidents WHERE status IN ('OPEN','ESCALATED')"
            " ORDER BY risk_score DESC, occurred_at DESC LIMIT $1",
            limit,
        )
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.get("/api/alerts/escalations")
async def escalated_alerts(_user: dict = Depends(require_role("operator"))) -> list:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT i.*, e.reason, e.escalated_at"
            " FROM incidents i"
            " JOIN alert_escalations e ON e.incident_id = i.id"
            " WHERE i.status = 'ESCALATED'"
            " ORDER BY e.escalated_at DESC LIMIT 100"
        )
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
        result.append(d)
    return result


@app.post("/api/alerts/{incident_id}/acknowledge")
async def acknowledge_alert(
    incident_id: str,
    _user: dict = Depends(require_role("operator")),
) -> dict:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE incidents SET status='OPEN', reviewed_by=$1"
            " WHERE id=$2 AND status='ESCALATED' RETURNING id, status",
            _user["sub"], incident_id,
        )
    if not row:
        raise HTTPException(404, "Incident not found or not escalated")
    return dict(row)


# ===========================================================================
# Phase 10: Camera calibration
# ===========================================================================

@app.post("/api/cameras/{camera_id}/calibrate")
async def calibrate_camera(
    camera_id: str,
    days: int = 30,
    _user: dict = Depends(require_role("manager")),
) -> dict:
    from backend.services.eval.threshold_tuner import AdaptiveThresholdTuner
    tuner = AdaptiveThresholdTuner(_pool, _redis)
    threshold = await tuner.calibrate(camera_id, days=days)
    if threshold is None:
        raise HTTPException(
            422,
            f"Insufficient labeled samples for camera '{camera_id}' (need {settings.calibration_min_samples})",
        )
    return {"camera_id": camera_id, "threshold": threshold, "days_used": days}


@app.get("/api/cameras/{camera_id}/calibration")
async def get_calibration(
    camera_id: str,
    _user: dict = Depends(require_role("operator")),
) -> dict:
    from backend.services.eval.threshold_tuner import AdaptiveThresholdTuner
    tuner = AdaptiveThresholdTuner(_pool, _redis)
    cal = await tuner.get_calibration(camera_id)
    if cal is None:
        return {"camera_id": camera_id, "threshold": settings.risk_alert_threshold, "calibrated": False}
    return {"camera_id": camera_id, **cal, "calibrated": True}


@app.post("/api/analytics/calibrate-all")
async def calibrate_all_cameras(
    days: int = 30,
    _user: dict = Depends(require_role("manager")),
) -> dict:
    from backend.services.eval.threshold_tuner import AdaptiveThresholdTuner
    tuner = AdaptiveThresholdTuner(_pool, _redis)
    results = await tuner.calibrate_all(days=days)
    return {"calibrated": len(results), "thresholds": results}


# ===========================================================================
# Phase 10: Model registry
# ===========================================================================

class ModelDeployBody(BaseModel):
    model_name: str
    model_version: str
    path: str
    notes: str = ""


@app.get("/api/models")
async def list_models(_user: dict = Depends(require_role("manager"))) -> list:
    from backend.services.data.model_registry import ModelRegistry
    return await ModelRegistry(_pool).all_active()


@app.get("/api/models/{model_name}/history")
async def model_history(model_name: str, _user: dict = Depends(require_role("manager"))) -> list:
    from backend.services.data.model_registry import ModelRegistry
    return await ModelRegistry(_pool).history(model_name)


@app.post("/api/models/deploy", status_code=201)
async def deploy_model(body: ModelDeployBody, _user: dict = Depends(require_role("admin"))) -> dict:
    from backend.services.data.model_registry import ModelRegistry
    return await ModelRegistry(_pool).deploy(
        body.model_name, body.model_version, body.path, notes=body.notes
    )


@app.post("/api/models/{model_name}/rollback")
async def rollback_model(model_name: str, _user: dict = Depends(require_role("admin"))) -> dict:
    from backend.services.data.model_registry import ModelRegistry
    row = await ModelRegistry(_pool).rollback(model_name)
    if not row:
        raise HTTPException(404, f"No previous version found for '{model_name}'")
    return row


# ===========================================================================
# Phase 10: Dataset collection
# ===========================================================================

@app.post("/api/data/collect-dataset")
async def collect_dataset(
    output_dir: str = "/data/training/dataset",
    days: int = 90,
    _user: dict = Depends(require_role("manager")),
) -> dict:
    from backend.services.data.collector import DatasetCollector
    return await DatasetCollector(_pool).collect(output_dir, since_days=days)


# ===========================================================================
# Phase 10: Demo mode control (admin only)
# ===========================================================================

@app.get("/api/demo/status")
async def demo_status(_user: dict = Depends(require_role("operator"))) -> dict:
    from backend.core.demo_mode import is_demo_mode
    gen = app.state.demo_gen
    return {
        "demo_mode_enabled": is_demo_mode(),
        "generator_running": gen.is_running,
        "events_per_minute": settings.demo_events_per_minute,
    }


@app.post("/api/demo/start")
async def demo_start(_user: dict = Depends(require_role("admin"))) -> dict:
    await app.state.demo_gen.start()
    return {"status": "started", "events_per_minute": settings.demo_events_per_minute}


@app.post("/api/demo/stop")
async def demo_stop(_user: dict = Depends(require_role("admin"))) -> dict:
    await app.state.demo_gen.stop()
    return {"status": "stopped"}


# ===========================================================================
# Phase 3: Eval log — incident review for TP/FP/FN labeling
# ===========================================================================

@app.get("/api/eval/incidents")
async def eval_incidents(
    limit: int = 100,
    _user: dict = Depends(require_role("operator")),
) -> list[dict]:
    """Return last `limit` entries from the eval log (most recent last)."""
    import json as _json
    from pathlib import Path as _Path
    eval_log = _Path(settings.eval_log_path)
    if not eval_log.exists():
        return []
    lines = eval_log.read_text().splitlines()
    recent = lines[-limit:] if len(lines) > limit else lines
    result = []
    for line in recent:
        try:
            result.append(_json.loads(line))
        except Exception:
            pass
    return result


class EvalVerdictBody(BaseModel):
    verdict: str   # "tp" | "fp" | "fn" | "unsure"
    notes:   str = ""


@app.patch("/api/eval/incidents/{incident_id}")
async def eval_update_incident(
    incident_id: str,
    body: EvalVerdictBody,
    _user: dict = Depends(require_role("operator")),
) -> dict:
    """Update verdict + notes for an incident in the eval log."""
    import json as _json
    from pathlib import Path as _Path
    valid_verdicts = {"tp", "fp", "fn", "unsure"}
    if body.verdict not in valid_verdicts:
        raise HTTPException(400, f"verdict must be one of {valid_verdicts}")
    eval_log = _Path(settings.eval_log_path)
    if not eval_log.exists():
        raise HTTPException(404, "Eval log not found")
    lines = eval_log.read_text().splitlines()
    updated = False
    new_lines = []
    for line in lines:
        try:
            entry = _json.loads(line)
        except Exception:
            new_lines.append(line)
            continue
        if entry.get("incident_id") == incident_id:
            entry["verdict"] = body.verdict
            entry["notes"]   = body.notes
            updated = True
        new_lines.append(_json.dumps(entry))
    if not updated:
        raise HTTPException(404, f"incident_id {incident_id!r} not found in eval log")
    eval_log.write_text("\n".join(new_lines) + "\n")
    return {"status": "updated", "incident_id": incident_id, "verdict": body.verdict}
