"""
Integration tests — Gateway API (no Docker required).

Uses httpx.AsyncClient with FastAPI TestClient pattern.
Redis and Postgres are mocked so these run offline.
"""
from __future__ import annotations

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# App fixture — patch external deps before importing gateway
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_pool():
    """asyncpg pool mock that returns preset data."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__  = AsyncMock(return_value=False)
    pool.close                           = AsyncMock()

    # Default responses
    conn.fetch.return_value = []
    conn.fetchrow.return_value = None
    conn.fetchval.return_value = 0
    conn.execute.return_value = None
    return pool, conn


@pytest.fixture(scope="module")
def mock_redis():
    r = AsyncMock()
    r.aclose = AsyncMock()
    r.set    = AsyncMock(return_value=True)
    r.get    = AsyncMock(return_value=None)
    r.smembers = AsyncMock(return_value=set())
    r.sadd   = AsyncMock(return_value=1)
    r.hset   = AsyncMock(return_value=1)
    r.xread  = AsyncMock(return_value=[])
    r.pipeline.return_value.__aenter__ = AsyncMock(return_value=r)
    r.pipeline.return_value.__aexit__  = AsyncMock(return_value=False)
    return r


@pytest.fixture(scope="module")
def client(mock_pool, mock_redis):
    """FastAPI TestClient with patched Redis + Postgres."""
    from httpx import AsyncClient
    from httpx._transports.asgi import ASGITransport

    pool, _ = mock_pool
    redis    = mock_redis

    with (
        patch("asyncpg.create_pool", AsyncMock(return_value=pool)),
        patch("redis.asyncio.Redis.from_url", MagicMock(return_value=redis)),
    ):
        from backend.gateway.main import app
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_valid(client, mock_redis) -> None:
    from backend.core.settings import get_settings
    settings = get_settings()
    r = await client.post("/api/auth/login", json={
        "username": "admin",
        "password": settings.jwt_secret,
    })
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data


@pytest.mark.asyncio
async def test_login_invalid(client) -> None:
    r = await client.post("/api/auth/login", json={
        "username": "admin",
        "password": "wrongpassword",
    })
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint(client, mock_redis) -> None:
    mock_redis.get = AsyncMock(return_value="ok")
    r = await client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert "gateway" in data
    assert data["gateway"] == "ok"


@pytest.mark.asyncio
async def test_health_shows_down_services(client, mock_redis) -> None:
    mock_redis.get = AsyncMock(return_value=None)  # all services down
    r = await client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    # All services should be "down" when heartbeat keys are missing
    for svc in ["mediabridge", "inference", "association"]:
        assert data.get(svc) == "down"


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_cameras_empty(client, mock_pool) -> None:
    _, conn = mock_pool
    conn.fetch.return_value = []
    r = await client.get("/api/cameras")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_create_camera(client, mock_pool, mock_redis) -> None:
    _, conn = mock_pool
    conn.execute.return_value = None
    mock_redis.sadd.return_value = 1
    mock_redis.hset.return_value = 1

    r = await client.post("/api/cameras", json={
        "id":           "cam-test-01",
        "display_name": "Test Camera 1",
        "location":     "Entrance",
        "rtsp_url":     "rtsp://192.168.1.100:554/stream",
        "source_type":  "rtsp",
        "enabled":      True,
    })
    assert r.status_code == 201
    data = r.json()
    assert data["id"] == "cam-test-01"


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_incidents_empty(client, mock_pool) -> None:
    _, conn = mock_pool
    conn.fetch.return_value  = []
    conn.fetchval.return_value = 0
    r = await client.get("/api/incidents")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["incidents"] == []


@pytest.mark.asyncio
async def test_list_incidents_with_filters(client, mock_pool) -> None:
    _, conn = mock_pool
    conn.fetch.return_value   = []
    conn.fetchval.return_value = 0
    r = await client.get("/api/incidents?severity=HIGH&camera_id=cam-01&status=OPEN")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_get_incident_not_found(client, mock_pool) -> None:
    _, conn = mock_pool
    conn.fetchrow.return_value = None
    r = await client.get("/api/incidents/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_review_incident(client, mock_pool) -> None:
    _, conn = mock_pool
    conn.execute.return_value = None
    r = await client.post("/api/incidents/test-id/review", json={
        "verdict": "CONFIRMED",
        "notes":   "Clearly visible concealment on camera",
    })
    assert r.status_code == 200
    assert r.json()["status"] == "reviewed"


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_endpoint_exists(client) -> None:
    r = await client.get("/metrics")
    # Should return 200 (prometheus format or JSON stub)
    assert r.status_code == 200
