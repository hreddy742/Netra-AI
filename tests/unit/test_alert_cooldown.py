"""Tests for CameraAlertCooldown and AlertPriority."""
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch


def test_alert_priority_critical():
    from backend.services.alerting.priority import score_incident, AlertPriority
    from unittest.mock import MagicMock
    inc = MagicMock()
    inc.severity.value = "CRITICAL"
    inc.theft_stage.value = "EXIT_AFTER_CONCEALMENT"
    inc.risk_score = 0.95
    inc.timestamp = time.time()
    p = score_incident(inc)
    assert p == AlertPriority.CRITICAL


def test_alert_priority_low():
    from backend.services.alerting.priority import score_incident, AlertPriority
    from unittest.mock import MagicMock
    inc = MagicMock()
    inc.severity.value = "LOW"
    inc.theft_stage.value = "SHELF_INTERACTION"
    inc.risk_score = 0.40
    inc.timestamp = time.time()
    p = score_incident(inc)
    assert p == AlertPriority.LOW


def test_priority_score_decays_with_age():
    from backend.services.alerting.priority import compute_score
    from unittest.mock import MagicMock
    inc = MagicMock()
    inc.severity.value = "HIGH"
    inc.theft_stage.value = "SHELF_INTERACTION"
    inc.risk_score = 0.80
    inc.timestamp = time.time()
    fresh_score = compute_score(inc)
    inc.timestamp = time.time() - 1800  # 30 min old
    old_score = compute_score(inc)
    assert fresh_score > old_score


@pytest.mark.asyncio
async def test_cooldown_suppresses_after_mark():
    pytest.importorskip("fakeredis")
    import fakeredis.aioredis as fakeredis
    from backend.services.alerting.cooldown import CameraAlertCooldown
    r = fakeredis.FakeRedis(decode_responses=True)
    cd = CameraAlertCooldown(r, cooldown_sec=60)
    assert not await cd.is_suppressed("cam-01", 5)
    await cd.mark_alerted("cam-01", 5)
    assert await cd.is_suppressed("cam-01", 5)


@pytest.mark.asyncio
async def test_cooldown_different_tracks_independent():
    pytest.importorskip("fakeredis")
    import fakeredis.aioredis as fakeredis
    from backend.services.alerting.cooldown import CameraAlertCooldown
    r = fakeredis.FakeRedis(decode_responses=True)
    cd = CameraAlertCooldown(r, cooldown_sec=60)
    await cd.mark_alerted("cam-01", 1)
    assert await cd.is_suppressed("cam-01", 1)
    assert not await cd.is_suppressed("cam-01", 2)
