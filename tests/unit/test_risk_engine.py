"""Unit tests for RiskEngine."""
from __future__ import annotations

import time
import pytest

from netra.types import (
    BehaviorEvent, ConcealmentType, IncidentStatus, ShopFormerScore, TheftStage,
)
from backend.services.risk.engine import RiskEngine


def _bev(
    track_id: int = 1,
    fsm_score: float = 10.0,
    fsm_state: TheftStage = TheftStage.CONCEALMENT,
    suspicious: bool = True,
    ts: float | None = None,
) -> BehaviorEvent:
    return BehaviorEvent(
        camera_id        = "cam-01",
        track_id         = track_id,
        fsm_state        = fsm_state,
        fsm_score        = fsm_score,
        signals          = [],
        concealment_type = ConcealmentType.HAND_TO_POCKET,
        suspicious       = suspicious,
        timestamp        = ts or time.time(),
    )


def _sf_score(track_id: int = 1, anomaly: float = 0.8) -> ShopFormerScore:
    return ShopFormerScore(
        camera_id            = "cam-01",
        track_id             = track_id,
        anomaly_score        = anomaly,
        reconstruction_error = 1.5,
        embedding            = [],
        pose_sequence_len    = 24,
    )


class TestRiskEngine:
    def setup_method(self) -> None:
        self.engine = RiskEngine("cam-01")

    def test_no_incident_below_threshold(self) -> None:
        bev     = _bev(fsm_score=1.0)  # very low
        result  = self.engine.update_behavior(bev)
        assert result is None

    def test_incident_above_threshold(self) -> None:
        bev    = _bev(fsm_score=20.0)  # normalised = 1.0 → 0.45 component alone = 0.45
        # Add shopformer
        self.engine.update_shopformer(_sf_score(anomaly=1.0))
        result = self.engine.update_behavior(bev)
        # 0.45*1.0 + 0.30*1.0 = 0.75 > 0.65 threshold
        assert result is not None
        assert result.risk_score >= 0.65
        assert result.status == IncidentStatus.OPEN

    def test_cooldown_suppresses_repeat(self) -> None:
        self.engine.update_shopformer(_sf_score(anomaly=1.0))
        ts     = time.time()
        result1 = self.engine.update_behavior(_bev(fsm_score=20.0, ts=ts))
        result2 = self.engine.update_behavior(_bev(fsm_score=20.0, ts=ts + 5))
        assert result1 is not None
        assert result2 is None   # within 30s cooldown

    def test_different_tracks_independent(self) -> None:
        self.engine.update_shopformer(_sf_score(track_id=1, anomaly=1.0))
        self.engine.update_shopformer(_sf_score(track_id=2, anomaly=0.0))
        r1 = self.engine.update_behavior(_bev(track_id=1, fsm_score=20.0))
        r2 = self.engine.update_behavior(_bev(track_id=2, fsm_score=20.0))
        assert r1 is not None
        # track 2 has no shopformer → 0.45*1.0 = 0.45 < 0.65 → no incident
        assert r2 is None

    def test_severity_critical_at_very_high_risk(self) -> None:
        from netra.types import AlertSeverity
        self.engine.update_shopformer(_sf_score(anomaly=1.0))
        bev    = _bev(fsm_score=20.0)
        result = self.engine.update_behavior(bev, interaction_event_count=10)
        assert result is not None
        assert result.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)
