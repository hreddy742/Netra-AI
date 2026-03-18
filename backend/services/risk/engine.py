"""
Netra AI — Risk Engine.

Combines FSM score + ShopFormer anomaly score + interaction score → final risk score.
Emits IncidentEvent when risk_score >= settings.risk_alert_threshold.

Formula:
    risk = 0.45 * fsm_norm
         + 0.30 * shopformer_score    (only when available)
         + 0.20 * interaction_norm
         + 0.05 * context_score

Incident de-duplication: one incident per track per cool-down window.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from netra.types import (
    AlertSeverity, BehaviorEvent, ConcealmentType, IncidentEvent,
    IncidentStatus, RiskAssessment, ShopFormerScore, TheftStage,
)
from backend.core.settings import get_settings
from backend.core.logging import get_logger
from backend.core.metrics import incidents_total, risk_score_hist, fsm_score as fsm_score_metric

log      = get_logger("risk.engine")
settings = get_settings()

_INCIDENT_COOLDOWN_SEC = 30.0    # suppress repeat incidents per track

# ---------------------------------------------------------------------------
# Eval log (phase 3) — append one JSON line per incident for TP/FP labeling
# ---------------------------------------------------------------------------

def _log_incident_for_eval(incident: IncidentEvent, bev: BehaviorEvent) -> None:
    try:
        eval_log = Path(settings.eval_log_path)
        eval_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts":           incident.timestamp,
            "incident_id":  incident.incident_id,
            "camera_id":    incident.camera_id,
            "track_id":     incident.track_id,
            "risk_score":   round(incident.risk_score, 3),
            "fsm_score":    round(incident.fsm_score, 3),
            "shopformer_score": round(incident.shopformer_score or 0.0, 3),
            "severity":     incident.severity.value,
            "stage":        incident.theft_stage.value,
            "signals":      [s.signal_name for s in bev.signals],
            # operator fills these via PATCH /api/eval/incidents/{id}:
            "verdict":      None,   # "tp" | "fp" | "fn" | "unsure"
            "notes":        None,
        }
        with eval_log.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning(f"eval log write failed: {exc}")


@dataclass
class TrackRiskState:
    last_shopformer_score: float = 0.0
    last_shopformer_ts:    float = 0.0
    last_incident_ts:      float = 0.0
    interaction_count:     int   = 0
    last_risk_score:       float = 0.0


def _normalise_fsm(fsm_score: float, max_score: float = 20.0) -> float:
    """Map raw FSM score (0–20+) to 0–1."""
    return min(1.0, max(0.0, fsm_score / max_score))


def _severity(risk_score: float) -> AlertSeverity:
    if risk_score >= 0.90:
        return AlertSeverity.CRITICAL
    if risk_score >= 0.80:
        return AlertSeverity.HIGH
    if risk_score >= 0.65:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


class RiskEngine:
    """
    Stateful per-camera risk combiner.
    Call update_behavior() and update_shopformer() as events arrive.
    Returns IncidentEvent when threshold crossed.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._states: dict[int, TrackRiskState] = {}

    def update_shopformer(self, score: ShopFormerScore) -> None:
        state = self._get(score.track_id)
        state.last_shopformer_score = score.anomaly_score
        state.last_shopformer_ts    = score.timestamp

    def update_behavior(
        self,
        bev: BehaviorEvent,
        interaction_event_count: int = 0,
    ) -> Optional[IncidentEvent]:
        """
        Called for each BehaviorEvent.
        Returns IncidentEvent if risk threshold crossed, else None.
        """
        state = self._get(bev.track_id)
        state.interaction_count += interaction_event_count

        # ---- FSM contribution ----
        fsm_norm = _normalise_fsm(bev.fsm_score)

        # ---- ShopFormer contribution (cached from last score) ----
        shopformer_fresh = (
            bev.timestamp - state.last_shopformer_ts < 5.0
        )
        sf_score = state.last_shopformer_score if shopformer_fresh else 0.0

        # ---- Interaction contribution ----
        # Normalise cumulative interaction count
        interaction_norm = min(1.0, state.interaction_count / 10.0)

        # ---- Context contribution ----
        # Time-of-day risk boost (evening hours 18:00–22:00 → higher risk baseline)
        from datetime import datetime
        hour = datetime.fromtimestamp(bev.timestamp).hour
        context = 0.3 if 18 <= hour <= 22 else 0.0

        # ---- Combined score ----
        w = settings
        risk_score = (
            w.risk_weight_fsm         * fsm_norm
          + w.risk_weight_shopformer  * sf_score
          + w.risk_weight_interaction * interaction_norm
          + w.risk_weight_context     * context
        )

        state.last_risk_score = risk_score

        if risk_score < settings.risk_alert_threshold:
            return None

        # De-duplicate: don't fire within cooldown
        if bev.timestamp - state.last_incident_ts < _INCIDENT_COOLDOWN_SEC:
            return None

        state.last_incident_ts = bev.timestamp

        incident = IncidentEvent(
            incident_id      = str(uuid.uuid4()),
            camera_id        = bev.camera_id,
            track_id         = bev.track_id,
            risk_score       = round(risk_score, 4),
            fsm_score        = bev.fsm_score,
            shopformer_score = sf_score,
            theft_stage      = bev.fsm_state,
            concealment_type = bev.concealment_type,
            severity         = _severity(risk_score),
            status           = IncidentStatus.OPEN,
            model_version    = settings.model_version,
            logic_version    = settings.logic_version,
            timestamp        = bev.timestamp,
        )

        log.info(
            f"INCIDENT {incident.incident_id[:8]} | "
            f"camera={bev.camera_id} track={bev.track_id} "
            f"risk={risk_score:.3f} severity={incident.severity} "
            f"stage={bev.fsm_state} conceal={bev.concealment_type}"
        )

        incidents_total.labels(camera_id=bev.camera_id, severity=incident.severity.value).inc()
        risk_score_hist.labels(camera_id=bev.camera_id).observe(risk_score)

        _log_incident_for_eval(incident, bev)

        return incident

    def _get(self, track_id: int) -> TrackRiskState:
        """Return existing TrackRiskState for track_id, creating it on first access."""
        if track_id not in self._states:
            self._states[track_id] = TrackRiskState()
        return self._states[track_id]

    def prune_stale(self, max_age_sec: float = 60.0) -> None:
        now = time.time()
        stale = [tid for tid, st in self._states.items()
                 if now - max(st.last_shopformer_ts, st.last_incident_ts) > max_age_sec]
        for tid in stale:
            del self._states[tid]
