"""
AlertPriority — score and rank incidents for operator attention queue.

Priority = severity_weight * risk_score + stage_bonus - time_decay

  severity_weight : CRITICAL=2.5, HIGH=1.5, MEDIUM=1.0, LOW=0.5
  stage_bonus     : EXIT_AFTER_CONCEALMENT=+0.3, CONCEALMENT=+0.2, else 0
  time_decay      : -0.01 per minute since incident (older = lower priority)

Score is bounded to [0.0, 3.0] and mapped to AlertPriority enum.

Usage:
    from backend.services.alerting.priority import AlertPriority, score_incident
    p = score_incident(incident)
    if p >= AlertPriority.HIGH:
        # fast-path notification
"""
from __future__ import annotations
import time
from enum import IntEnum
from netra.events import IncidentEvent


class AlertPriority(IntEnum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


_SEVERITY_WEIGHTS: dict[str, float] = {
    "CRITICAL": 2.5,
    "HIGH":     1.5,
    "MEDIUM":   1.0,
    "LOW":      0.5,
}

_STAGE_BONUSES: dict[str, float] = {
    "EXIT_AFTER_CONCEALMENT": 0.30,
    "CONCEALMENT":            0.20,
    "VISUAL_PICK":            0.10,
}

_DECAY_PER_MINUTE = 0.01
_MAX_DECAY        = 0.30   # cap decay at 30 minutes


def compute_score(incident: IncidentEvent) -> float:
    """Raw priority score in [0.0, 3.0]."""
    severity_w = _SEVERITY_WEIGHTS.get(
        incident.severity.value if hasattr(incident.severity, "value") else str(incident.severity),
        1.0,
    )
    stage_str = (
        incident.theft_stage.value
        if hasattr(incident.theft_stage, "value")
        else str(incident.theft_stage)
    )
    stage_bonus = _STAGE_BONUSES.get(stage_str, 0.0)

    # Time decay (cap at 30 min)
    age_minutes = max(0.0, (time.time() - incident.timestamp) / 60.0)
    decay = min(_DECAY_PER_MINUTE * age_minutes, _MAX_DECAY)

    raw = severity_w * incident.risk_score + stage_bonus - decay
    return max(0.0, min(3.0, raw))


def score_incident(incident: IncidentEvent) -> AlertPriority:
    """Map incident to AlertPriority enum."""
    s = compute_score(incident)
    if s >= 2.5:
        return AlertPriority.CRITICAL
    if s >= 1.5:
        return AlertPriority.HIGH
    if s >= 0.8:
        return AlertPriority.MEDIUM
    return AlertPriority.LOW


def priority_label(p: AlertPriority) -> str:
    return {
        AlertPriority.CRITICAL: "CRITICAL",
        AlertPriority.HIGH:     "HIGH",
        AlertPriority.MEDIUM:   "MEDIUM",
        AlertPriority.LOW:      "LOW",
    }[p]
