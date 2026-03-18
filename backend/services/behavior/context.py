"""
SceneContextModifier — scene-aware risk scaling for false positive reduction.

Applies multipliers to risk threshold based on:
  1. Time of day (peak hours = higher FP rate → raise threshold)
  2. Scene density (crowded = more occlusion/confusion → raise threshold)
  3. Camera zone type (checkout vs shelf vs exit)

All adjustments are ADDITIVE to threshold, not multiplicative to score.
The FSM and RiskEngine scores are NEVER modified.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

from backend.core.settings import get_settings

settings = get_settings()


@dataclass
class ContextSnapshot:
    hour_of_day: int        # 0-23
    person_count: int       # visible persons in frame
    zone_type: str          # "checkout" | "shelf" | "exit" | "entrance" | "unknown"
    store_id: str = "default"


class SceneContextModifier:
    """
    Computes a threshold adjustment based on scene context.
    Positive adjustment = stricter threshold (fewer alerts, fewer FPs).
    Negative adjustment = looser threshold (more alerts).

    Usage:
        modifier = SceneContextModifier()
        adjusted_threshold = settings.risk_alert_threshold + modifier.threshold_delta(ctx)
    """

    # Peak shopping hours: higher FP rate, raise threshold slightly
    _PEAK_HOURS = {11, 12, 13, 14, 15, 16, 17, 18}
    _PEAK_DELTA = 0.03          # +0.03 during peak hours

    # Crowded scene: ByteTrack struggles with IDs → raise threshold
    _CROWD_THRESHOLD = 6        # persons in frame
    _CROWD_DELTA = 0.05

    # Zone-specific adjustments
    _ZONE_DELTAS: dict[str, float] = {
        "checkout":  -0.05,   # checkout: lower threshold (high risk)
        "exit":      -0.03,   # near exit: slightly lower
        "entrance":   0.02,   # entrance: higher (browsing common)
        "shelf":      0.00,   # shelf: baseline
        "unknown":    0.02,   # unknown zone: be conservative
    }

    def threshold_delta(self, ctx: ContextSnapshot) -> float:
        delta = 0.0
        if ctx.hour_of_day in self._PEAK_HOURS:
            delta += self._PEAK_DELTA
        if ctx.person_count >= self._CROWD_THRESHOLD:
            delta += self._CROWD_DELTA
        delta += self._ZONE_DELTAS.get(ctx.zone_type, 0.0)
        # Cap total adjustment: never raise threshold above 0.85 or below 0.50
        base = settings.risk_alert_threshold
        return max(0.50 - base, min(0.85 - base, delta))

    @staticmethod
    def current_context(camera_id: str, person_count: int, zone_type: str = "unknown") -> ContextSnapshot:
        return ContextSnapshot(
            hour_of_day=time.localtime().tm_hour,
            person_count=person_count,
            zone_type=zone_type,
            store_id=settings.store_id,
        )
