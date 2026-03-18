"""Tests for eval.metrics using mock pool."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.services.eval.metrics import _compute, _metric_dict, MetricSet


def test_precision_perfect():
    m = _compute(tp=10, fp=0)
    assert m.precision == 1.0


def test_precision_half():
    m = _compute(tp=5, fp=5)
    assert m.precision == pytest.approx(0.5)


def test_precision_zero_reviewed():
    m = _compute(tp=0, fp=0)
    assert m.precision == 0.0


def test_metric_dict_rounds():
    m = _compute(tp=7, fp=3)
    d = _metric_dict(m)
    assert d["precision"] == 0.7
    assert "tp" in d and "fp" in d


def test_context_modifier_peak_hour():
    from backend.services.behavior.context import SceneContextModifier, ContextSnapshot
    mod = SceneContextModifier()
    ctx = ContextSnapshot(hour_of_day=14, person_count=2, zone_type="shelf")
    delta = mod.threshold_delta(ctx)
    assert delta > 0  # peak hour → stricter threshold


def test_context_modifier_checkout():
    from backend.services.behavior.context import SceneContextModifier, ContextSnapshot
    mod = SceneContextModifier()
    ctx = ContextSnapshot(hour_of_day=3, person_count=1, zone_type="checkout")
    delta = mod.threshold_delta(ctx)
    assert delta < 0  # checkout → looser threshold (higher sensitivity)


def test_context_modifier_crowded():
    from backend.services.behavior.context import SceneContextModifier, ContextSnapshot
    mod = SceneContextModifier()
    ctx = ContextSnapshot(hour_of_day=3, person_count=8, zone_type="shelf")
    delta = mod.threshold_delta(ctx)
    assert delta > 0  # crowded → stricter


def test_occlusion_tracker_person_overlap():
    from backend.services.association.occlusion import OcclusionTracker
    from netra.types import PersonDetection, BBox
    import time
    ot = OcclusionTracker()
    # Two heavily overlapping persons
    p1 = PersonDetection(1, "cam", BBox(100, 100, 300, 400), 0.9, [], time.time())
    p2 = PersonDetection(2, "cam", BBox(120, 100, 320, 400), 0.9, [], time.time())
    ot.update("cam", [p1, p2], frame_seq=1)
    assert ot.is_occluded("cam", 1) or ot.is_occluded("cam", 2)


def test_occlusion_tracker_clear():
    from backend.services.association.occlusion import OcclusionTracker
    from netra.types import PersonDetection, BBox
    import time
    ot = OcclusionTracker()
    p = PersonDetection(1, "cam", BBox(0, 0, 100, 200), 0.9, [], time.time())
    ot.update("cam", [p], frame_seq=1)
    assert not ot.is_occluded("cam", 1)
    assert ot.confidence_factor("cam", 1) == 1.0
