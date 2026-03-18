"""Tests for CrossSignalValidator."""
import pytest
from netra.types import AssocEventType
from backend.services.behavior.cross_signal import CrossSignalValidator


def test_passthrough_when_min_1():
    v = CrossSignalValidator(min_distinct=1)
    v.record("cam", 1, AssocEventType.ITEM_PICKUP)
    assert v.is_valid("cam", 1)


def test_blocked_with_single_type():
    v = CrossSignalValidator(min_distinct=2)
    v.record("cam", 1, AssocEventType.ITEM_PICKUP)
    v.record("cam", 1, AssocEventType.ITEM_PICKUP)
    assert not v.is_valid("cam", 1)


def test_valid_after_two_distinct_types():
    v = CrossSignalValidator(min_distinct=2)
    v.record("cam", 1, AssocEventType.ITEM_PICKUP)
    v.record("cam", 1, AssocEventType.SHELF_INTERACTION)
    assert v.is_valid("cam", 1)


def test_speed_spike_excluded():
    v = CrossSignalValidator(min_distinct=2)
    v.record("cam", 1, AssocEventType.SPEED_SPIKE)
    v.record("cam", 1, AssocEventType.SPEED_SPIKE)
    # SPEED_SPIKE excluded from cross-validation
    assert not v.is_valid("cam", 1)


def test_independent_tracks():
    v = CrossSignalValidator(min_distinct=2)
    v.record("cam", 1, AssocEventType.ITEM_PICKUP)
    v.record("cam", 1, AssocEventType.SHELF_INTERACTION)
    v.record("cam", 2, AssocEventType.ITEM_PICKUP)
    assert v.is_valid("cam", 1)
    assert not v.is_valid("cam", 2)


def test_clear_track():
    v = CrossSignalValidator(min_distinct=2)
    v.record("cam", 1, AssocEventType.ITEM_PICKUP)
    v.record("cam", 1, AssocEventType.SHELF_INTERACTION)
    v.clear_track("cam", 1)
    assert not v.is_valid("cam", 1)
