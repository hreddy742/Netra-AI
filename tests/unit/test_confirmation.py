"""Tests for SignalConfirmationBuffer."""
import pytest
import time
from netra.types import AssociationEvent, AssocEventType
from backend.services.behavior.confirmation import SignalConfirmationBuffer


def _ev(track_id=1, event_type=AssocEventType.ITEM_PICKUP):
    return AssociationEvent(
        camera_id="test-cam", track_id=track_id,
        event_type=event_type, timestamp=time.time(),
    )


def test_passthrough_when_required_1():
    buf = SignalConfirmationBuffer(required=1)
    evs = [_ev()]
    assert buf.filter(evs, "test-cam") == evs


def test_confirms_at_majority_threshold():
    # required=3 -> majority = ceil(3/2) = 2, per the documented majority-vote
    # design (settings.signal_confirmation_frames: "majority-vote window").
    buf = SignalConfirmationBuffer(required=3)
    ev = _ev()
    assert buf.filter([ev], "test-cam") == []  # frame 1: 1/3, below majority
    result = buf.filter([ev], "test-cam")      # frame 2: 2/3, majority reached
    assert len(result) == 1


def test_reset_on_gap():
    buf = SignalConfirmationBuffer(required=2)
    ev = _ev()
    buf.filter([ev], "test-cam")  # count=1
    buf.filter([], "test-cam")    # gap → reset count to 0
    buf.filter([ev], "test-cam")  # count=1 again
    result = buf.filter([ev], "test-cam")  # count=2 → confirmed
    assert len(result) == 1


def test_different_tracks_independent():
    buf = SignalConfirmationBuffer(required=2)
    ev1 = _ev(track_id=1)
    ev2 = _ev(track_id=2)
    buf.filter([ev1], "test-cam")
    buf.filter([ev2], "test-cam")
    r1 = buf.filter([ev1], "test-cam")  # track 1 count=2, track 2 count=0
    assert any(e.track_id == 1 for e in r1)
    assert not any(e.track_id == 2 for e in r1)
