"""
Unit tests for TheftRiskFSM.

These mirror edgeguard's test_theft_fsm.py but use Netra types.
"""
from __future__ import annotations

import time
import pytest

from netra.types import (
    AssocEventType, AssociationEvent, BBox, ConcealmentType, Keypoint,
    PersonDetection, TheftStage,
)
from backend.services.behavior.fsm import TheftRiskFSM, classify_concealment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _person(track_id: int = 1, keypoints: list[Keypoint] | None = None) -> PersonDetection:
    return PersonDetection(
        track_id  = track_id,
        camera_id = "test-cam",
        bbox      = BBox(100, 100, 200, 400),
        confidence = 0.9,
        keypoints = keypoints or [],
        timestamp = time.time(),
    )


def _assoc(event_type: AssocEventType, track_id: int = 1, zone: str | None = None) -> AssociationEvent:
    return AssociationEvent(
        camera_id  = "test-cam",
        track_id   = track_id,
        event_type = event_type,
        zone_name  = zone,
    )


def _kp_list(
    lw=(150, 300), rw=(160, 300),
    lh=(140, 370), rh=(155, 370),
    ls=(140, 200), rs=(155, 200),
) -> list[Keypoint]:
    """Build 17 COCO keypoints. Fill non-interesting ones with (0,0,0)."""
    kps = [Keypoint(0, 0, 0)] * 17
    kps[5]  = Keypoint(*ls, conf=0.9)
    kps[6]  = Keypoint(*rs, conf=0.9)
    kps[9]  = Keypoint(*lw, conf=0.9)
    kps[10] = Keypoint(*rw, conf=0.9)
    kps[11] = Keypoint(*lh, conf=0.9)
    kps[12] = Keypoint(*rh, conf=0.9)
    return kps


# ---------------------------------------------------------------------------
# FSM tests
# ---------------------------------------------------------------------------

class TestTheftRiskFSM:
    def setup_method(self) -> None:
        self.fsm = TheftRiskFSM("test-cam")
        self.ts  = time.time()

    def test_initial_state_is_browsing(self) -> None:
        events = self.fsm.process([_person()], [], self.ts)
        assert len(events) == 1
        assert events[0].fsm_state == TheftStage.BROWSING
        assert events[0].fsm_score == 0.0

    def test_shelf_interaction_adds_score(self) -> None:
        assoc = [_assoc(AssocEventType.SHELF_INTERACTION)]
        events = self.fsm.process([_person()], assoc, self.ts)
        assert events[0].fsm_score == pytest.approx(2.0, abs=0.1)
        assert events[0].fsm_state == TheftStage.SHELF_INTERACTION

    def test_item_pickup_adds_visual_pick_score(self) -> None:
        assoc = [_assoc(AssocEventType.ITEM_PICKUP)]
        events = self.fsm.process([_person()], assoc, self.ts)
        assert events[0].fsm_score == pytest.approx(4.0, abs=0.1)
        assert events[0].fsm_state == TheftStage.ITEM_PICKED

    def test_concealment_after_pick(self) -> None:
        # Establish pick
        assoc_pick = [_assoc(AssocEventType.ITEM_PICKUP)]
        self.fsm.process([_person()], assoc_pick, self.ts)

        # Pose showing wrist near hip → concealment
        kps = _kp_list(lw=(148, 368), rw=(158, 368))  # wrists ≈ hip
        p   = _person(keypoints=kps)
        events = self.fsm.process([p], [], self.ts + 0.5)

        concealment_events = [e for e in events if e.concealment_type != ConcealmentType.NONE]
        assert len(concealment_events) > 0

    def test_exit_after_concealment_highest_weight(self) -> None:
        # Pick
        self.fsm.process([_person()], [_assoc(AssocEventType.ITEM_PICKUP)], self.ts)
        # Conceal pose
        kps = _kp_list(lw=(148, 368))
        self.fsm.process([_person(keypoints=kps)], [], self.ts + 0.5)
        # Exit
        events = self.fsm.process(
            [_person()],
            [_assoc(AssocEventType.ZONE_EXIT, zone="exit_zone")],
            self.ts + 1.0,
        )
        # EXIT_AFTER_CONCEALMENT weight = 6.0 — should push score high
        assert any(
            s.signal_name == "EXIT_AFTER_CONCEALMENT"
            for e in events for s in e.signals
        )

    def test_risk_decays_over_time(self) -> None:
        assoc = [_assoc(AssocEventType.SHELF_INTERACTION)]
        self.fsm.process([_person()], assoc, self.ts)
        # 5 seconds later with no new signals
        events = self.fsm.process([_person()], [], self.ts + 5.0)
        assert events[0].fsm_score < 2.0   # should have decayed

    def test_suspicious_flag_above_threshold(self) -> None:
        # Accumulate: SHELF_INTERACTION(2) + VISUAL_PICK(4) = 6 → above shopformer_trigger(4)
        self.fsm.process([_person()], [_assoc(AssocEventType.SHELF_INTERACTION)], self.ts)
        events = self.fsm.process([_person()], [_assoc(AssocEventType.ITEM_PICKUP)], self.ts + 0.1)
        assert events[0].suspicious is True

    def test_multiple_tracks_independent(self) -> None:
        p1 = _person(track_id=1)
        p2 = _person(track_id=2)
        assoc1 = [_assoc(AssocEventType.ITEM_PICKUP, track_id=1)]
        events = self.fsm.process([p1, p2], assoc1, self.ts)
        scores = {e.track_id: e.fsm_score for e in events}
        assert scores[1] == pytest.approx(4.0, abs=0.1)
        assert scores[2] == 0.0


# ---------------------------------------------------------------------------
# Concealment classifier tests
# ---------------------------------------------------------------------------

class TestConcealmentClassifier:
    def test_no_concealment_when_wrist_far(self) -> None:
        kps  = _kp_list(lw=(150, 50), rw=(160, 50))   # wrists way above hips
        ctype, dist = classify_concealment(kps, 110.0)
        assert ctype == ConcealmentType.NONE

    def test_hand_to_pocket_near_hip(self) -> None:
        kps  = _kp_list(lw=(148, 370), rw=(158, 370))   # wrists at hip level
        ctype, dist = classify_concealment(kps, 110.0)
        assert ctype != ConcealmentType.NONE
        assert dist < 110.0

    def test_hand_to_pants_below_hip(self) -> None:
        kps  = _kp_list(lh=(140, 350), rh=(155, 350), lw=(140, 420), rw=(155, 420))
        ctype, dist = classify_concealment(kps, 200.0)
        assert ctype == ConcealmentType.HAND_TO_PANTS
