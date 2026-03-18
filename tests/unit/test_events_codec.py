"""Round-trip encode/decode tests for all Redis stream events."""
from __future__ import annotations

import time
from netra.events import (
    encode_frame_pointer, decode_frame_pointer,
    encode_detection_event, decode_detection_event,
    encode_association_event, decode_association_event,
    encode_behavior_event, decode_behavior_event,
    encode_shopformer_score, decode_shopformer_score,
    encode_incident_event, decode_incident_event,
)
from netra.types import (
    AlertSeverity, AssocEventType, AssociationEvent, BBox,
    BehaviorEvent, BehaviorSignal, ConcealmentType,
    FramePointer, IncidentEvent, IncidentStatus,
    ItemDetection, Keypoint, PersonDetection, ShopFormerScore,
    TheftStage,
)


def test_frame_pointer_roundtrip() -> None:
    fp = FramePointer("cam-1", "netra_cam_1_ring", 3, 7, 1280, 720, 3, time.time(), 42)
    assert decode_frame_pointer(encode_frame_pointer(fp)) == fp


def test_detection_event_roundtrip() -> None:
    persons = [PersonDetection(
        track_id=1, camera_id="cam-1",
        bbox=BBox(10, 20, 100, 200), confidence=0.9,
        keypoints=[Keypoint(50, 100, 0.8)],
        timestamp=1.0, frame_seq=5,
    )]
    items = [ItemDetection(
        item_id=0, camera_id="cam-1",
        bbox=BBox(200, 200, 250, 250), class_name="bottle", confidence=0.7,
        timestamp=1.0,
    )]
    enc = encode_detection_event("cam-1", persons, items, frame_seq=5, timestamp=1.0)
    cam, p_out, i_out, fseq, ts = decode_detection_event(enc)
    assert cam == "cam-1"
    assert fseq == 5
    assert len(p_out) == 1
    assert p_out[0].track_id == 1
    assert len(i_out) == 1
    assert i_out[0].class_name == "bottle"


def test_association_event_roundtrip() -> None:
    ev = AssociationEvent(
        camera_id="cam-1", track_id=2,
        event_type=AssocEventType.ITEM_PICKUP,
        item_id=5, zone_name="shelf_zone",
        confidence=0.85, metadata={"foo": "bar"}, timestamp=2.0,
    )
    assert decode_association_event(encode_association_event(ev)) == ev


def test_behavior_event_roundtrip() -> None:
    sig = BehaviorSignal(
        camera_id="cam-1", track_id=3, signal_name="HAND_TO_POCKET",
        weight=4.0, fsm_state=TheftStage.CONCEALMENT,
        cumulative_fsm_score=6.0, concealment_type=ConcealmentType.HAND_TO_POCKET,
        timestamp=3.0,
    )
    bev = BehaviorEvent(
        camera_id="cam-1", track_id=3,
        fsm_state=TheftStage.CONCEALMENT, fsm_score=6.0,
        signals=[sig], concealment_type=ConcealmentType.HAND_TO_POCKET,
        suspicious=True, timestamp=3.0,
    )
    out = decode_behavior_event(encode_behavior_event(bev))
    assert out.fsm_score == 6.0
    assert out.suspicious is True
    assert len(out.signals) == 1
    assert out.signals[0].signal_name == "HAND_TO_POCKET"


def test_shopformer_score_roundtrip() -> None:
    s = ShopFormerScore(
        camera_id="cam-1", track_id=4,
        anomaly_score=0.72, reconstruction_error=1.8,
        embedding=[0.1, 0.2, 0.3], pose_sequence_len=24, timestamp=4.0,
    )
    assert decode_shopformer_score(encode_shopformer_score(s)) == s


def test_incident_event_roundtrip() -> None:
    inc = IncidentEvent(
        incident_id="test-id", camera_id="cam-1", track_id=5,
        risk_score=0.82, fsm_score=12.0, shopformer_score=0.75,
        theft_stage=TheftStage.HIGH_RISK_EXIT,
        concealment_type=ConcealmentType.HAND_TO_PANTS,
        severity=AlertSeverity.HIGH, status=IncidentStatus.OPEN,
        evidence_frame_paths=["a.jpg"], clip_path="clip.mp4",
        model_version="1.0.0", logic_version="1.0.0", timestamp=5.0,
    )
    out = decode_incident_event(encode_incident_event(inc))
    assert out.incident_id == "test-id"
    assert out.risk_score == 0.82
    assert out.clip_path == "clip.mp4"
