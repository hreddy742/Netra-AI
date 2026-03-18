"""
Redis Stream event serialisation / deserialisation.

All events are encoded as flat dicts of strings (Redis hash values).
Each stream has a helper encode/decode pair.
"""
from __future__ import annotations

import json
import time
from typing import Any

from netra.types import (
    AssocEventType, AssociationEvent, BBox, BehaviorEvent, BehaviorSignal,
    ConcealmentType, IncidentEvent, IncidentStatus, ItemDetection,
    Keypoint, PersonDetection, RiskAssessment, ShopFormerScore,
    TheftStage, AlertSeverity, FramePointer,
)

# ---------------------------------------------------------------------------
# Redis stream key names
# ---------------------------------------------------------------------------

STREAM_FRAMES       = "netra:frames"          # mediabridge → inference
STREAM_DETECTION    = "netra:detection"        # inference → association
STREAM_ASSOCIATION  = "netra:association"      # association → behavior
STREAM_BEHAVIOR     = "netra:behavior"         # behavior → risk + shopformer
STREAM_SHOPFORMER   = "netra:shopformer"       # shopformer → risk
STREAM_INCIDENTS    = "netra:incidents"        # risk → alerting + persistence + streaming
STREAM_TELEMETRY    = "netra:telemetry"        # all services → monitoring


def _j(v: Any) -> str:
    return json.dumps(v)


def _dj(s: str) -> Any:
    return json.loads(s)


# ---------------------------------------------------------------------------
# FramePointer
# ---------------------------------------------------------------------------

def encode_frame_pointer(fp: FramePointer) -> dict[str, str]:
    return {
        "camera_id":  fp.camera_id,
        "shm_name":   fp.shm_name,
        "slot_index": str(fp.slot_index),
        "generation": str(fp.generation),
        "width":      str(fp.width),
        "height":     str(fp.height),
        "channels":   str(fp.channels),
        "timestamp":  str(fp.timestamp),
        "frame_seq":  str(fp.frame_seq),
    }


def decode_frame_pointer(d: dict[str, str]) -> FramePointer:
    return FramePointer(
        camera_id  = d["camera_id"],
        shm_name   = d["shm_name"],
        slot_index = int(d["slot_index"]),
        generation = int(d["generation"]),
        width      = int(d["width"]),
        height     = int(d["height"]),
        channels   = int(d["channels"]),
        timestamp  = float(d["timestamp"]),
        frame_seq  = int(d["frame_seq"]),
    )


# ---------------------------------------------------------------------------
# DetectionEvent (persons + items, one message per frame per camera)
# ---------------------------------------------------------------------------

def encode_detection_event(
    camera_id: str,
    persons: list[PersonDetection],
    items: list[ItemDetection],
    frame_seq: int,
    timestamp: float | None = None,
) -> dict[str, str]:
    def _enc_person(p: PersonDetection) -> dict:
        return {
            "track_id":   p.track_id,
            "bbox":       [p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2],
            "confidence": p.confidence,
            "keypoints":  [[kp.x, kp.y, kp.conf] for kp in p.keypoints],
            "frame_seq":  p.frame_seq,
        }

    def _enc_item(it: ItemDetection) -> dict:
        return {
            "item_id":    it.item_id,
            "bbox":       [it.bbox.x1, it.bbox.y1, it.bbox.x2, it.bbox.y2],
            "class_name": it.class_name,
            "confidence": it.confidence,
        }

    return {
        "camera_id": camera_id,
        "frame_seq": str(frame_seq),
        "timestamp": str(timestamp or time.time()),
        "persons":   _j([_enc_person(p) for p in persons]),
        "items":     _j([_enc_item(it) for it in items]),
    }


def decode_detection_event(d: dict[str, str]) -> tuple[str, list[PersonDetection], list[ItemDetection], int, float]:
    camera_id = d["camera_id"]
    frame_seq = int(d["frame_seq"])
    timestamp = float(d["timestamp"])

    persons = []
    for p in _dj(d["persons"]):
        b = p["bbox"]
        kps = [Keypoint(k[0], k[1], k[2]) for k in p["keypoints"]]
        persons.append(PersonDetection(
            track_id   = p["track_id"],
            camera_id  = camera_id,
            bbox       = BBox(*b),
            confidence = p["confidence"],
            keypoints  = kps,
            timestamp  = timestamp,
            frame_seq  = p["frame_seq"],
        ))

    items = []
    for it in _dj(d["items"]):
        b = it["bbox"]
        items.append(ItemDetection(
            item_id    = it["item_id"],
            camera_id  = camera_id,
            bbox       = BBox(*b),
            class_name = it["class_name"],
            confidence = it["confidence"],
            timestamp  = timestamp,
        ))

    return camera_id, persons, items, frame_seq, timestamp


# ---------------------------------------------------------------------------
# AssociationEvent
# ---------------------------------------------------------------------------

def encode_association_event(ev: AssociationEvent) -> dict[str, str]:
    return {
        "camera_id":  ev.camera_id,
        "track_id":   str(ev.track_id),
        "event_type": ev.event_type.value,
        "item_id":    str(ev.item_id) if ev.item_id is not None else "",
        "zone_name":  ev.zone_name or "",
        "confidence": str(ev.confidence),
        "metadata":   _j(ev.metadata),
        "timestamp":  str(ev.timestamp),
    }


def decode_association_event(d: dict[str, str]) -> AssociationEvent:
    return AssociationEvent(
        camera_id  = d["camera_id"],
        track_id   = int(d["track_id"]),
        event_type = AssocEventType(d["event_type"]),
        item_id    = int(d["item_id"]) if d["item_id"] else None,
        zone_name  = d["zone_name"] or None,
        confidence = float(d["confidence"]),
        metadata   = _dj(d["metadata"]),
        timestamp  = float(d["timestamp"]),
    )


# ---------------------------------------------------------------------------
# BehaviorEvent
# ---------------------------------------------------------------------------

def encode_behavior_event(ev: BehaviorEvent) -> dict[str, str]:
    signals = [
        {
            "signal_name":           s.signal_name,
            "weight":                s.weight,
            "fsm_state":             s.fsm_state.value,
            "concealment_type":      s.concealment_type.value,
        }
        for s in ev.signals
    ]
    return {
        "camera_id":        ev.camera_id,
        "track_id":         str(ev.track_id),
        "fsm_state":        ev.fsm_state.value,
        "fsm_score":        str(ev.fsm_score),
        "signals":          _j(signals),
        "concealment_type": ev.concealment_type.value,
        "suspicious":       "1" if ev.suspicious else "0",
        "timestamp":        str(ev.timestamp),
    }


def decode_behavior_event(d: dict[str, str]) -> BehaviorEvent:
    signals = [
        BehaviorSignal(
            camera_id              = d["camera_id"],
            track_id               = int(d["track_id"]),
            signal_name            = s["signal_name"],
            weight                 = s["weight"],
            fsm_state              = TheftStage(s["fsm_state"]),
            cumulative_fsm_score   = float(d["fsm_score"]),
            concealment_type       = ConcealmentType(s["concealment_type"]),
            timestamp              = float(d["timestamp"]),
        )
        for s in _dj(d["signals"])
    ]
    return BehaviorEvent(
        camera_id        = d["camera_id"],
        track_id         = int(d["track_id"]),
        fsm_state        = TheftStage(d["fsm_state"]),
        fsm_score        = float(d["fsm_score"]),
        signals          = signals,
        concealment_type = ConcealmentType(d["concealment_type"]),
        suspicious       = d["suspicious"] == "1",
        timestamp        = float(d["timestamp"]),
    )


# ---------------------------------------------------------------------------
# ShopFormerScore
# ---------------------------------------------------------------------------

def encode_shopformer_score(s: ShopFormerScore) -> dict[str, str]:
    return {
        "camera_id":           s.camera_id,
        "track_id":            str(s.track_id),
        "anomaly_score":       str(s.anomaly_score),
        "reconstruction_error": str(s.reconstruction_error),
        "embedding":           _j(s.embedding),
        "pose_sequence_len":   str(s.pose_sequence_len),
        "timestamp":           str(s.timestamp),
    }


def decode_shopformer_score(d: dict[str, str]) -> ShopFormerScore:
    return ShopFormerScore(
        camera_id             = d["camera_id"],
        track_id              = int(d["track_id"]),
        anomaly_score         = float(d["anomaly_score"]),
        reconstruction_error  = float(d["reconstruction_error"]),
        embedding             = _dj(d["embedding"]),
        pose_sequence_len     = int(d["pose_sequence_len"]),
        timestamp             = float(d["timestamp"]),
    )


# ---------------------------------------------------------------------------
# IncidentEvent
# ---------------------------------------------------------------------------

def encode_incident_event(ev: IncidentEvent) -> dict[str, str]:
    return {
        "incident_id":          ev.incident_id,
        "camera_id":            ev.camera_id,
        "track_id":             str(ev.track_id),
        "risk_score":           str(ev.risk_score),
        "fsm_score":            str(ev.fsm_score),
        "shopformer_score":     str(ev.shopformer_score),
        "theft_stage":          ev.theft_stage.value,
        "concealment_type":     ev.concealment_type.value,
        "severity":             ev.severity.value,
        "status":               ev.status.value,
        "evidence_frame_paths": _j(ev.evidence_frame_paths),
        "clip_path":            ev.clip_path or "",
        "model_version":        ev.model_version,
        "logic_version":        ev.logic_version,
        "timestamp":            str(ev.timestamp),
    }


def decode_incident_event(d: dict[str, str]) -> IncidentEvent:
    return IncidentEvent(
        incident_id          = d["incident_id"],
        camera_id            = d["camera_id"],
        track_id             = int(d["track_id"]),
        risk_score           = float(d["risk_score"]),
        fsm_score            = float(d["fsm_score"]),
        shopformer_score     = float(d["shopformer_score"]),
        theft_stage          = TheftStage(d["theft_stage"]),
        concealment_type     = ConcealmentType(d["concealment_type"]),
        severity             = AlertSeverity(d["severity"]),
        status               = IncidentStatus(d["status"]),
        evidence_frame_paths = _dj(d["evidence_frame_paths"]),
        clip_path            = d["clip_path"] or None,
        model_version        = d["model_version"],
        logic_version        = d["logic_version"],
        timestamp            = float(d["timestamp"]),
    )
