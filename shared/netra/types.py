"""
Netra AI — canonical data types shared across all services.

All inter-service messages are based on these types.
Services MUST NOT define their own versions of these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TheftStage(str, Enum):
    BROWSING            = "BROWSING"
    NEAR_SHELF          = "NEAR_SHELF"
    SHELF_INTERACTION   = "SHELF_INTERACTION"
    ITEM_PICKED         = "ITEM_PICKED"
    CONCEALMENT         = "CONCEALMENT"
    HIGH_RISK_EXIT      = "HIGH_RISK_EXIT"


class ConcealmentType(str, Enum):
    NONE         = "NONE"
    HAND_TO_POCKET = "HAND_TO_POCKET"
    HAND_TO_BAG  = "HAND_TO_BAG"
    HAND_TO_PANTS = "HAND_TO_PANTS"
    HAND_TO_SHIRT = "HAND_TO_SHIRT"
    HAND_TO_HOODIE = "HAND_TO_HOODIE"
    NONSCAN_BAGGING = "NONSCAN_BAGGING"


class AlertSeverity(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    OPEN       = "OPEN"
    REVIEWING  = "REVIEWING"
    CONFIRMED  = "CONFIRMED"
    DISMISSED  = "DISMISSED"


# ---------------------------------------------------------------------------
# Frame transport
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Keypoint:
    x: float
    y: float
    conf: float = 0.0


# COCO keypoint indices used by Netra
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12


@dataclass
class FramePointer:
    """Zero-copy SHM frame reference pushed to Redis."""
    camera_id: str
    shm_name: str
    slot_index: int
    generation: int
    width: int
    height: int
    channels: int
    timestamp: float = field(default_factory=time.time)
    frame_seq: int = 0


# ---------------------------------------------------------------------------
# Detection layer
# ---------------------------------------------------------------------------

@dataclass
class PersonDetection:
    track_id: int
    camera_id: str
    bbox: BBox
    confidence: float
    keypoints: list[Keypoint]        # 17 COCO keypoints; empty if pose not run
    timestamp: float = field(default_factory=time.time)
    frame_seq: int = 0


@dataclass
class ItemDetection:
    item_id: int                     # unique per camera per frame (not tracked over time)
    camera_id: str
    bbox: BBox
    class_name: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Association layer
# ---------------------------------------------------------------------------

class AssocEventType(str, Enum):
    SHELF_APPROACH      = "SHELF_APPROACH"
    SHELF_INTERACTION   = "SHELF_INTERACTION"
    ITEM_PICKUP         = "ITEM_PICKUP"
    ITEM_DISAPPEAR      = "ITEM_DISAPPEAR"
    ITEM_TRANSFER       = "ITEM_TRANSFER"
    ZONE_ENTER          = "ZONE_ENTER"
    ZONE_EXIT           = "ZONE_EXIT"
    SPEED_SPIKE         = "SPEED_SPIKE"


@dataclass
class AssociationEvent:
    camera_id: str
    track_id: int
    event_type: AssocEventType
    item_id: Optional[int] = None
    zone_name: Optional[str] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Behavior / FSM layer
# ---------------------------------------------------------------------------

@dataclass
class BehaviorSignal:
    """A single FSM signal emitted for a track."""
    camera_id: str
    track_id: int
    signal_name: str
    weight: float
    fsm_state: TheftStage
    cumulative_fsm_score: float
    concealment_type: ConcealmentType = ConcealmentType.NONE
    timestamp: float = field(default_factory=time.time)


@dataclass
class BehaviorEvent:
    """Aggregated per-frame FSM output for a track."""
    camera_id: str
    track_id: int
    fsm_state: TheftStage
    fsm_score: float                  # cumulative, time-decayed
    signals: list[BehaviorSignal] = field(default_factory=list)
    concealment_type: ConcealmentType = ConcealmentType.NONE
    suspicious: bool = False          # True if fsm_score > shopformer_trigger_threshold
    timestamp: float = field(default_factory=time.time)
    store_id: str = "default"
    org_id: str = "default"


# ---------------------------------------------------------------------------
# ShopFormer layer
# ---------------------------------------------------------------------------

@dataclass
class ShopFormerScore:
    """Output of the ShopFormer GCAE+Transformer anomaly detector."""
    camera_id: str
    track_id: int
    anomaly_score: float              # 0.0 (normal) → 1.0 (highly anomalous)
    reconstruction_error: float       # raw MSE before normalisation
    embedding: list[float]            # pose sequence embedding (for ReID / logging)
    pose_sequence_len: int            # number of frames used
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Risk / Incident layer
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    camera_id: str
    track_id: int
    risk_score: float                 # 0.0–1.0 combined
    fsm_contribution: float
    shopformer_contribution: float
    interaction_contribution: float
    context_contribution: float
    severity: AlertSeverity
    timestamp: float = field(default_factory=time.time)


@dataclass
class IncidentEvent:
    incident_id: str
    camera_id: str
    track_id: int
    risk_score: float
    fsm_score: float
    shopformer_score: float
    theft_stage: TheftStage
    concealment_type: ConcealmentType
    severity: AlertSeverity
    status: IncidentStatus = IncidentStatus.OPEN
    evidence_frame_paths: list[str] = field(default_factory=list)
    clip_path: Optional[str] = None
    model_version: str = "unknown"
    logic_version: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    store_id: str = "default"
    org_id: str = "default"
