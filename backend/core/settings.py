"""
Netra AI — global settings (Pydantic BaseSettings).

Every service imports this. Values come from environment variables or .env.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    environment: Literal["development", "staging", "production"] = "development"
    model_version: str = "1.0.0"
    logic_version: str = "1.0.0"

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_pool_size: int = 20

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "netra"
    postgres_user: str = "netra"
    postgres_password: str = "netra_secret"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ------------------------------------------------------------------
    # SHM ring buffer
    # ------------------------------------------------------------------
    shm_num_slots: int = 8
    shm_frame_height: int = 720
    shm_frame_width: int = 1280

    # ------------------------------------------------------------------
    # Inference — detection
    # ------------------------------------------------------------------
    detector_model_path: str = "/models/yolo11s.pt"   # upgraded from yolov8n
    pose_model_path: str     = "/models/yolov8n-pose.pt"  # fallback if RTMPose absent
    detection_conf: float    = 0.35
    detection_iou: float     = 0.45
    detector_device: str     = "cpu"          # "cpu" | "cuda" | "0"

    # RTMPose-m ONNX (phase-2.2) — set pose_use_rtmpose=False to use YOLOv8-pose fallback
    pose_onnx_path: str      = "/models/rtmpose-m.onnx"
    pose_use_rtmpose: bool   = True           # disable to fall back to yolov8n-pose
    keypoint_confidence_min: float = 0.3      # min confidence per keypoint

    # ------------------------------------------------------------------
    # Inference — tracking
    # ------------------------------------------------------------------
    tracker_max_age: int      = 90            # 6 s at 15 FPS (upgraded from 30)
    tracker_min_hits: int     = 2
    tracker_iou_threshold: float = 0.3
    tracker_use_reid: bool    = True          # enable OcSort ReID
    reid_weights_path: str    = "/models/osnet_x0_25_msmt17.pt"

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------
    hand_to_hip_distance_px: float  = 110.0
    n_frames_hand_in_shelf: int     = 4
    conceal_window_sec: float       = 4.0
    speed_spike_multiplier: float   = 2.5     # x baseline speed → spike
    zone_config_path: str           = "/config/zones.json"

    # ------------------------------------------------------------------
    # Behavior / FSM
    # ------------------------------------------------------------------
    fsm_risk_threshold: float          = 8.0   # edgeguard-validated
    fsm_decay_rate: float              = 1.8   # pts/sec
    fsm_shopformer_trigger_score: float = 4.0  # run ShopFormer above this

    # Signal weights (DO NOT CHANGE without evaluation data — edgeguard-validated)
    weight_shelf_interaction: float    = 2.0
    weight_visual_pick: float          = 4.0
    weight_hand_to_pocket: float       = 4.0
    weight_hand_to_bag: float          = 3.0
    weight_hand_to_pants: float        = 4.5
    weight_hand_to_shirt: float        = 3.5
    weight_hand_to_hoodie: float       = 4.0
    weight_exit_after_conceal: float   = 6.0
    weight_nonscan_bagging: float      = 5.0
    weight_speed_spike: float          = 1.5

    # ------------------------------------------------------------------
    # ShopFormer
    # ------------------------------------------------------------------
    shopformer_model_path: str         = "/models/shopformer/transformer_best.pth"
    shopformer_gcae_path: str          = "/models/shopformer/gcae_best.pth"
    shopformer_seq_len: int            = 24     # frames per sequence
    shopformer_stride: int             = 12     # frames between sequences
    shopformer_num_kp: int             = 17     # COCO keypoints (pose model output)
    shopformer_latent_dim: int         = 64
    shopformer_num_heads: int          = 12
    shopformer_num_layers: int         = 4
    shopformer_anomaly_threshold: float = 0.60  # above → flag as anomalous
    shopformer_device: str             = "cpu"

    # ------------------------------------------------------------------
    # Risk engine
    # ------------------------------------------------------------------
    risk_weight_fsm: float          = 0.45
    risk_weight_shopformer: float   = 0.30
    risk_weight_interaction: float  = 0.20
    risk_weight_context: float      = 0.05
    risk_alert_threshold: float     = 0.65    # combined score → incident

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    evidence_dir: str              = "/data/evidence"
    evidence_blur_threshold: float = 100.0   # min Laplacian variance to save frame
    clip_pre_event_sec: float      = 10.0
    clip_post_event_sec: float     = 5.0
    evidence_retention_days: int   = 30

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------
    telegram_bot_token: str        = ""
    telegram_chat_id: str          = ""
    telegram_enabled: bool         = False

    email_enabled: bool            = False
    email_smtp_host: str           = ""
    email_smtp_port: int           = 587
    email_smtp_user: str           = ""
    email_smtp_password: str       = ""
    email_from: str                = ""
    email_to: list[str]            = Field(default_factory=list)

    sms_enabled: bool              = False
    twilio_account_sid: str        = ""
    twilio_auth_token: str         = ""
    twilio_from_number: str        = ""
    sms_to_numbers: list[str]      = Field(default_factory=list)

    webhook_url: str               = ""
    webhook_enabled: bool          = False

    # ------------------------------------------------------------------
    # Gateway / API
    # ------------------------------------------------------------------
    api_host: str                  = "0.0.0.0"
    api_port: int                  = 8000
    jwt_secret: str                = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str             = "HS256"
    jwt_expire_minutes: int        = 1440
    cors_origins: list[str]        = Field(default_factory=lambda: ["http://localhost:3000"])
    api_key: str                   = ""        # optional static key for service-to-service

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    webrtc_host: str               = "0.0.0.0"
    webrtc_port: int               = 8080
    sse_max_clients_per_camera: int = 20
    streaming_port: int            = 8001
    mjpeg_jpeg_quality: int        = 75    # 0-100
    mjpeg_target_fps: float        = 10.0  # cap MJPEG output FPS per client
    overlay_bbox_thickness: int    = 2
    overlay_font_scale: float      = 0.5

    # ------------------------------------------------------------------
    # Backpressure / load control
    # ------------------------------------------------------------------
    max_queue_depth_per_camera: int = 30     # mediabridge trims to this
    inference_max_frame_age_ms: float = 500  # drop frames older than this
    inference_max_concurrent: int = 4         # max parallel inference tasks
    adaptive_fps_enabled: bool = True
    adaptive_fps_min: float = 2.0             # minimum FPS when system is overloaded
    adaptive_fps_target_latency_ms: float = 200  # target processing latency

    # ------------------------------------------------------------------
    # Multi-store identity (optional — single-store deployments use defaults)
    # ------------------------------------------------------------------
    store_id: str = "default"
    org_id: str = "default"

    # ------------------------------------------------------------------
    # Phase 8 — Performance & scaling
    # ------------------------------------------------------------------
    gpu_jpeg_encode: bool = False          # enable NVJPEG if available
    ws_shard_count: int = 2               # dispatcher worker count
    broadcaster_pattern_sub: bool = True   # use pattern pubsub (50+ cameras)
    redis_reconnect_max_delay: float = 30.0
    redis_circuit_fail_thresh: int = 5

    # ------------------------------------------------------------------
    # Phase 9 — Accuracy & feedback
    # ------------------------------------------------------------------
    signal_confirmation_frames: int = 3    # majority-vote window; 1=passthrough
    signal_confirmation_age_ms: float = 500.0  # reset buffer after this gap
    heatmap_update_interval_frames: int = 5  # update heatmap every N frames
    training_output_dir: str = "/data/training"
    edge_queue_dir: str = "/data/edge_queue"

    # ------------------------------------------------------------------
    # Phase 10 — Commercial deployment
    # ------------------------------------------------------------------
    alert_cooldown_sec: int = 300          # legacy per-track TTL (kept for compat)
    alert_cooldown_window_seconds: int = 60   # sliding window duration
    alert_cooldown_max_alerts: int = 1        # max alerts per camera per window
    alert_cooldown_burst: int = 2             # burst limit for multi-person CRITICAL events
    alert_escalation_minutes: int = 15     # escalate after N min unreviewed
    cross_signal_min_distinct: int = 2     # min distinct signal types required
    track_confidence_min: float = 0.4      # min composite confidence to fire incident
    calibration_min_samples: int = 20      # min labeled incidents for auto-calibration
    demo_mode: bool = False                # enable synthetic event generation
    demo_events_per_minute: int = 3        # synthetic incident rate in demo mode

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    metrics_port: int = 9090
    metrics_enabled: bool = True

    # ------------------------------------------------------------------
    # Eval log (phase 3)
    # ------------------------------------------------------------------
    eval_log_path: str = "/data/eval/incidents.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
