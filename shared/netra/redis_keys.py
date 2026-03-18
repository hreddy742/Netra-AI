"""
Redis key registry — single source of truth for all key names.
"""

# Frame queues (one list per camera, fair round-robin pull by inference)
def frame_queue(camera_id: str) -> str:
    return f"netra:queue:frames:{camera_id}"

# Config hash per camera (read by inference, written by gateway API)
def camera_config(camera_id: str) -> str:
    return f"netra:config:camera:{camera_id}"

# Per-track ShopFormer trigger flag (set by behavior, consumed by shopformer worker)
def shopformer_trigger(camera_id: str, track_id: int) -> str:
    return f"netra:shopformer:trigger:{camera_id}:{track_id}"

# Per-track latest risk score (used by streaming for live overlay)
def track_risk(camera_id: str, track_id: int) -> str:
    return f"netra:risk:{camera_id}:{track_id}"

# Active camera set
ACTIVE_CAMERAS = "netra:cameras:active"

# Health heartbeat keys (each service sets with TTL=10s)
def service_heartbeat(service_name: str) -> str:
    return f"netra:health:{service_name}"

# Latest JPEG frame per camera (for streaming service)
def latest_frame_jpeg(camera_id: str) -> str:
    return f"netra:frame:jpeg:{camera_id}"

# ShopFormer pose sequence buffer (sorted set: score=timestamp, value=keypoints_json)
def shopformer_pose_buffer(camera_id: str, track_id: int) -> str:
    return f"netra:shopformer:poses:{camera_id}:{track_id}"

# Per-camera overlay data for streaming (bboxes, track IDs, risk scores as JSON, TTL=2s)
def overlay_data(camera_id: str) -> str:
    return f"netra:overlay:{camera_id}"

# Per-camera frame counter (for FPS metrics)
def frame_counter(camera_id: str) -> str:
    return f"netra:metrics:frames:{camera_id}"

# Per-camera inference latency (milliseconds, rolling average)
def inference_latency(camera_id: str) -> str:
    return f"netra:metrics:latency:{camera_id}"

# Pub/sub channel: inference notifies broadcaster when a new frame is in SHM
def frame_notify(camera_id: str) -> str:
    return f"netra:frame-notify:{camera_id}"


# Per-camera partitioned streams (optional; global streams still exist)
# Write to these from behavior/risk services for per-camera WS subscriptions
def stream_behavior_cam(camera_id: str) -> str:
    return f"netra:behavior:{camera_id}"

def stream_incidents_cam(camera_id: str) -> str:
    return f"netra:incidents:{camera_id}"

# WebSocket connection counter (for metrics)
WS_CLIENTS_KEY = "netra:metrics:ws_clients"
