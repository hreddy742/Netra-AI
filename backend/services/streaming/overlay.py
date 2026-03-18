"""
Netra AI — Frame overlay renderer.

Draws bounding boxes, track IDs, FSM state, and risk score bars
onto JPEG frames for MJPEG stream output.

Does NOT copy or store frames — works on in-memory numpy arrays.
"""
from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from backend.core.settings import get_settings

settings = get_settings()

# Severity → BGR colour
_SEVERITY_COLOURS: dict[str, tuple[int, int, int]] = {
    "LOW":      (120, 200, 120),   # green
    "MEDIUM":   (0,   200, 255),   # yellow
    "HIGH":     (0,   140, 255),   # orange
    "CRITICAL": (0,   0,   255),   # red
    "NONE":     (200, 200, 200),   # grey
}

_FSM_LABEL: dict[str, str] = {
    "BROWSING":          "B",
    "NEAR_SHELF":        "NS",
    "SHELF_INTERACTION": "SI",
    "ITEM_PICKED":       "IP",
    "CONCEALMENT":       "!C!",
    "HIGH_RISK_EXIT":    "!!!",
}


def draw_overlay(frame: np.ndarray, overlay_json: str) -> np.ndarray:
    """
    Apply overlay to a frame.

    overlay_json: JSON string produced by inference service:
        [{"track_id": int, "bbox": [x1,y1,x2,y2], "risk": float,
          "fsm_state": str, "severity": str}, ...]

    Returns the frame with overlay drawn (same array, mutated in-place).
    """
    try:
        tracks: list[dict[str, Any]] = json.loads(overlay_json)
    except (json.JSONDecodeError, TypeError):
        return frame

    font       = cv2.FONT_HERSHEY_SIMPLEX
    thickness  = settings.overlay_bbox_thickness
    font_scale = settings.overlay_font_scale

    for t in tracks:
        try:
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            track_id        = t.get("track_id", 0)
            risk            = float(t.get("risk", 0.0))
            fsm_state       = t.get("fsm_state", "BROWSING")
            severity        = t.get("severity", "NONE")

            colour = _SEVERITY_COLOURS.get(severity, _SEVERITY_COLOURS["NONE"])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

            # Label: "ID:5 | !C! | 0.72"
            fsm_label = _FSM_LABEL.get(fsm_state, fsm_state[:3])
            label     = f"ID:{track_id} {fsm_label} {risk:.0%}"
            lx, ly    = x1, max(y1 - 6, 12)

            # Background pill for readability
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(frame, (lx, ly - lh - 4), (lx + lw + 4, ly + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (lx + 2, ly), font, font_scale, colour, 1, cv2.LINE_AA)

            # Risk bar at bottom of bbox
            bar_w    = x2 - x1
            bar_h    = 4
            bar_y    = y2
            filled_w = int(bar_w * risk)
            # Background bar (dark)
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (40, 40, 40), -1)
            # Filled portion — colour based on risk level
            if risk < 0.4:
                bar_colour = (0, 200, 0)    # green
            elif risk < 0.65:
                bar_colour = (0, 200, 255)  # yellow
            else:
                bar_colour = (0, 0, 220)    # red
            cv2.rectangle(frame, (x1, bar_y), (x1 + filled_w, bar_y + bar_h), bar_colour, -1)

        except (KeyError, ValueError, TypeError):
            continue

    # Timestamp watermark
    import time
    ts_str = f"Netra AI  {time.strftime('%H:%M:%S')}"
    cv2.putText(frame, ts_str, (8, frame.shape[0] - 8), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    """Encode numpy frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
