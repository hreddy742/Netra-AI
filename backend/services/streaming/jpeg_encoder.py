"""
Netra AI — JPEG encoder with GPU acceleration (NVJPEG) and fast CPU fallback.

Backend priority:
  1. NVJPEG  — CUDA hosts with pynvjpeg installed
  2. simplejpeg — fast CPU (libjpeg-turbo, ~3x vs OpenCV)
  3. OpenCV imencode — universal fallback

API:
    encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes
    get_backend() -> str   # "nvjpeg" | "simplejpeg" | "opencv"
"""
from __future__ import annotations

import numpy as np
from backend.core.logging import get_logger

log = get_logger("jpeg_encoder")

_backend: str = ""


def _init_backend() -> str:
    global _backend
    if _backend:
        return _backend

    # 1. Try NVJPEG
    try:
        import pynvjpeg  # noqa: F401
        import torch
        if torch.cuda.is_available():
            _backend = "nvjpeg"
            log.info("JPEG encoder backend: NVJPEG (GPU)")
            return _backend
    except ImportError:
        pass

    # 2. Try simplejpeg (libjpeg-turbo)
    try:
        import simplejpeg  # noqa: F401
        _backend = "simplejpeg"
        log.info("JPEG encoder backend: simplejpeg (libjpeg-turbo)")
        return _backend
    except ImportError:
        pass

    # 3. OpenCV fallback
    _backend = "opencv"
    log.info("JPEG encoder backend: OpenCV (fallback)")
    return _backend


def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    """
    Encode a BGR numpy frame to JPEG bytes.
    Thread-safe. Uses the fastest available backend.
    """
    backend = _init_backend()

    if backend == "nvjpeg":
        try:
            import pynvjpeg
            import torch
            tensor = torch.from_numpy(frame[:, :, ::-1].copy()).cuda()  # BGR->RGB
            return pynvjpeg.encode(tensor, quality=quality)
        except Exception:
            # Fallback on any GPU error
            pass

    if backend == "simplejpeg":
        try:
            import simplejpeg
            # simplejpeg expects RGB; frame is BGR
            rgb = frame[:, :, ::-1]
            return simplejpeg.encode_jpeg(
                np.ascontiguousarray(rgb),
                quality=quality,
                colorspace="RGB",
                fastdct=True,
            )
        except Exception:
            pass

    # OpenCV fallback
    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def get_backend() -> str:
    """Return the active encoder backend name."""
    return _init_backend()
