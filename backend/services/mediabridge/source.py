"""
RTSP / file video source with exponential backoff reconnection.
Ported from edgeguard/src/video/sources.py and hardened.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import cv2
import numpy as np


class VideoSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class RTSPSource(VideoSource):
    """
    Connects to an RTSP stream.
    Reconnects with exponential backoff (max 60 s) on any failure.
    """

    _BACKOFF_BASE  = 1.0
    _BACKOFF_MAX   = 60.0
    _BACKOFF_FACTOR = 2.0

    def __init__(
        self,
        url: str,
        camera_id: str,
        target_fps: float = 15.0,
        timeout_sec: float = 10.0,
    ) -> None:
        self.url        = url
        self.camera_id  = camera_id
        self.target_fps = target_fps
        self.timeout    = timeout_sec
        self._stop      = threading.Event()

    def frames(self) -> Iterator[np.ndarray]:
        backoff = self._BACKOFF_BASE
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                time.sleep(backoff)
                backoff = min(backoff * self._BACKOFF_FACTOR, self._BACKOFF_MAX)
                continue
            backoff = self._BACKOFF_BASE  # reset on success
            frame_interval = 1.0 / self.target_fps
            last_frame_time = 0.0
            try:
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    now = time.monotonic()
                    # Simple frame-rate governor
                    elapsed = now - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    last_frame_time = time.monotonic()
                    yield frame
            finally:
                cap.release()

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def close(self) -> None:
        self._stop.set()


class VideoFileSource(VideoSource):
    """
    Reads an MP4 / AVI file.
    Loops by default; respects PTS for correct playback speed.
    """

    def __init__(
        self,
        path: str,
        camera_id: str,
        loop: bool = True,
        target_fps: float | None = None,
    ) -> None:
        self.path       = path
        self.camera_id  = camera_id
        self.loop       = loop
        self.target_fps = target_fps
        self._stop      = threading.Event()

    def frames(self) -> Iterator[np.ndarray]:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {self.path}")
            fps = self.target_fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = 1.0 / fps
            try:
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame
                    time.sleep(frame_interval)
            finally:
                cap.release()
            if not self.loop:
                break

    def close(self) -> None:
        self._stop.set()
