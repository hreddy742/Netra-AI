"""
Netra AI — Shared Memory ring buffer.

Zero-copy frame transport between mediabridge and inference service.
Ported from pipeline_opencv with generation-based stale-read guard added.
"""
from __future__ import annotations

import ctypes
import multiprocessing.shared_memory as mp_shm
from typing import Final

import numpy as np


SLOT_DTYPE: Final = np.uint8


class RingBufferWriter:
    """
    Written by mediabridge.
    Allocates SHM, writes frames slot by slot, bumps generation counter.
    """

    def __init__(self, camera_id: str, num_slots: int, height: int, width: int) -> None:
        self.camera_id   = camera_id
        self.num_slots   = num_slots
        self.frame_shape = (height, width, 3)
        self.frame_bytes = int(np.prod(self.frame_shape))
        self.shm_name    = f"netra_cam_{camera_id}_ring"
        self.idx_name    = f"netra_cam_{camera_id}_idx"
        self.gen_name    = f"netra_cam_{camera_id}_gen"

        self._shm     = self._alloc(self.shm_name, num_slots * self.frame_bytes)
        self._idx_shm = self._alloc(self.idx_name, ctypes.sizeof(ctypes.c_int64))
        self._gen_shm = self._alloc(self.gen_name, num_slots * ctypes.sizeof(ctypes.c_int64))

        self._idx_arr = np.ndarray((1,),          dtype=np.int64, buffer=self._idx_shm.buf)
        self._gen_arr = np.ndarray((num_slots,),   dtype=np.int64, buffer=self._gen_shm.buf)
        self._idx_arr[0] = 0
        self._gen_arr.fill(0)

    @staticmethod
    def _alloc(name: str, size: int) -> mp_shm.SharedMemory:
        try:
            return mp_shm.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            return mp_shm.SharedMemory(name=name, create=False)

    def write(self, frame: np.ndarray) -> tuple[int, int]:
        """Write frame; return (slot_index, generation)."""
        slot   = int(self._idx_arr[0]) % self.num_slots
        offset = slot * self.frame_bytes
        view   = np.ndarray(self.frame_shape, dtype=SLOT_DTYPE, buffer=self._shm.buf, offset=offset)
        np.copyto(view, frame)
        self._idx_arr[0] += 1
        self._gen_arr[slot] += 1
        return slot, int(self._gen_arr[slot])

    def close(self) -> None:
        for shm in (self._shm, self._idx_shm, self._gen_shm):
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, PermissionError):
                pass


class RingBufferReader:
    """
    Read by inference.
    Attaches to existing SHM; validates generation to catch stale pointers.
    """

    def __init__(self, camera_id: str, num_slots: int, height: int, width: int) -> None:
        self.camera_id   = camera_id
        self.num_slots   = num_slots
        self.frame_shape = (height, width, 3)
        self.frame_bytes = int(np.prod(self.frame_shape))
        shm_name         = f"netra_cam_{camera_id}_ring"
        gen_name         = f"netra_cam_{camera_id}_gen"

        try:
            self._shm     = mp_shm.SharedMemory(name=shm_name, create=False)
            self._gen_shm = mp_shm.SharedMemory(name=gen_name, create=False)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"SHM segment '{shm_name}' not found — is mediabridge running for camera {camera_id}?"
            )
        self._gen_arr = np.ndarray((num_slots,), dtype=np.int64, buffer=self._gen_shm.buf)

    def read(self, slot_index: int, generation: int) -> np.ndarray:
        """Return a copy of the frame; raises RuntimeError if stale."""
        slot   = slot_index % self.num_slots
        if int(self._gen_arr[slot]) != generation:
            raise RuntimeError(
                f"Stale frame: camera={self.camera_id} slot={slot_index} expected_gen={generation} "
                f"actual_gen={int(self._gen_arr[slot])}"
            )
        offset = slot * self.frame_bytes
        view   = np.ndarray(self.frame_shape, dtype=SLOT_DTYPE, buffer=self._shm.buf, offset=offset)
        frame  = view.copy()   # defensive copy to avoid race during long inference
        # second generation check after copy
        if int(self._gen_arr[slot]) != generation:
            raise RuntimeError(
                f"Frame overwritten during read: camera={self.camera_id} slot={slot_index}"
            )
        return frame

    def close(self) -> None:
        try:
            self._shm.close()
            self._gen_shm.close()
        except Exception:
            pass


class ReaderCache:
    """Thread-safe per-process cache so each camera attaches SHM only once."""
    _readers: dict[str, RingBufferReader] = {}

    @classmethod
    def get(cls, camera_id: str, num_slots: int, height: int, width: int) -> RingBufferReader:
        if camera_id not in cls._readers:
            import time as _time
            exc: Exception = FileNotFoundError("not yet initialised")
            for _ in range(5):
                try:
                    cls._readers[camera_id] = RingBufferReader(camera_id, num_slots, height, width)
                    return cls._readers[camera_id]
                except FileNotFoundError as e:
                    exc = e
                    _time.sleep(1.0)
            raise exc
        return cls._readers[camera_id]

    @classmethod
    def clear(cls) -> None:
        for r in cls._readers.values():
            r.close()
        cls._readers.clear()
