"""
CameraAlertCooldown — per-camera sliding window rate limiter using Redis sorted set + Lua.

Prevents alert floods that cause operators to mute notifications.

Key: netra:cooldown:window:{camera_id}  (sorted set, score=timestamp)

Usage:
    cooldown = CameraAlertCooldown(redis)
    if not await cooldown.try_dispatch("cam-01"):
        continue  # suppressed
    await dispatcher.dispatch(incident)
"""
from __future__ import annotations

import random
import time

import redis.asyncio as aioredis

from backend.core.logging import get_logger
from backend.core.settings import get_settings

log      = get_logger("cooldown")
settings = get_settings()

_KEY_PREFIX = "netra:cooldown:window:"

# Atomically: remove expired entries, count remaining, add if under limit.
# Returns 1 if allowed, 0 if suppressed.
_COOLDOWN_LUA = """
local key      = KEYS[1]
local now      = tonumber(ARGV[1])
local window   = tonumber(ARGV[2])
local max_alerts = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < max_alerts then
    redis.call('ZADD', key, now, now .. '-' .. math.random(1000000))
    redis.call('PEXPIRE', key, window * 1000)
    return 1
end
return 0
"""


def _key(camera_id: str) -> str:
    return f"{_KEY_PREFIX}{camera_id}"


class CameraAlertCooldown:

    def __init__(
        self,
        redis: aioredis.Redis,
        window_seconds: int | None = None,
        max_alerts: int | None = None,
        burst: int | None = None,
        # legacy param kept for test compat
        cooldown_sec: int | None = None,
    ) -> None:
        self._redis   = redis
        self._window  = window_seconds if window_seconds is not None \
                        else settings.alert_cooldown_window_seconds
        self._max     = max_alerts if max_alerts is not None \
                        else settings.alert_cooldown_max_alerts
        self._burst   = burst if burst is not None \
                        else settings.alert_cooldown_burst
        # legacy cooldown_sec overrides window for backward-compat tests
        if cooldown_sec is not None:
            self._window = cooldown_sec
        self._script: object = None

    async def _get_script(self) -> object:
        if self._script is None:
            self._script = self._redis.register_script(_COOLDOWN_LUA)
        return self._script

    async def try_dispatch(self, camera_id: str, burst: bool = False) -> bool:
        """Atomic check-and-record. Returns True if alert should be dispatched."""
        limit  = self._burst if burst else self._max
        now    = time.time()
        script = await self._get_script()
        result = await script(keys=[_key(camera_id)], args=[now, self._window, limit])
        allowed = bool(result)
        if not allowed:
            log.debug("Alert suppressed camera=%s window=%ss", camera_id, self._window)
        return allowed

    # ------------------------------------------------------------------
    # Backward-compatible helpers (used by existing tests + main.py)
    # ------------------------------------------------------------------

    async def is_suppressed(self, camera_id: str, track_id: int | None = None) -> bool:
        """Read-only check. Prefer try_dispatch() on the hot path."""
        now = time.time()
        await self._redis.zremrangebyscore(_key(camera_id), 0, now - self._window)
        count = await self._redis.zcard(_key(camera_id))
        return count >= self._max

    async def mark_alerted(self, camera_id: str, track_id: int | None = None) -> None:
        """Non-atomic record. Prefer try_dispatch() on the hot path."""
        now = time.time()
        k   = _key(camera_id)
        await self._redis.zadd(k, {f"{now}-{random.randint(0, 1_000_000)}": now})
        await self._redis.pexpire(k, int(self._window * 1000))

    async def clear(self, camera_id: str, track_id: int | None = None) -> None:
        await self._redis.delete(_key(camera_id))

    async def remaining_sec(self, camera_id: str, track_id: int | None = None) -> int:
        ttl = await self._redis.pttl(_key(camera_id))
        return max(0, ttl // 1000)
