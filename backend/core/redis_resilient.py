"""
Netra AI — Resilient Redis client wrapper.

Wraps aioredis.Redis with:
  - Automatic reconnect on connection failure (exponential backoff, cap 30s)
  - Circuit breaker (open after consecutive_failures threshold)
  - Configurable retry on specific commands (xread, get, publish)
  - Health gauge update on state transitions
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Optional

import redis.asyncio as aioredis

from backend.core.logging import get_logger

log = get_logger("redis_resilient")

_BACKOFF_BASE   = 0.5   # seconds
_BACKOFF_CAP    = 30.0  # seconds
_CB_FAIL_THRESH = 5     # consecutive failures before opening circuit
_CB_RESET_AFTER = 60.0  # seconds before trying half-open


class CircuitState(Enum):
    CLOSED    = "closed"      # normal operation
    OPEN      = "open"        # failing; reject calls immediately
    HALF_OPEN = "half_open"   # one probe allowed


class ResilientRedis:
    """
    Drop-in wrapper around aioredis.Redis.
    Transparently reconnects and tracks circuit-breaker state.

    Usage:
        rr = ResilientRedis(url, decode_responses=True)
        await rr.connect()
        await rr.get("key")    # auto-reconnects on failure
        await rr.close()
    """

    def __init__(
        self,
        url: str,
        password: Optional[str] = None,
        decode_responses: bool = True,
        max_connections: int = 20,
        service_name: str = "redis",
    ) -> None:
        self._url              = url
        self._password         = password
        self._decode_responses = decode_responses
        self._max_connections  = max_connections
        self._service_name     = service_name

        self._client: Optional[aioredis.Redis] = None
        self._state             = CircuitState.CLOSED
        self._consecutive_fails = 0
        self._last_failure_ts   = 0.0
        self._lock              = asyncio.Lock()

    async def connect(self) -> None:
        """Initial connection — called once at startup."""
        self._client = self._make_client()
        await self._ping()

    def _make_client(self) -> aioredis.Redis:
        return aioredis.Redis.from_url(
            self._url,
            password=self._password or None,
            decode_responses=self._decode_responses,
            max_connections=self._max_connections,
        )

    async def _ping(self) -> None:
        assert self._client is not None
        await self._client.ping()

    async def _reconnect(self) -> None:
        """Attempt reconnect with exponential backoff."""
        attempt = 0
        while True:
            delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt))
            log.warning(
                "Redis reconnect attempt %d for %s (backoff=%.1fs)",
                attempt + 1, self._service_name, delay,
            )
            await asyncio.sleep(delay)
            try:
                if self._client:
                    try:
                        await self._client.aclose()
                    except Exception:
                        pass
                self._client = self._make_client()
                await self._ping()
                self._state             = CircuitState.CLOSED
                self._consecutive_fails = 0
                log.info("Redis reconnected: %s", self._service_name)
                return
            except Exception as exc:
                attempt += 1
                log.debug("Reconnect failed: %s", exc)

    def _check_circuit(self) -> bool:
        """Return True if the call should proceed, False if circuit is open."""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_ts >= _CB_RESET_AFTER:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow one probe
        return True

    def _record_success(self) -> None:
        self._consecutive_fails = 0
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            log.info("Circuit breaker closed for %s", self._service_name)

    def _record_failure(self) -> None:
        self._consecutive_fails += 1
        self._last_failure_ts = time.monotonic()
        if self._consecutive_fails >= _CB_FAIL_THRESH:
            if self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                log.error(
                    "Circuit breaker OPEN for %s (failures=%d)",
                    self._service_name, self._consecutive_fails,
                )

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a Redis command with reconnect on failure."""
        if not self._check_circuit():
            raise RuntimeError(f"Circuit breaker OPEN for {self._service_name}")

        try:
            assert self._client is not None
            result = await getattr(self._client, method)(*args, **kwargs)
            self._record_success()
            return result
        except (
            aioredis.ConnectionError,
            aioredis.TimeoutError,
            ConnectionRefusedError,
            OSError,
        ) as exc:
            self._record_failure()
            log.warning("Redis command %s failed: %s — reconnecting", method, exc)
            await self._reconnect()
            # Retry once after reconnect
            result = await getattr(self._client, method)(*args, **kwargs)
            self._record_success()
            return result

    # Proxy the most-used commands
    async def get(self, *a: Any, **kw: Any) -> Any:
        return await self._call("get", *a, **kw)

    async def set(self, *a: Any, **kw: Any) -> Any:
        return await self._call("set", *a, **kw)

    async def xread(self, *a: Any, **kw: Any) -> Any:
        return await self._call("xread", *a, **kw)

    async def xadd(self, *a: Any, **kw: Any) -> Any:
        return await self._call("xadd", *a, **kw)

    async def xtrim(self, *a: Any, **kw: Any) -> Any:
        return await self._call("xtrim", *a, **kw)

    async def publish(self, *a: Any, **kw: Any) -> Any:
        return await self._call("publish", *a, **kw)

    async def pubsub(self) -> Any:
        assert self._client is not None
        return self._client.pubsub()

    async def ping(self) -> Any:
        return await self._call("ping")

    @property
    def circuit_state(self) -> CircuitState:
        return self._state

    @property
    def raw(self) -> Optional[aioredis.Redis]:
        """Access underlying client (use sparingly)."""
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
