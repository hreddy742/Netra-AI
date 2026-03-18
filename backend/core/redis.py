"""
Async Redis connection pool — shared by all services.
"""
from __future__ import annotations

from functools import lru_cache
from typing import AsyncIterator

import redis.asyncio as aioredis

from backend.core.settings import get_settings


@lru_cache(maxsize=1)
def _pool() -> aioredis.ConnectionPool:
    s = get_settings()
    return aioredis.ConnectionPool.from_url(
        f"redis://{s.redis_host}:{s.redis_port}/{s.redis_db}",
        password=s.redis_password or None,
        max_connections=s.redis_pool_size,
        decode_responses=True,
    )


def get_redis() -> aioredis.Redis:
    """Return a Redis client from the shared pool."""
    return aioredis.Redis(connection_pool=_pool())


async def redis_lifespan() -> AsyncIterator[aioredis.Redis]:
    """Use as async context manager in FastAPI lifespan."""
    r = get_redis()
    try:
        yield r
    finally:
        await r.aclose()
