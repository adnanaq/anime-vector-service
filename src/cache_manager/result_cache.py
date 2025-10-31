"""
Result-level caching for crawler functions.

Since crawlers use browser automation (crawl4ai/Playwright) rather than HTTP libraries,
we cache the final extracted results instead of HTTP responses.

Schema hashing: Cache keys include a hash of the function's source code.
When code changes (CSS selectors, extraction logic, etc.), the hash changes,
automatically invalidating old cache entries.
"""

import functools
import hashlib
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from redis.asyncio import Redis

from .config import get_cache_config

if TYPE_CHECKING:
    from redis.asyncio import Redis as RedisType
else:
    RedisType = Redis  # type: ignore[misc,assignment]

# --- Singleton Redis Client for @cached_result ---

_redis_client: Optional["RedisType[str]"] = None


async def get_result_cache_redis_client() -> "RedisType[str]":
    """Initializes and returns a singleton async Redis client for result caching."""
    global _redis_client
    if _redis_client is None:
        config = get_cache_config()
        redis_url = config.redis_url or "redis://localhost:6379/0"
        logging.info(
            f"Initializing singleton Redis client for result cache: {redis_url}"
        )
        _redis_client = Redis.from_url(redis_url, decode_responses=True)
    return _redis_client


async def close_result_cache_redis_client() -> None:
    """Closes the singleton Redis client for result caching."""
    global _redis_client
    if _redis_client:
        logging.info("Closing singleton Redis client for result cache.")
        await _redis_client.close()
        _redis_client = None


# --- End Singleton ---


# Type variable for generic function return type
T = TypeVar("T")


def _compute_schema_hash(func: Callable[..., Any]) -> str:
    """
    Compute a hash of the function's source code.

    This hash changes whenever the function's implementation changes,
    automatically invalidating cached results when code is modified.

    Args:
        func: Function to hash

    Returns:
        8-character hexadecimal hash of the function's source code
    """
    try:
        # Get the source code of the function
        source = inspect.getsource(func)
        # Generate MD5 hash and take first 8 characters
        return hashlib.md5(source.encode()).hexdigest()[:8]
    except (OSError, TypeError):
        # If we can't get source (built-in, lambda, etc.), use function name
        return hashlib.md5(func.__name__.encode()).hexdigest()[:8]


def _generate_cache_key(
    prefix: str, schema_hash: str, *args: Any, **kwargs: Any
) -> str:
    """
    Generate a stable cache key from function arguments and schema hash.

    Args:
        prefix: Cache key prefix (usually function name)
        schema_hash: Hash of function's source code
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key string with format: result_cache:{prefix}:{schema_hash}:{args}
    """
    # Serialize arguments to create stable key
    key_parts = [prefix, schema_hash]  # Include schema hash

    # Add positional args
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # For complex types, use JSON serialization
            key_parts.append(json.dumps(arg, sort_keys=True))

    # Add keyword args (sorted for stability)
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, (str, int, float, bool, type(None))):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={json.dumps(v, sort_keys=True)}")

    # Hash the combined key if it's too long
    key_string = ":".join(key_parts)
    if len(key_string) > 200:
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"result_cache:{prefix}:{schema_hash}:{key_hash}"

    return f"result_cache:{key_string}"


def cached_result(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to cache async function results in Redis with automatic schema invalidation.

    The cache key includes a hash of the function's source code. When you modify
    the function (change CSS selectors, extraction logic, etc.), the hash changes
    and old cache entries are automatically invalidated.

    Usage:
        @cached_result(ttl=86400, key_prefix="animeplanet_anime")
        async def fetch_anime(slug: str) -> Optional[Dict[str, Any]]:
            # Expensive crawler operation
            return data

    Args:
        ttl: Time-to-live in seconds (None = use default from config)
        key_prefix: Optional custom key prefix (default: function name)

    Returns:
        Decorated function with caching
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Compute schema hash once when decorator is applied
        schema_hash = _compute_schema_hash(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache config
            config = get_cache_config()

            # Skip caching if disabled
            if not config.enabled or config.storage_type != "redis":
                return await func(*args, **kwargs)

            # Generate cache key with schema hash
            prefix = key_prefix or func.__name__
            cache_key = _generate_cache_key(prefix, schema_hash, *args, **kwargs)

            try:
                # Get singleton Redis client
                redis_client = await get_result_cache_redis_client()

                # Try to get from cache
                cached_data = await redis_client.get(cache_key)

                if cached_data:
                    # Cache hit - deserialize and return
                    return json.loads(cached_data)

                # Cache miss - execute function
                result = await func(*args, **kwargs)

                # Store in cache if result is not None
                if result is not None:
                    serialized = json.dumps(result, ensure_ascii=False)
                    cache_ttl = ttl if ttl is not None else 86400  # Default 24h
                    await redis_client.setex(cache_key, cache_ttl, serialized)

                return result

            except Exception as e:
                # On any error, just execute the function without caching
                logging.warning(f"Cache error in {func.__name__}: {e}")
                return await func(*args, **kwargs)

        return wrapper

    return decorator
