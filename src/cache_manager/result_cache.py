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
    """
    Close the module's singleton Redis client used for result caching.
    
    If a client instance exists, close its connection and reset the module-level client reference to None; otherwise do nothing.
    """
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
    Compute an 8-character MD5 hash representing a function's source code.
    
    If the function's source cannot be retrieved (built-in, dynamically created, etc.), the hash is derived from the function's name. This enables cache invalidation when a function's implementation changes.
    
    Parameters:
        func (Callable[..., Any]): The function whose source will be hashed.
    
    Returns:
        str: 8-character hexadecimal MD5 hash of the function's source or name.
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
    Generate a stable, namespaced cache key from a prefix, function schema hash, and call arguments.
    
    Parameters:
        prefix (str): Key prefix, typically the function name.
        schema_hash (str): Short hash representing the function's source to enable schema-based invalidation.
        *args: Positional arguments that influence the cache key; non-primitive values are JSON-serialized with sorted keys.
        **kwargs: Keyword arguments included in sorted order; non-primitive values are JSON-serialized with sorted keys.
    
    Returns:
        str: A namespaced cache key beginning with `result_cache:`. If the assembled key exceeds 200 characters, a SHA-256 digest of the key is used to keep the returned key length bounded.
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
    Decorator that caches a function's return value in Redis and invalidates cached entries when the function's source code changes.
    
    The cache key embeds an 8-character hash of the decorated function's source to ensure that changes to the function (for example, selector or extraction logic changes) cause prior cache entries to be ignored. The decorator respects the global cache configuration and will bypass caching if caching is disabled or the configured storage type is not Redis.
    
    Parameters:
        ttl (Optional[int]): Time-to-live for the cached entry in seconds. If `None`, a default TTL is used.
        key_prefix (Optional[str]): Custom prefix for the cache key; if omitted the decorated function's name is used.
    
    Returns:
        Callable: A decorator that wraps the target function with Redis-backed result caching.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Compute schema hash once when decorator is applied
        """
        Cache results of the decorated async function in Redis using a schema-based key to invalidate when the function's source changes.
        
        The returned wrapper will consult the global cache configuration and, when enabled with Redis storage, generate a stable cache key composed of the provided key prefix (or the function name), a schema hash computed from the function's source at decoration time, and the call arguments. On cache hit the wrapper returns the cached value (deserialized from JSON). On cache miss it executes the original function and, if the result is not None, serializes and stores it in Redis with the configured TTL (or 86400 seconds by default). Any caching errors or disabled configuration cause the wrapper to execute the original function without caching.
        
        Parameters:
            func (Callable[..., Any]): Async function to be decorated.
        
        Returns:
            Callable[..., Any]: An async wrapper that returns the decorated function's result, using Redis-backed caching when enabled.
        """
        schema_hash = _compute_schema_hash(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache config
            """
            Cache the wrapped coroutine's result in Redis using a schema-based key and an optional TTL.
            
            This wrapper checks the global cache configuration and only uses Redis when caching is enabled and storage_type is "redis". It builds a stable cache key from the provided key_prefix (or the wrapped function's name), the precomputed schema hash, and the call arguments. On a cache hit it returns the cached value (deserialized from JSON). On a cache miss it awaits the wrapped function, stores a JSON-serialized result in Redis if the result is not None using the specified TTL (or 86400 seconds by default), and returns the result. Any exceptions during caching are logged and the wrapped function is executed directly without caching.
            
            Parameters:
                *args: Positional arguments forwarded to the wrapped function and incorporated into the cache key.
                **kwargs: Keyword arguments forwarded to the wrapped function and incorporated into the cache key.
            
            Returns:
                The wrapped function's return value; when a cached entry is found, the deserialized cached value is returned.
            """
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