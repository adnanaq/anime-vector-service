"""
HTTP Cache Manager for enrichment pipeline.

Provides cached HTTP sessions for aiohttp (async) clients.
Supports Redis (multi-agent) and SQLite (single-agent) storage backends.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import hishel
from redis.asyncio import Redis as AsyncRedis

from .config import CacheConfig

logger = logging.getLogger(__name__)


class HTTPCacheManager:
    """Manages HTTP cache for enrichment pipeline with Redis/SQLite backends."""

    def __init__(self, config: CacheConfig):
        """
        Initialize HTTPCacheManager with the given cache configuration.
        
        Parameters:
            config (CacheConfig): Configuration controlling cache enablement and backend settings. If `config.enabled` is True, the appropriate storage backend is initialized.
        """
        self.config = config
        self._storage: Optional[Any] = None
        self._async_redis_client: Optional[AsyncRedis[bytes]] = None

        if self.config.enabled:
            self._init_storage()

    def _init_storage(self) -> None:
        """
        Initialize the cache storage backend based on the manager configuration.
        
        Chooses the storage initializer according to `self.config.storage_type`:
        - "redis": initialize Redis-backed async storage
        - "sqlite": initialize local SQLite storage
        
        Raises:
            ValueError: If `self.config.storage_type` is not "redis" or "sqlite".
        """
        if self.config.storage_type == "redis":
            self._init_redis_storage()
        elif self.config.storage_type == "sqlite":
            self._init_sqlite_storage()
        else:
            raise ValueError(f"Unknown storage type: {self.config.storage_type}")

    def _init_redis_storage(self) -> None:
        """
        Initialize and store an asynchronous Redis client for aiohttp session caching.
        
        If `self.config.redis_url` is provided and the client is created successfully, sets
        `self._async_redis_client` to the new AsyncRedis client; if `redis_url` is missing
        or initialization fails, leaves `self._async_redis_client` as `None`.
        """
        try:
            if not self.config.redis_url:
                raise ValueError("redis_url required for Redis storage")

            # Initialize async client for aiohttp sessions
            self._async_redis_client = AsyncRedis.from_url(
                self.config.redis_url, decode_responses=False
            )
            logger.info(
                f"Async Redis client initialized for aiohttp sessions: {self.config.redis_url}"
            )

        except (ValueError, Exception) as e:
            logger.warning(
                f"Async Redis client initialization failed: {e}. "
                "Async (aiohttp) requests will not be cached on Redis."
            )

    def _init_sqlite_storage(self) -> None:
        """
        Create and initialize a file-based SQLite cache storage for HTTP sessions.
        
        Creates the configured cache directory (including parents), builds an absolute
        path to an http_cache.db file inside that directory, and sets self._storage to
        a hishel.SyncSqliteStorage instance using that database path. Logs the
        initialized database path.
        """
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use Hishel 1.0 SQLite storage with absolute path
        database_path = (cache_dir / "http_cache.db").absolute()
        self._storage = hishel.SyncSqliteStorage(database_path=str(database_path))
        logger.info(f"SQLite cache initialized: {database_path}")

    def get_aiohttp_session(
        self, service: str, **session_kwargs: Any
    ) -> Any:  # Returns aiohttp.ClientSession or CachedAiohttpSession
        """
        Provide an aiohttp session for the named service, using an async Redis-backed cached session when caching is enabled and available.
        
        Parameters:
            service (str): Service identifier (e.g., "jikan", "anilist", "anidb") used to determine per-service TTL.
            **session_kwargs: Additional keyword arguments passed to aiohttp.ClientSession.
        
        Returns:
            An aiohttp.ClientSession or a CachedAiohttpSession that applies Redis-backed HTTP response caching (including body-based caching).
        """
        if (
            not self.config.enabled
            or self.config.storage_type != "redis"
            or not self._async_redis_client
        ):
            return aiohttp.ClientSession(**session_kwargs)

        # Create async Redis storage for aiohttp caching
        try:
            from src.cache_manager.aiohttp_adapter import CachedAiohttpSession
            from src.cache_manager.async_redis_storage import AsyncRedisStorage

            # Get service-specific TTL
            ttl = self._get_service_ttl(service)

            # Create async storage from shared client
            async_storage = AsyncRedisStorage(
                client=self._async_redis_client,
                default_ttl=float(ttl),
                refresh_ttl_on_access=True,
                key_prefix="hishel_cache",
            )

            # Enable body-based caching by adding X-Hishel-Body-Key header
            # This ensures POST requests (GraphQL, etc.) include body in cache key
            headers = session_kwargs.get("headers", {})
            headers["X-Hishel-Body-Key"] = "true"
            session_kwargs["headers"] = headers

            # Return cached session
            logger.info(
                f"Async Redis cache initialized for {service} (TTL: {ttl}s, body-based caching: enabled)"
            )
            return CachedAiohttpSession(storage=async_storage, **session_kwargs)

        except ImportError as e:
            logger.warning(f"Async caching dependencies missing: {e}")
            return aiohttp.ClientSession(**session_kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize async cache: {e}")
            return aiohttp.ClientSession(**session_kwargs)

    def _get_service_ttl(self, service: str) -> int:
        """
        Return the cache time-to-live (TTL) in seconds for the given service.
        
        Looks up the `ttl_<service>` attribute on the manager's config and returns its value; if not present, returns 86400 (24 hours).
        
        Parameters:
            service (str): Service identifier used to form the config attribute name.
        
        Returns:
            int: TTL for the service in seconds.
        """
        ttl_attr = f"ttl_{service}"
        return getattr(self.config, ttl_attr, 86400)  # Default 24 hours

    def close(self) -> None:
        """
        No-op synchronous cleanup hook for cache resources.
        
        This method performs no action; asynchronous resources (for example the Redis client) are closed by `close_async`.
        """

    async def close_async(self) -> None:
        """
        Close any open asynchronous cache connections.
        
        If an async Redis client was initialized on this manager, attempt to close it. Exceptions raised while closing are caught and suppressed (a warning is logged).
        """
        if self._async_redis_client:
            try:
                await self._async_redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing async Redis client: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Report whether HTTP caching is enabled and the active storage configuration.
        
        Returns:
            dict: A mapping with key "enabled" (bool). If enabled is True the mapping also includes
            "storage_type" (str), "cache_dir" (str or None) when storage_type is "sqlite", and
            "redis_url" (str or None) when storage_type is "redis". If caching is disabled the
            mapping is {"enabled": False}.
        """
        if not self.config.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "storage_type": self.config.storage_type,
            "cache_dir": (
                self.config.cache_dir if self.config.storage_type == "sqlite" else None
            ),
            "redis_url": (
                self.config.redis_url if self.config.storage_type == "redis" else None
            ),
        }