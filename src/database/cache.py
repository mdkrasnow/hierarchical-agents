"""
Caching interface for evaluation summaries and hierarchical agent results.

Provides in-memory caching with key patterns like eval:{id}, teacher:{id}, school:{id}
to improve performance for repeated queries during hierarchical agent processing.
"""

import json
import time
import asyncio
from typing import Any, Optional, Dict, Set, List
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import hashlib
import logging


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed: Optional[float] = None


class CacheInterface(ABC):
    """Abstract interface for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache with optional TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key from cache. Returns True if key existed."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        pass

    @abstractmethod
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally matching a pattern."""
        pass


class InMemoryCache(CacheInterface):
    """
    In-memory cache implementation with TTL support and LRU-like eviction.
    
    Features:
    - TTL-based expiration
    - Size-based eviction (LRU)
    - Pattern-based key operations
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 10000, default_ttl: Optional[int] = 3600):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        async with self._lock:
            await self._cleanup_expired()
            
            entry = self._cache.get(key)
            if not entry:
                return None
            
            # Check if expired
            if entry.expires_at and time.time() > entry.expires_at:
                del self._cache[key]
                return None
            
            # Update access tracking
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache with optional TTL."""
        async with self._lock:
            await self._cleanup_expired()
            await self._ensure_space()
            
            # Calculate expiration
            if ttl is None:
                ttl = self.default_ttl
            
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                access_count=0,
                last_accessed=time.time()
            )
            
            self._cache[key] = entry

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        async with self._lock:
            return self._cache.pop(key, None) is not None

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        value = await self.get(key)
        return value is not None

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally matching a pattern."""
        async with self._lock:
            await self._cleanup_expired()
            
            all_keys = list(self._cache.keys())
            
            if pattern is None:
                return all_keys
            
            # Simple pattern matching (supports * wildcards)
            import fnmatch
            return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            await self._cleanup_expired()
            
            total_entries = len(self._cache)
            total_memory = sum(len(str(entry.value)) for entry in self._cache.values())
            
            # Calculate hit rate (approximation)
            total_accesses = sum(entry.access_count for entry in self._cache.values())
            
            return {
                'total_entries': total_entries,
                'max_size': self.max_size,
                'utilization': total_entries / self.max_size if self.max_size > 0 else 0,
                'estimated_memory_bytes': total_memory,
                'total_accesses': total_accesses,
                'average_access_count': total_accesses / total_entries if total_entries > 0 else 0
            }

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and current_time > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _ensure_space(self) -> None:
        """Ensure there's space for new entries by evicting LRU items."""
        if len(self._cache) >= self.max_size:
            # Sort by last accessed time (LRU)
            lru_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed or 0
            )
            
            # Remove oldest 20% of entries to make space
            keys_to_remove = lru_keys[:max(1, len(lru_keys) // 5)]
            
            for key in keys_to_remove:
                del self._cache[key]
            
            logger.debug(f"Evicted {len(keys_to_remove)} LRU cache entries")


class EvaluationCache:
    """
    Specialized cache for evaluation-related data with standard key patterns.
    
    Key patterns:
    - eval:{evaluation_id}: Full evaluation data
    - eval_summary:{evaluation_id}: Evaluation agent summary
    - teacher:{teacher_name}:{school_name}: Teacher summary
    - school:{school_name}:{organization_id}: School summary
    - district:{organization_id}: District summary
    - user_perms:{user_id}: User permissions
    """

    def __init__(self, cache: Optional[CacheInterface] = None):
        """
        Initialize evaluation cache.
        
        Args:
            cache: Cache implementation to use (defaults to InMemoryCache)
        """
        self.cache = cache or InMemoryCache()

    def _make_key(self, prefix: str, *args: str) -> str:
        """Create a standardized cache key."""
        # Sanitize arguments and create consistent key
        sanitized_args = []
        for arg in args:
            if arg is None:
                arg = "none"
            # Replace problematic characters
            sanitized = str(arg).replace(":", "_").replace(" ", "_").lower()
            sanitized_args.append(sanitized)
        
        return f"{prefix}:{':'.join(sanitized_args)}"

    def _hash_key(self, key: str) -> str:
        """Create a hash of the key to handle long/complex keys."""
        if len(key) > 250:  # Reasonable key length limit
            return hashlib.md5(key.encode()).hexdigest()
        return key

    # Evaluation-level caching
    async def get_evaluation(self, evaluation_id: str) -> Optional[Any]:
        """Get cached evaluation data."""
        key = self._hash_key(self._make_key("eval", evaluation_id))
        return await self.cache.get(key)

    async def set_evaluation(self, evaluation_id: str, evaluation_data: Any, ttl: int = 3600) -> None:
        """Cache evaluation data."""
        key = self._hash_key(self._make_key("eval", evaluation_id))
        await self.cache.set(key, evaluation_data, ttl)

    async def get_evaluation_summary(self, evaluation_id: str) -> Optional[Any]:
        """Get cached evaluation agent summary."""
        key = self._hash_key(self._make_key("eval_summary", evaluation_id))
        return await self.cache.get(key)

    async def set_evaluation_summary(self, evaluation_id: str, summary: Any, ttl: int = 7200) -> None:
        """Cache evaluation agent summary."""
        key = self._hash_key(self._make_key("eval_summary", evaluation_id))
        await self.cache.set(key, summary, ttl)

    # Teacher-level caching
    async def get_teacher_summary(self, teacher_name: str, school_name: str) -> Optional[Any]:
        """Get cached teacher summary."""
        key = self._hash_key(self._make_key("teacher", teacher_name, school_name))
        return await self.cache.get(key)

    async def set_teacher_summary(self, teacher_name: str, school_name: str, summary: Any, ttl: int = 1800) -> None:
        """Cache teacher summary."""
        key = self._hash_key(self._make_key("teacher", teacher_name, school_name))
        await self.cache.set(key, summary, ttl)

    # School-level caching
    async def get_school_summary(self, school_name: str, organization_id: str) -> Optional[Any]:
        """Get cached school summary."""
        key = self._hash_key(self._make_key("school", school_name, organization_id))
        return await self.cache.get(key)

    async def set_school_summary(self, school_name: str, organization_id: str, summary: Any, ttl: int = 900) -> None:
        """Cache school summary."""
        key = self._hash_key(self._make_key("school", school_name, organization_id))
        await self.cache.set(key, summary, ttl)

    # District-level caching
    async def get_district_summary(self, organization_id: str) -> Optional[Any]:
        """Get cached district summary."""
        key = self._hash_key(self._make_key("district", organization_id))
        return await self.cache.get(key)

    async def set_district_summary(self, organization_id: str, summary: Any, ttl: int = 600) -> None:
        """Cache district summary."""
        key = self._hash_key(self._make_key("district", organization_id))
        await self.cache.set(key, summary, ttl)

    # User permissions caching
    async def get_user_permissions(self, user_id: str) -> Optional[Any]:
        """Get cached user permissions."""
        key = self._hash_key(self._make_key("user_perms", user_id))
        return await self.cache.get(key)

    async def set_user_permissions(self, user_id: str, permissions: Any, ttl: int = 1800) -> None:
        """Cache user permissions."""
        key = self._hash_key(self._make_key("user_perms", user_id))
        await self.cache.set(key, permissions, ttl)

    # Invalidation methods
    async def invalidate_teacher(self, teacher_name: str, school_name: str) -> None:
        """Invalidate all cache entries for a teacher."""
        patterns = [
            f"teacher:{teacher_name.lower().replace(' ', '_')}:*",
            f"school:{school_name.lower().replace(' ', '_')}:*",
        ]
        
        for pattern in patterns:
            keys = await self.cache.keys(pattern)
            for key in keys:
                await self.cache.delete(key)

    async def invalidate_school(self, school_name: str, organization_id: str) -> None:
        """Invalidate all cache entries for a school."""
        patterns = [
            f"school:{school_name.lower().replace(' ', '_')}:*",
            f"district:{organization_id}*",
        ]
        
        for pattern in patterns:
            keys = await self.cache.keys(pattern)
            for key in keys:
                await self.cache.delete(key)

    async def invalidate_organization(self, organization_id: str) -> None:
        """Invalidate all cache entries for an organization."""
        pattern = f"district:{organization_id}*"
        keys = await self.cache.keys(pattern)
        for key in keys:
            await self.cache.delete(key)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if hasattr(self.cache, 'get_cache_stats'):
            return await self.cache.get_cache_stats()
        
        # Fallback basic stats
        keys = await self.cache.keys()
        return {
            'total_entries': len(keys),
            'key_patterns': {
                'eval': len([k for k in keys if k.startswith('eval:')]),
                'eval_summary': len([k for k in keys if k.startswith('eval_summary:')]),
                'teacher': len([k for k in keys if k.startswith('teacher:')]),
                'school': len([k for k in keys if k.startswith('school:')]),
                'district': len([k for k in keys if k.startswith('district:')]),
                'user_perms': len([k for k in keys if k.startswith('user_perms:')]),
            }
        }


# Global cache instance
_global_cache: Optional[EvaluationCache] = None


async def get_evaluation_cache() -> EvaluationCache:
    """Get the global evaluation cache instance."""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = EvaluationCache()
    
    return _global_cache


async def clear_evaluation_cache() -> None:
    """Clear the global evaluation cache."""
    cache = await get_evaluation_cache()
    await cache.cache.clear()