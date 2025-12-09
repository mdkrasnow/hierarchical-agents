"""
Database connection and pooling for Supabase PostgreSQL.

Provides async connection pooling and connection management for the hierarchical agents system.
Handles connection lifecycle, error recovery, and proper resource cleanup.
"""

import asyncio
import asyncpg
import os
from typing import Optional, Any, Dict, List
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str
    port: int
    database: str
    user: str
    password: str
    min_size: int = 5
    max_size: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class DatabasePool:
    """
    Async PostgreSQL connection pool manager for Supabase.
    
    Provides connection pooling, automatic retries, and proper resource management
    for concurrent access patterns needed by hierarchical agents.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._is_closed = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            return

        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                max_queries=self.config.max_queries,
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                command_timeout=60,
                server_settings={
                    'jit': 'off',  # Disable JIT for better connection stability
                }
            )
            logger.info(f"Database pool initialized with {self.config.min_size}-{self.config.max_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}") from e

    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        if self._pool and not self._is_closed:
            await self._pool.close()
            self._is_closed = True
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire_connection(self):
        """
        Acquire a database connection from the pool.
        
        Usage:
            async with pool.acquire_connection() as conn:
                result = await conn.fetch("SELECT * FROM table")
        """
        if self._pool is None:
            raise DatabaseConnectionError("Database pool not initialized")

        if self._is_closed:
            raise DatabaseConnectionError("Database pool is closed")

        connection = None
        try:
            connection = await self._pool.acquire()
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                try:
                    await self._pool.release(connection)
                except Exception as e:
                    logger.warning(f"Error releasing connection: {e}")

    async def execute_query(self, query: str, *args) -> List[asyncpg.Record]:
        """Execute a query and return all results."""
        async with self.acquire_connection() as conn:
            try:
                return await conn.fetch(query, *args)
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Args: {args}")
                raise

    async def execute_query_one(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Execute a query and return one result or None."""
        async with self.acquire_connection() as conn:
            try:
                return await conn.fetchrow(query, *args)
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Args: {args}")
                raise

    async def execute_command(self, command: str, *args) -> str:
        """Execute a command (INSERT, UPDATE, DELETE) and return status."""
        async with self.acquire_connection() as conn:
            try:
                return await conn.execute(command, *args)
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                logger.error(f"Command: {command}")
                logger.error(f"Args: {args}")
                raise

    @property
    def is_initialized(self) -> bool:
        """Check if the pool is initialized."""
        return self._pool is not None and not self._is_closed

    async def health_check(self) -> bool:
        """Perform a health check on the database connection."""
        try:
            result = await self.execute_query_one("SELECT 1 as health_check")
            return result is not None and result['health_check'] == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


def create_database_config_from_env() -> DatabaseConfig:
    """
    Create database configuration from environment variables.
    
    Expected environment variable:
    DATABASE_URL=postgresql://user:password@host:port/database
    
    Or individual variables:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
    """
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        # Parse DATABASE_URL
        # Format: postgresql://user:password@host:port/database
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            
            if not all([parsed.hostname, parsed.port, parsed.username, parsed.password, parsed.path]):
                raise ValueError("Incomplete DATABASE_URL")
                
            return DatabaseConfig(
                host=parsed.hostname,
                port=parsed.port,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path.lstrip('/'),
                min_size=int(os.getenv('DB_POOL_MIN_SIZE', '5')),
                max_size=int(os.getenv('DB_POOL_MAX_SIZE', '20')),
            )
        except Exception as e:
            raise ValueError(f"Invalid DATABASE_URL format: {e}")
    else:
        # Use individual environment variables
        host = os.getenv('DB_HOST')
        port = int(os.getenv('DB_PORT', '5432'))
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        database = os.getenv('DB_NAME')
        
        if not all([host, user, password, database]):
            raise ValueError("Missing required database environment variables")
            
        return DatabaseConfig(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            min_size=int(os.getenv('DB_POOL_MIN_SIZE', '5')),
            max_size=int(os.getenv('DB_POOL_MAX_SIZE', '20')),
        )


# Global pool instance - initialized once per application
_global_pool: Optional[DatabasePool] = None


async def get_database_pool() -> DatabasePool:
    """
    Get the global database pool instance, initializing it if needed.
    
    This should be called during application startup to ensure the pool
    is properly initialized before any database operations.
    """
    global _global_pool
    
    if _global_pool is None:
        config = create_database_config_from_env()
        _global_pool = DatabasePool(config)
        await _global_pool.initialize()
    elif not _global_pool.is_initialized:
        await _global_pool.initialize()
    
    return _global_pool


async def close_database_pool() -> None:
    """Close the global database pool. Call during application shutdown."""
    global _global_pool
    
    if _global_pool:
        await _global_pool.close()
        _global_pool = None


# Convenience functions for common operations
async def execute_query(query: str, *args) -> List[asyncpg.Record]:
    """Execute a query using the global pool."""
    pool = await get_database_pool()
    return await pool.execute_query(query, *args)


async def execute_query_one(query: str, *args) -> Optional[asyncpg.Record]:
    """Execute a query and return one result using the global pool."""
    pool = await get_database_pool()
    return await pool.execute_query_one(query, *args)


async def execute_command(command: str, *args) -> str:
    """Execute a command using the global pool."""
    pool = await get_database_pool()
    return await pool.execute_command(command, *args)