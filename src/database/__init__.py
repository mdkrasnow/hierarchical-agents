"""
Database integration layer for hierarchical agents system.

Provides async PostgreSQL connectivity, data access with permission filtering,
organizational queries, caching, and text chunking utilities.
"""

from .connection import (
    DatabaseConfig,
    DatabasePool,
    DatabaseConnectionError,
    get_database_pool,
    close_database_pool,
    create_database_config_from_env,
    execute_query,
    execute_query_one,
    execute_command
)

from .queries import (
    UserPermissions,
    EvaluationRecord,
    TeacherSummary,
    SchoolSummary,
    OrganizationTree,
    EvaluationQueries
)

from .cache import (
    CacheInterface,
    InMemoryCache,
    EvaluationCache,
    get_evaluation_cache,
    clear_evaluation_cache
)

__all__ = [
    # Connection
    'DatabaseConfig',
    'DatabasePool', 
    'DatabaseConnectionError',
    'get_database_pool',
    'close_database_pool',
    'create_database_config_from_env',
    'execute_query',
    'execute_query_one',
    'execute_command',
    
    # Data models and queries
    'UserPermissions',
    'EvaluationRecord',
    'TeacherSummary',
    'SchoolSummary',
    'OrganizationTree',
    'EvaluationQueries',
    
    # Caching
    'CacheInterface',
    'InMemoryCache',
    'EvaluationCache',
    'get_evaluation_cache',
    'clear_evaluation_cache',
]