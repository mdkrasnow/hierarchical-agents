"""Tests for configuration management."""

import os
import pytest
from unittest.mock import patch

from hierarchical_agents.config import DatabaseConfig, LLMConfig, AppConfig


def test_database_config_valid():
    """Test valid database configuration."""
    with patch.dict(os.environ, {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/test",
        "DATABASE_POOL_MIN_SIZE": "2",
        "DATABASE_POOL_MAX_SIZE": "20"
    }):
        config = DatabaseConfig()
        assert config.url == "postgresql://user:pass@localhost:5432/test"
        assert config.pool_min_size == 2
        assert config.pool_max_size == 20


def test_database_config_invalid_url():
    """Test invalid database URL."""
    with patch.dict(os.environ, {
        "DATABASE_URL": "invalid-url"
    }):
        with pytest.raises(ValueError, match="must be a valid PostgreSQL"):
            DatabaseConfig()


def test_database_config_placeholder_password():
    """Test database URL with placeholder password."""
    with patch.dict(os.environ, {
        "DATABASE_URL": "postgresql://user:[PASSWORD]@localhost:5432/test"
    }):
        with pytest.raises(ValueError, match="contains placeholder password"):
            DatabaseConfig()


def test_llm_config_defaults():
    """Test LLM configuration with defaults."""
    config = LLMConfig()
    assert config.max_concurrent_requests == 10
    assert config.request_timeout == 30.0
    assert config.max_retries == 3
    assert config.retry_delay == 1.0


def test_app_config_defaults():
    """Test app configuration with defaults."""
    with patch.dict(os.environ, {
        "SECRET_KEY": "test-secret-key"
    }):
        config = AppConfig()
        assert config.name == "hierarchical-agents"
        assert config.version == "0.1.0"
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.secret_key == "test-secret-key"