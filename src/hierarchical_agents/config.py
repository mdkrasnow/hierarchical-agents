"""Configuration management for hierarchical agents."""

import os
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(..., env="DATABASE_URL")
    pool_min_size: int = Field(1, env="DATABASE_POOL_MIN_SIZE")
    pool_max_size: int = Field(10, env="DATABASE_POOL_MAX_SIZE")
    
    @validator("url")
    def validate_url(cls, v):
        """Validate database URL format."""
        if not v or not v.startswith("postgresql://"):
            raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string")
        if "[PASSWORD]" in v:
            raise ValueError("DATABASE_URL contains placeholder password - please set actual password")
        return v


class LLMConfig(BaseSettings):
    """LLM API configuration settings."""
    
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: float = Field(30.0, env="REQUEST_TIMEOUT")
    max_retries: int = Field(3, env="MAX_RETRIES")
    retry_delay: float = Field(1.0, env="RETRY_DELAY")


class AppConfig(BaseSettings):
    """Application configuration settings."""
    
    name: str = Field("hierarchical-agents", env="APP_NAME")
    version: str = Field("0.1.0", env="APP_VERSION")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    dev_mode: bool = Field(False, env="DEV_MODE")
    testing: bool = Field(False, env="TESTING")
    secret_key: str = Field(..., env="SECRET_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Settings(BaseSettings):
    """Main application settings."""
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    app: AppConfig = Field(default_factory=AppConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment."""
        return cls()


# Global settings instance
settings = Settings.load()