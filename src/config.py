"""
Configuration management for ETL pipeline.
Uses Pydantic for type safety and validation.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""
    host: str = Field(..., env="POSTGRES_HOST")
    port: int = Field(5432, env="POSTGRES_PORT")
    database: str = Field(..., env="POSTGRES_DATABASE")
    user: str = Field(..., env="POSTGRES_USER")
    password: str = Field(..., env="POSTGRES_PASSWORD")


class SnowflakeConfig(BaseSettings):
    """Snowflake connection configuration."""
    account: str = Field(..., env="SNOWFLAKE_ACCOUNT")
    user: str = Field(..., env="SNOWFLAKE_USER")
    password: str = Field(..., env="SNOWFLAKE_PASSWORD")
    warehouse: str = Field(..., env="SNOWFLAKE_WAREHOUSE")
    database: str = Field(..., env="SNOWFLAKE_DATABASE")
    schema: str = Field("PUBLIC", env="SNOWFLAKE_SCHEMA")


class S3Config(BaseSettings):
    """AWS S3 configuration."""
    access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    region: str = Field(..., env="AWS_REGION")
    bucket: str = Field(..., env="S3_BUCKET")


class APIConfig(BaseSettings):
    """API configuration for data extraction."""
    base_url: str = Field(..., env="API_BASE_URL")
    api_key: str = Field(..., env="API_KEY")
    timeout: int = Field(30, env="API_TIMEOUT")


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field("INFO", env="LOG_LEVEL")
    file_path: str = Field("logs/etl_pipeline.log", env="LOG_FILE_PATH")


class ETLConfig(BaseSettings):
    """ETL pipeline configuration."""
    batch_size: int = Field(1000, env="BATCH_SIZE")
    max_retries: int = Field(3, env="MAX_RETRIES")
    retry_delay: int = Field(5, env="RETRY_DELAY")
    data_source_type: str = Field("api", env="DATA_SOURCE_TYPE")
    data_destination_type: str = Field("snowflake", env="DATA_DESTINATION_TYPE")


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_port: int = Field(8000, env="METRICS_PORT")


class Config(BaseSettings):
    """Main configuration class that combines all configs."""
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # Remove direct instantiation of sub-configs
    # Instead, instantiate them as needed
    @property
    def database(self):
        return DatabaseConfig()

    @property
    def snowflake(self):
        return SnowflakeConfig()

    @property
    def s3(self):
        return S3Config()

    @property
    def api(self):
        return APIConfig()

    @property
    def logging(self):
        return LoggingConfig()

    @property
    def etl(self):
        return ETLConfig()

    @property
    def monitoring(self):
        return MonitoringConfig()


def get_config():
    """Lazily load the main config object."""
    return Config() 