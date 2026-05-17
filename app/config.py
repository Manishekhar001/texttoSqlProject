"""
Configuration management using Pydantic Settings.
Loads environment variables from .env file.
"""

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Multi-Source RAG + Text to SQL"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    ROOT_PATH: str = ""

    # OPENAI Configuration
    OPENAI_API_KEY: Optional[str] = None

    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX: str = "rag-cache-docsqa"

    # Supabase/PostgreSQL Configuration
    DATABASE_URL: Optional[str] = None

    # Opik monitoring
    OPIK_API_KEY: Optional[str] = None
    OPIK_WORKSPACE: Optional[str] = None
    OPIK_PROJECT_NAME: str = "RAG-Text2Sql"

    # Vanna 2.0 Configuration (Text2Sql)
    VANNA_MODEL: str = "gpt-4o"
    VANNA_PINECONE_INDEX: str = "vanna-sql-training"
    VANNA_NAMESPACE: str = "sql-agent"

    # SQL LLM Configuration for Determinism
    VANNA_TEMPERATURE: float = (
        0.0  # 0.0 = fully deterministic, 1.0 = creative (range: 0.0-2.0)
    )
    VANNA_TOP_P: float = 0.1  # Nucleus sampling threshold (range: 0.0-1.0)
    VANNA_SEED: int = 42  # Random seed for reproducibility
    VANNA_MAX_TOKENS: int = 2000  # Maximum tokens for SQL generation

    # Text chunking Configuration
    CHUNK_SIZE: int = 512
    MIN_CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 50

    # Document Processing Configuration
    USE_DOCLING: bool = True

    # StorageBackend Configuration
    STORAGE_BACKEND: str = "s3"  # Options: "local", "s3"
    # UPLOAD_DIR: str = "data/uploads"
    # CACHE_DIR: str = "data/cached_chunks"

    # Storage paths (auto-detects Lambda environment)
    @property
    def UPLOAD_DIR(self) -> str:
        # Use /tmp in Lambda/production, data/ locally
        if self.ENVIRONMENT == "production" or self.STORAGE_BACKEND == "s3":
            return "/tmp/uploads"
        return "data/uploads"

    @property
    def CACHE_DIR(self) -> str:
        # Use /tmp in Lambda/production, data/ locally
        if self.ENVIRONMENT == "production" or self.STORAGE_BACKEND == "s3":
            return "/tmp/cached_chunks"
        return "data/cached_chunks"

    # S3 Storage Configuration (for Lambda deployment)
    S3_CACHE_BUCKET: str = "rag-cache-docsqa"
    AWS_REGION: str = "us-east-1"
    # AWS credentials from environment or IAM role (recommended for Lambda)
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are read automatically by boto3
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # Upstash Redis Configuration (Query-level caching)
    UPSTASH_REDIS_URL: Optional[str] = None  # Optional - app works without caching
    UPSTASH_REDIS_TOKEN: Optional[str] = None  # Optional - app works without caching

    # Cache TTL Configuration (in seconds)
    CACHE_TTL_EMBEDDINGS: int = 604800  # 7 days - embeddings are static
    CACHE_TTL_RAG: int = 3600  # 1 hour - may change with new documents
    CACHE_TTL_SQL_GEN: int = 86400  # 24 hours - schema relatively stable
    CACHE_TTL_SQL_RESULT: int = 900  # 15 minutes - data changes frequently

    @property
    def is_lambda(self) -> bool:
        """Check if running in AWS Lambda environment."""
        return os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    # Global settings instance


settings = Settings()
