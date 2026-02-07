"""
Application settings using Pydantic Settings management.
Loads configuration from environment variables and .env file.
"""

from functools import lru_cache
from typing import List, Optional
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===========================================
    # Application Settings
    # ===========================================
    app_name: str = Field(default="JD Jones RAG System")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # ===========================================
    # API Keys
    # ===========================================
    openai_api_key: str = Field(default="")
    anthropic_api_key: Optional[str] = Field(default=None)
    google_ai_api_key: Optional[str] = Field(default=None)
    
    # ===========================================
    # Database - PostgreSQL
    # ===========================================
    database_url: str = Field(
        default="postgresql+asyncpg://jdjones:password@localhost:5432/jd_jones_rag"
    )
    postgres_user: str = Field(default="jdjones")
    postgres_password: str = Field(default="password")
    postgres_db: str = Field(default="jd_jones_rag")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    
    # ===========================================
    # Redis
    # ===========================================
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    
    # ===========================================
    # ChromaDB
    # ===========================================
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8001)
    chroma_persist_directory: str = Field(default="./data/chroma")
    
    # ===========================================
    # JWT Authentication
    # ===========================================
    jwt_secret_key: str = Field(default="change-this-secret-key-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=480)
    
    # ===========================================
    # LLM Settings
    # ===========================================
    # Provider: "openai", "ollama", "anthropic"
    llm_provider: str = Field(default="ollama")
    llm_model: str = Field(default="llama3.2")  # For Ollama: llama3.2, mistral, phi3
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=2000)
    
    # Ollama Settings (for free local LLM)
    ollama_base_url: str = Field(default="http://localhost:11434")
    
    # Embedding Settings (local by default)
    # Provider: "openai", "local"
    embedding_provider: str = Field(default="local")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")  # Local: all-MiniLM-L6-v2
    embedding_dimensions: int = Field(default=384)  # 384 for MiniLM, 1536 for OpenAI
    
    # Reranker Settings (local by default)
    # Provider: "cohere", "cross_encoder"
    reranker_provider: str = Field(default="cross_encoder")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # ===========================================
    # RAG Settings
    # ===========================================
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_retrieval_results: int = Field(default=20)
    min_relevance_score: float = Field(default=0.7)
    
    # ===========================================
    # Super Memory Settings
    # ===========================================
    memory_max_per_user: int = Field(default=10000)
    memory_cache_ttl: int = Field(default=3600)
    memory_similarity_threshold: float = Field(default=0.92)
    auto_sync_enabled: bool = Field(default=True)
    sync_batch_size: int = Field(default=100)
    
    # ===========================================
    # Celery Settings
    # ===========================================
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")
    
    # ===========================================
    # CORS Settings
    # ===========================================
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001", "http://localhost:8000"]
    )
    
    # ===========================================
    # File Upload Settings
    # ===========================================
    upload_dir: str = Field(default="./uploads")
    max_upload_size_mb: int = Field(default=50)
    allowed_extensions: str = Field(
        default=".pdf,.docx,.doc,.txt,.md,.xlsx,.xls,.pptx,.csv"
    )
    
    # ===========================================
    # SMTP Email Settings
    # ===========================================
    smtp_enabled: bool = Field(default=False)
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_from_email: str = Field(default="noreply@jdjones.com")
    smtp_from_name: str = Field(default="JD Jones Customer Support")
    smtp_use_tls: bool = Field(default=True)
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Get allowed extensions as a list."""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]
    
    @property
    def upload_path(self) -> Path:
        """Get upload directory as Path object."""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def sync_database_url(self) -> str:
        """Get synchronous database URL for psycopg."""
        return self.database_url.replace("+asyncpg", "")
    
    @property
    def async_database_url(self) -> str:
        """Get async database URL for asyncpg."""
        if "+asyncpg" not in self.database_url:
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        return self.database_url


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


def get_llm(temperature: float = 0.7):
    """
    Create an LLM instance based on current settings.
    
    This factory function should be used throughout the codebase
    instead of directly instantiating ChatOpenAI with hardcoded models.
    
    Args:
        temperature: LLM temperature setting
        
    Returns:
        Configured ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI
    
    provider = settings.llm_provider.lower()
    
    if provider == 'ollama':
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temperature,
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama"
        )
    else:
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temperature,
            openai_api_key=settings.openai_api_key
        )
