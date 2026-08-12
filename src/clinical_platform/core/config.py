"""
Application settings, read from environment variables / .env.

Design decision: ONE typed Settings object, never hardcoded secrets in
code. The JWT secret in particular must never live in a committed file -
Kiro's version had it hardcoded directly in jwt.py, which means it was
sitting in plain text in Git history. This fixes that.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret_key: str = "CHANGE_ME_LOCAL_ONLY_NEVER_USE_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # ------------------------------------------------------------------
    # Phase 3 — Retrieval & Embeddings
    # ------------------------------------------------------------------

    # Directory scanned by IngestPipeline (relative paths resolved from cwd)
    docs_dir: str = "data/sample_documents"

    # Path to the local JSON vector store file
    vector_store_path: str = "data/vector_store.json"

    # AWS Bedrock settings
    bedrock_region: str = "us-east-1"
    embedding_model_id: str = "cohere.embed-english-v3"

    # Expected output dimension of the embedding model
    # Cohere Embed English v3 produces 1024-d vectors
    embedding_dimension: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()