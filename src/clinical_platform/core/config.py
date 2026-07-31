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


@lru_cache
def get_settings() -> Settings:
    return Settings()