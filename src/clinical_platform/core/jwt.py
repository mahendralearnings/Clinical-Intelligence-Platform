"""JWT creation and verification utilities."""

from datetime import UTC, datetime, timedelta

import jwt

# ---------------------------------------------------------------------------
# Configuration – in production these would come from environment variables.
# ---------------------------------------------------------------------------
from clinical_platform.core.config import get_settings

_settings = get_settings()
SECRET_KEY: str = _settings.jwt_secret_key
ALGORITHM: str = _settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES: int = _settings.jwt_access_token_expire_minutes

# SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_use_a_long_random_secret"
# ALGORITHM: str = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES: int = 30


class TokenError(Exception):
    """Raised when a JWT cannot be decoded or is invalid."""


def create_access_token(subject: str, role: str) -> str:
    """Create a signed JWT that expires in ACCESS_TOKEN_EXPIRE_MINUTES minutes.

    Args:
        subject: The user identifier (email) to embed as the ``sub`` claim.
        role: The user's role string to embed as a custom ``role`` claim.

    Returns:
        A compact, URL-safe JWT string.
    """
    now = datetime.now(tz=UTC)
    payload: dict[str, object] = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, object]:
    """Decode and validate *token*, returning its payload.

    Raises:
        TokenError: If the token is expired, tampered-with, or otherwise invalid.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError(f"Invalid token: {exc}") from exc
