"""Auth service: orchestrates login and current-user resolution.

This layer sits between the HTTP boundary (routes) and the domain/core
utilities. It knows nothing about FastAPI — no Request, no Response, no
HTTPException. That keeps it independently testable.
"""

from dataclasses import dataclass

from clinical_platform.core.jwt import TokenError, create_access_token, decode_access_token
from clinical_platform.core.security import verify_password
from clinical_platform.domain.models import User
from clinical_platform.infrastructure.repositories.user_repository import UserRepository


class AuthenticationError(Exception):
    """Raised when credentials are invalid."""


class InvalidTokenError(Exception):
    """Raised when a bearer token cannot be validated."""


@dataclass
class TokenResponse:
    access_token: str
    token_type: str = "bearer"


class AuthService:
    """Stateless auth service — safe to instantiate once and reuse."""

    def __init__(self, user_repo: UserRepository) -> None:
        self._repo = user_repo

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    def login(self, email: str, password: str) -> TokenResponse:
        """Validate *email*/*password* and return a signed JWT.

        Raises:
            AuthenticationError: If the email is unknown or the password
                does not match.
        """
        user = self._repo.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            # Deliberate vague message — do not reveal whether the email exists.
            raise AuthenticationError("Invalid email or password")

        token = create_access_token(subject=user.email, role=user.role.value)
        return TokenResponse(access_token=token)

    # ------------------------------------------------------------------
    # Resolve current user from a raw bearer token string
    # ------------------------------------------------------------------

    def get_current_user(self, token: str) -> User:
        """Decode *token* and return the corresponding User.

        Raises:
            InvalidTokenError: If the token is expired, malformed, or
                references an unknown user.
        """
        try:
            payload = decode_access_token(token)
        except TokenError as exc:
            raise InvalidTokenError(str(exc)) from exc

        email = payload.get("sub")
        if not isinstance(email, str):
            raise InvalidTokenError("Token is missing subject claim")

        user = self._repo.get_by_email(email)
        if user is None:
            raise InvalidTokenError("User not found")

        return user
