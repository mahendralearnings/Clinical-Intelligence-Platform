"""FastAPI dependencies for authentication and permission checking."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from clinical_platform.domain.models import Permission, User
from clinical_platform.infrastructure.repositories.user_repository import UserRepository
from clinical_platform.services.auth_service import (
    AuthService,
    InvalidTokenError,
)

# ---------------------------------------------------------------------------
# Shared singletons — constructed once per process.
# ---------------------------------------------------------------------------

_user_repo = UserRepository()
_auth_service = AuthService(_user_repo)
_bearer_scheme = HTTPBearer()

# ---------------------------------------------------------------------------
# Core dependency: resolve the current user from the Authorization header
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
) -> User:
    """Extract and validate the Bearer token; return the authenticated User.

    Raises HTTP 401 if the token is missing, expired, or invalid.
    """
    try:
        return _auth_service.get_current_user(credentials.credentials)
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ---------------------------------------------------------------------------
# Permission dependency factory
# ---------------------------------------------------------------------------


def require_permission(action: Permission) -> Callable[[User], User]:
    """Return a FastAPI dependency that enforces *action* on the current user.

    Usage::

        @router.get("/sensitive")
        def sensitive_endpoint(
            _: Annotated[User, Depends(require_permission(Permission.VIEW_AUDIT_LOGS))]
        ):
            ...

    Raises HTTP 403 if the authenticated user's role does not grant *action*.
    """
    
    def _check(user: Annotated[User, Depends(get_current_user)]) -> User:
        if not user.has_permission(action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: '{action}' is not allowed for role '{user.role}'",
            )
        return user

    # Give the inner function a unique name so FastAPI's dependency cache
    # treats different permission checks as distinct dependencies.
    _check.__name__ = f"require_{action}"
    return _check


# Convenience type alias for routes that just need an authenticated user.
CurrentUser = Annotated[User, Depends(get_current_user)]

def get_auth_service() -> AuthService:
    """Public accessor for the shared AuthService instance."""
    return _auth_service