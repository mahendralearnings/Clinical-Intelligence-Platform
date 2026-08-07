"""Auth routes: POST /auth/login and GET /auth/me."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from clinical_platform.api.middleware.auth_dependencies import (
    CurrentUser,
    get_auth_service,
)
from clinical_platform.api.schemas.auth import (
    CurrentUserResponse,
    LoginRequest,
    TokenResponse,
)
from clinical_platform.services.auth_service import AuthenticationError, AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Obtain a JWT access token",
)
def login(
    body: LoginRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    ...
    try:
        result = auth_service.login(email=body.email, password=body.password)
        
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return TokenResponse(access_token=result.access_token, token_type=result.token_type)


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="Return the currently authenticated user",
)
def get_me(current_user: CurrentUser) -> CurrentUserResponse:
    """Decode the bearer token and return the caller's profile."""
    return CurrentUserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role.value,
    )
