#import pytest
from fastapi.testclient import TestClient

from clinical_platform.main import app

client = TestClient(app)

DOCTOR_EMAIL = "bob@clinic.dev"
DOCTOR_PASSWORD = "doctor_pass_2!"


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_login_with_correct_credentials_returns_token() -> None:
    response = client.post(
        "/auth/login", json={"email": DOCTOR_EMAIL, "password": DOCTOR_PASSWORD}
    )
    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_with_wrong_password_returns_401() -> None:
    response = client.post(
        "/auth/login", json={"email": DOCTOR_EMAIL, "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_login_with_unknown_email_returns_401() -> None:
    response = client.post(
        "/auth/login", json={"email": "nobody@clinic.dev", "password": "whatever"}
    )
    assert response.status_code == 401


def test_me_without_token_returns_401() -> None:
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_with_garbage_token_returns_401() -> None:
    response = client.get("/auth/me", headers={"Authorization": "Bearer garbage"})
    assert response.status_code == 401


def test_me_with_valid_token_returns_user_profile() -> None:
    login_response = client.post(
        "/auth/login", json={"email": DOCTOR_EMAIL, "password": DOCTOR_PASSWORD}
    )
    token = login_response.json()["access_token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == DOCTOR_EMAIL
    assert body["role"] == "doctor"