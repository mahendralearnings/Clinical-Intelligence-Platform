import json
from pathlib import Path

from fastapi.testclient import TestClient

from clinical_platform.core.config import Settings, get_settings
from clinical_platform.main import app

client = TestClient(app)


def _login() -> str:
    response = client.post(
        "/auth/login",
        json={"email": "bob@clinic.dev", "password": "doctor_pass_2!"},
    )
    return response.json()["access_token"]


def test_summary_with_no_log_file_returns_zeros(tmp_path: Path) -> None:
    fake_log = tmp_path / "does_not_exist.jsonl"

    def fake_settings() -> Settings:
        settings = get_settings()
        return settings.model_copy(update={"query_log_path": str(fake_log)})

    app.dependency_overrides[get_settings] = fake_settings
    try:
        token = _login()
        response = client.get(
            "/observability/summary", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["total_queries"] == 0
    assert body["most_common_source_document"] is None


def test_summary_without_auth_returns_401() -> None:
    response = client.get("/observability/summary")
    assert response.status_code == 401