from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app

FRONTEND = "https://maestro-ai-kappa.vercel.app"


def test_deployed_frontend_allowed_by_default():
    assert FRONTEND in Settings(CORS_ORIGIN="http://localhost:5173").origins


def test_trailing_slash_is_stripped():
    origins = Settings(CORS_ORIGIN="https://example.com/").origins
    assert "https://example.com" in origins


def test_wildcard_kept():
    assert Settings(CORS_ORIGIN="*").origins == ["*"]


def test_register_preflight_from_frontend():
    client = TestClient(app)
    response = client.options(
        "/api/v1/auth/register",
        headers={
            "Origin": FRONTEND,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == FRONTEND


def test_server_error_keeps_cors_headers():
    from app.main import get_store

    def broken_store():
        raise RuntimeError("database unreachable")

    app.dependency_overrides[get_store] = broken_store
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/auth/register",
            json={"name": "Test User", "email": "t@example.com", "password": "secret123"},
            headers={"Origin": FRONTEND},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == FRONTEND
    assert "RuntimeError" in response.json()["detail"]
