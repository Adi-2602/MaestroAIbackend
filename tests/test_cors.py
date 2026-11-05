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
