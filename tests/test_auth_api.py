from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app import main
from app.main import app, get_store
from app.storage import DraftStore


def make_client(tmp_path) -> tuple[TestClient, DraftStore]:
    store = DraftStore(str(tmp_path / "auth.db"))
    app.dependency_overrides[get_store] = lambda: store
    main._otp_failed_attempts.clear()
    return TestClient(app), store


def register(client: TestClient, email: str = "user@example.com", password: str = "secret123") -> str:
    response = client.post(
        "/api/v1/auth/register",
        json={"name": "Test User", "email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    return response.json()["token"]


def test_forgot_password_does_not_reveal_unknown_email(tmp_path) -> None:
    client, store = make_client(tmp_path)
    try:
        register(client)
        known = client.post("/api/v1/auth/forgot-password", json={"email": "user@example.com"})
        unknown = client.post("/api/v1/auth/forgot-password", json={"email": "nobody@example.com"})
        assert known.status_code == unknown.status_code == 200
        assert known.json() == unknown.json()
        assert store.get_valid_otp("nobody@example.com") is None
    finally:
        app.dependency_overrides.clear()


def test_reset_password_rotates_password_and_revokes_sessions(tmp_path) -> None:
    client, store = make_client(tmp_path)
    try:
        old_token = register(client)
        client.post("/api/v1/auth/forgot-password", json={"email": "user@example.com"})
        otp = store.get_valid_otp("user@example.com")["otp_code"]

        response = client.post(
            "/api/v1/auth/reset-password",
            json={"email": "user@example.com", "otp": otp, "new_password": "newsecret456"},
        )
        assert response.status_code == 200, response.text

        me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {old_token}"})
        assert me.status_code == 401

        old_login = client.post("/api/v1/auth/login", json={"email": "user@example.com", "password": "secret123"})
        assert old_login.status_code == 401
        new_login = client.post("/api/v1/auth/login", json={"email": "user@example.com", "password": "newsecret456"})
        assert new_login.status_code == 200

        reused = client.post(
            "/api/v1/auth/reset-password",
            json={"email": "user@example.com", "otp": otp, "new_password": "another789"},
        )
        assert reused.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_expired_otp_reports_expiry(tmp_path) -> None:
    client, store = make_client(tmp_path)
    try:
        register(client)
        expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        store.save_otp("user@example.com", "12345678", expired)
        response = client.post(
            "/api/v1/auth/reset-password",
            json={"email": "user@example.com", "otp": "12345678", "new_password": "newsecret456"},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Verification code has expired."
    finally:
        app.dependency_overrides.clear()


def test_otp_locks_after_too_many_wrong_attempts(tmp_path) -> None:
    client, store = make_client(tmp_path)
    try:
        register(client)
        client.post("/api/v1/auth/forgot-password", json={"email": "user@example.com"})
        real_otp = store.get_valid_otp("user@example.com")["otp_code"]
        wrong = "00000000" if real_otp != "00000000" else "11111111"

        statuses = []
        for _ in range(main.MAX_OTP_ATTEMPTS):
            response = client.post(
                "/api/v1/auth/reset-password",
                json={"email": "user@example.com", "otp": wrong, "new_password": "newsecret456"},
            )
            statuses.append(response.status_code)
        assert statuses[-1] == 429
        assert all(code == 400 for code in statuses[:-1])

        after_lock = client.post(
            "/api/v1/auth/reset-password",
            json={"email": "user@example.com", "otp": real_otp, "new_password": "newsecret456"},
        )
        assert after_lock.status_code == 400
    finally:
        app.dependency_overrides.clear()
