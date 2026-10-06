import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, get_store, get_current_user
from app.schemas import Composition, UserPublic
from app.storage import DraftStore


def load_golden_payload() -> dict:
    return json.loads(Path(__file__).with_name("golden_composition.json").read_text())


def test_create_draft_from_manual_composition(tmp_path) -> None:
    store = DraftStore(str(tmp_path / "drafts.db"))
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_current_user] = lambda: UserPublic(
        user_id="test_user_123",
        name="Test User",
        email="test@example.com",
        created_at="2026-06-10T00:00:00Z"
    )
    try:
        client = TestClient(app)
        payload = load_golden_payload()
        payload["title"] = "Manual Editable Draft"

        response = client.post(
            "/api/v1/drafts",
            json=payload,
            headers={"Authorization": "Bearer dummy-token"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["draft_id"]
        assert body["composition"]["title"] == "Manual Editable Draft"
        assert store.get("test_user_123", body["draft_id"]).composition == Composition.model_validate(payload)
    finally:
        app.dependency_overrides.clear()


def test_draft_isolation(tmp_path) -> None:
    store = DraftStore(str(tmp_path / "drafts.db"))
    app.dependency_overrides[get_store] = lambda: store
    try:
        client = TestClient(app)
        payload = load_golden_payload()

        # 1. Create a draft as user 1
        app.dependency_overrides[get_current_user] = lambda: UserPublic(
            user_id="user_1",
            name="User One",
            email="user1@example.com",
            created_at="2026-06-10T00:00:00Z"
        )
        response = client.post(
            "/api/v1/drafts",
            json=payload,
            headers={"Authorization": "Bearer token1"}
        )
        assert response.status_code == 200
        draft_id = response.json()["draft_id"]

        # 2. Try to get draft as user 2 (should return 404 Draft not found)
        app.dependency_overrides[get_current_user] = lambda: UserPublic(
            user_id="user_2",
            name="User Two",
            email="user2@example.com",
            created_at="2026-06-10T00:00:00Z"
        )
        response_get = client.get(
            f"/api/v1/drafts/{draft_id}",
            headers={"Authorization": "Bearer token2"}
        )
        assert response_get.status_code == 404

        # 3. Try to list drafts as user 2 (should return empty list)
        response_list = client.get(
            "/api/v1/drafts",
            headers={"Authorization": "Bearer token2"}
        )
        assert response_list.status_code == 200
        assert len(response_list.json()) == 0

        # 4. List drafts as user 1 (should return 1 draft)
        app.dependency_overrides[get_current_user] = lambda: UserPublic(
            user_id="user_1",
            name="User One",
            email="user1@example.com",
            created_at="2026-06-10T00:00:00Z"
        )
        response_list_1 = client.get(
            "/api/v1/drafts",
            headers={"Authorization": "Bearer token1"}
        )
        assert response_list_1.status_code == 200
        assert len(response_list_1.json()) == 1
        assert response_list_1.json()[0]["draft_id"] == draft_id
    finally:
        app.dependency_overrides.clear()


def test_project_rejects_foreign_draft_id(tmp_path) -> None:
    store = DraftStore(str(tmp_path / "drafts.db"))
    app.dependency_overrides[get_store] = lambda: store
    try:
        user_a = UserPublic(user_id="user_a", name="A", email="a@example.com", created_at="2026-06-10T00:00:00Z")
        user_b = UserPublic(user_id="user_b", name="B", email="b@example.com", created_at="2026-06-10T00:00:00Z")
        payload = load_golden_payload()

        app.dependency_overrides[get_current_user] = lambda: user_a
        client = TestClient(app)
        draft_id = client.post("/api/v1/drafts", json=payload, headers={"Authorization": "Bearer a"}).json()["draft_id"]
        workspace_id = client.post(
            "/api/v1/workspaces",
            json={"name": "Studio"},
            headers={"Authorization": "Bearer a"},
        ).json()["workspace_id"]
        project_id = client.post(
            f"/api/v1/workspaces/{workspace_id}/projects",
            json={"title": "Song"},
            headers={"Authorization": "Bearer a"},
        ).json()["project_id"]

        app.dependency_overrides[get_current_user] = lambda: user_b
        response = client.put(
            f"/api/v1/projects/{project_id}",
            json={"draft_id": draft_id},
            headers={"Authorization": "Bearer b"},
        )
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()
