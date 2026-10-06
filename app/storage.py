import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Protocol

from .schemas import Composition, DraftRecord, DraftSummary, ProjectRecord, UserPublic, WorkspaceRecord

SESSION_TTL_DAYS = 30


def _session_cutoff() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=SESSION_TTL_DAYS)).isoformat()


class DraftStoreProtocol(Protocol):
    def create(self, user_id: str, composition: Composition) -> str:
        ...

    def update(self, user_id: str, draft_id: str, composition: Composition) -> None:
        ...

    def get(self, user_id: str, draft_id: str) -> Optional[DraftRecord]:
        ...

    def list(self, user_id: str) -> List[DraftSummary]:
        ...

    def create_user(self, name: str, email: str, password_hash: str) -> UserPublic:
        ...

    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        ...

    def create_session(self, user_id: str, token: str) -> None:
        ...

    def get_user_by_token(self, token: str) -> Optional[UserPublic]:
        ...

    def delete_session(self, token: str) -> None:
        ...

    def delete_user_sessions(self, user_id: str) -> None:
        ...

    def create_workspace(self, user_id: str, name: str) -> WorkspaceRecord:
        ...

    def list_workspaces(self, user_id: str) -> List[WorkspaceRecord]:
        ...

    def create_project(self, user_id: str, workspace_id: str, title: str) -> ProjectRecord:
        ...

    def list_projects(self, user_id: str, workspace_id: str) -> List[ProjectRecord]:
        ...

    def update_project(self, user_id: str, project_id: str, title: Optional[str] = None, draft_id: Optional[str] = None) -> ProjectRecord:
        ...

    def save_otp(self, email: str, code: str, expires_at: str) -> None:
        ...

    def get_valid_otp(self, email: str) -> Optional[dict[str, Any]]:
        ...

    def delete_otp(self, email: str) -> None:
        ...

    def update_user_password(self, email: str, password_hash: str) -> None:
        ...



class DraftStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _init(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS drafts (
                    draft_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    style TEXT NOT NULL,
                    mood TEXT NOT NULL,
                    composition_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """
            )
            # Check if user_id column exists (SQLite migration)
            cursor = connection.execute("PRAGMA table_info(drafts)")
            columns = [info[1] for info in cursor.fetchall()]
            if "user_id" not in columns:
                connection.execute("ALTER TABLE drafts ADD COLUMN user_id TEXT")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    draft_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (draft_id) REFERENCES drafts(draft_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS password_reset_otps (
                    email TEXT PRIMARY KEY,
                    otp_code TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def create(self, user_id: str, composition: Composition) -> str:
        draft_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        payload = composition.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO drafts (draft_id, user_id, title, style, mood, composition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (draft_id, user_id, composition.title, composition.style, composition.mood, payload, now, now),
            )
            connection.commit()
        return draft_id

    def update(self, user_id: str, draft_id: str, composition: Composition) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE drafts
                SET title = ?, style = ?, mood = ?, composition_json = ?, updated_at = ?
                WHERE draft_id = ? AND user_id = ?
                """,
                (
                    composition.title,
                    composition.style,
                    composition.mood,
                    composition.model_dump_json(),
                    now,
                    draft_id,
                    user_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(draft_id)
            connection.commit()

    def get(self, user_id: str, draft_id: str) -> Optional[DraftRecord]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT draft_id, composition_json, created_at, updated_at
                FROM drafts
                WHERE draft_id = ? AND user_id = ?
                """,
                (draft_id, user_id),
            ).fetchone()
        if not row:
            return None
        return DraftRecord(
            draft_id=row[0],
            composition=Composition.model_validate(json.loads(row[1])),
            created_at=row[2],
            updated_at=row[3],
        )

    def list(self, user_id: str) -> List[DraftSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT draft_id, title, style, mood, updated_at
                FROM drafts
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT 50
                """,
                (user_id,),
            ).fetchall()
        return [
            DraftSummary(
                draft_id=row[0],
                title=row[1],
                style=row[2],
                mood=row[3],
                updated_at=row[4],
            )
            for row in rows
        ]

    def create_user(self, name: str, email: str, password_hash: str) -> UserPublic:
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (user_id, name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, name, email.lower(), password_hash, now),
            )
            connection.commit()
        return UserPublic(user_id=user_id, name=name, email=email.lower(), created_at=now)

    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, name, email, password_hash, created_at
                FROM users
                WHERE email = ?
                """,
                (email.lower(),),
            ).fetchone()
        if not row:
            return None
        return {"user_id": row[0], "name": row[1], "email": row[2], "password_hash": row[3], "created_at": row[4]}

    def create_session(self, user_id: str, token: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (token, user_id, created_at)
                VALUES (?, ?, ?)
                """,
                (token, user_id, now),
            )
            connection.commit()

    def get_user_by_token(self, token: str) -> Optional[UserPublic]:
        cutoff = _session_cutoff()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT users.user_id, users.name, users.email, users.created_at
                FROM sessions
                JOIN users ON users.user_id = sessions.user_id
                WHERE sessions.token = ? AND sessions.created_at >= ?
                """,
                (token, cutoff),
            ).fetchone()
        if not row:
            return None
        return UserPublic(user_id=row[0], name=row[1], email=row[2], created_at=row[3])

    def delete_session(self, token: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
            connection.commit()

    def delete_user_sessions(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            connection.commit()

    def create_workspace(self, user_id: str, name: str) -> WorkspaceRecord:
        workspace_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workspaces (workspace_id, user_id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (workspace_id, user_id, name, now, now),
            )
            connection.commit()
        return WorkspaceRecord(workspace_id=workspace_id, user_id=user_id, name=name, created_at=now, updated_at=now)

    def list_workspaces(self, user_id: str) -> List[WorkspaceRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT workspace_id, user_id, name, created_at, updated_at
                FROM workspaces
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            WorkspaceRecord(workspace_id=row[0], user_id=row[1], name=row[2], created_at=row[3], updated_at=row[4])
            for row in rows
        ]

    def create_project(self, user_id: str, workspace_id: str, title: str) -> ProjectRecord:
        project_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            workspace = connection.execute(
                "SELECT workspace_id FROM workspaces WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            ).fetchone()
            if not workspace:
                raise KeyError(workspace_id)
            connection.execute(
                """
                INSERT INTO projects (project_id, workspace_id, user_id, title, draft_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, NULL, ?, ?)
                """,
                (project_id, workspace_id, user_id, title, now, now),
            )
            connection.commit()
        return ProjectRecord(
            project_id=project_id,
            workspace_id=workspace_id,
            user_id=user_id,
            title=title,
            draft_id=None,
            created_at=now,
            updated_at=now,
        )

    def list_projects(self, user_id: str, workspace_id: str) -> List[ProjectRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT project_id, workspace_id, user_id, title, draft_id, created_at, updated_at
                FROM projects
                WHERE user_id = ? AND workspace_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id, workspace_id),
            ).fetchall()
        return [
            ProjectRecord(
                project_id=row[0],
                workspace_id=row[1],
                user_id=row[2],
                title=row[3],
                draft_id=row[4],
                created_at=row[5],
                updated_at=row[6],
            )
            for row in rows
        ]

    def update_project(self, user_id: str, project_id: str, title: Optional[str] = None, draft_id: Optional[str] = None) -> ProjectRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT project_id, workspace_id, user_id, title, draft_id, created_at, updated_at FROM projects WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            ).fetchone()
            if not row:
                raise KeyError(project_id)
            next_title = title if title is not None else row[3]
            next_draft_id = draft_id if draft_id is not None else row[4]
            if next_draft_id:
                owned = connection.execute(
                    "SELECT 1 FROM drafts WHERE draft_id = ? AND user_id = ?",
                    (next_draft_id, user_id),
                ).fetchone()
                if not owned:
                    raise KeyError(next_draft_id)
            connection.execute(
                """
                UPDATE projects
                SET title = ?, draft_id = ?, updated_at = ?
                WHERE project_id = ? AND user_id = ?
                """,
                (next_title, next_draft_id, now, project_id, user_id),
            )
            connection.commit()
        return ProjectRecord(
            project_id=row[0],
            workspace_id=row[1],
            user_id=row[2],
            title=next_title,
            draft_id=next_draft_id,
            created_at=row[5],
            updated_at=now,
        )

    def save_otp(self, email: str, code: str, expires_at: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO password_reset_otps (email, otp_code, expires_at)
                VALUES (?, ?, ?)
                """,
                (email.lower(), code, expires_at),
            )
            connection.commit()

    def get_valid_otp(self, email: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT email, otp_code, expires_at
                FROM password_reset_otps
                WHERE email = ?
                """,
                (email.lower(),),
            ).fetchone()
        if not row:
            return None
        return {"email": row[0], "otp_code": row[1], "expires_at": row[2]}

    def delete_otp(self, email: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM password_reset_otps WHERE email = ?", (email.lower(),))
            connection.commit()

    def update_user_password(self, email: str, password_hash: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (password_hash, email.lower()),
            )
            connection.commit()



class PostgresDraftStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._init()

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Postgres storage requires psycopg. Run: pip install -r backend/requirements.txt") from exc
        return psycopg.connect(self.database_url)

    def _init(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS drafts (
                        draft_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        title TEXT NOT NULL,
                        style TEXT NOT NULL,
                        mood TEXT NOT NULL,
                        composition_json JSONB NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name='drafts' AND column_name='user_id'
                """)
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE drafts ADD COLUMN user_id TEXT REFERENCES users(user_id)")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        token TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(user_id),
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workspaces (
                        workspace_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(user_id),
                        name TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                        project_id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
                        user_id TEXT NOT NULL REFERENCES users(user_id),
                        title TEXT NOT NULL,
                        draft_id TEXT REFERENCES drafts(draft_id),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS password_reset_otps (
                        email TEXT PRIMARY KEY,
                        otp_code TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    )
                    """
                )

    def create(self, user_id: str, composition: Composition) -> str:
        from psycopg.types.json import Jsonb

        draft_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        payload = composition.model_dump()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO drafts (draft_id, user_id, title, style, mood, composition_json, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (draft_id, user_id, composition.title, composition.style, composition.mood, Jsonb(payload), now, now),
                )
        return draft_id

    def update(self, user_id: str, draft_id: str, composition: Composition) -> None:
        from psycopg.types.json import Jsonb

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE drafts
                    SET title = %s, style = %s, mood = %s, composition_json = %s, updated_at = %s
                    WHERE draft_id = %s AND user_id = %s
                    """,
                    (
                        composition.title,
                        composition.style,
                        composition.mood,
                        Jsonb(composition.model_dump()),
                        now,
                        draft_id,
                        user_id,
                    ),
                )
                if cursor.rowcount == 0:
                    raise KeyError(draft_id)

    def get(self, user_id: str, draft_id: str) -> Optional[DraftRecord]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT draft_id, composition_json, created_at, updated_at
                    FROM drafts
                    WHERE draft_id = %s AND user_id = %s
                    """,
                    (draft_id, user_id),
                )
                row = cursor.fetchone()
        if not row:
            return None
        composition_payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
        return DraftRecord(
            draft_id=row[0],
            composition=Composition.model_validate(composition_payload),
            created_at=row[2],
            updated_at=row[3],
        )

    def list(self, user_id: str) -> List[DraftSummary]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT draft_id, title, style, mood, updated_at
                    FROM drafts
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 50
                    """,
                    (user_id,),
                )
                rows = cursor.fetchall()
        return [
            DraftSummary(
                draft_id=row[0],
                title=row[1],
                style=row[2],
                mood=row[3],
                updated_at=row[4],
            )
            for row in rows
        ]

    def create_user(self, name: str, email: str, password_hash: str) -> UserPublic:
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (user_id, name, email, password_hash, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, name, email.lower(), password_hash, now),
                )
        return UserPublic(user_id=user_id, name=name, email=email.lower(), created_at=now)

    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, name, email, password_hash, created_at
                    FROM users
                    WHERE email = %s
                    """,
                    (email.lower(),),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return {"user_id": row[0], "name": row[1], "email": row[2], "password_hash": row[3], "created_at": row[4]}

    def create_session(self, user_id: str, token: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (token, user_id, created_at)
                    VALUES (%s, %s, %s)
                    """,
                    (token, user_id, now),
                )

    def get_user_by_token(self, token: str) -> Optional[UserPublic]:
        cutoff = _session_cutoff()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT users.user_id, users.name, users.email, users.created_at
                    FROM sessions
                    JOIN users ON users.user_id = sessions.user_id
                    WHERE sessions.token = %s AND sessions.created_at >= %s
                    """,
                    (token, cutoff),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return UserPublic(user_id=row[0], name=row[1], email=row[2], created_at=row[3])

    def delete_session(self, token: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM sessions WHERE token = %s", (token,))

    def delete_user_sessions(self, user_id: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))

    def create_workspace(self, user_id: str, name: str) -> WorkspaceRecord:
        workspace_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO workspaces (workspace_id, user_id, name, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (workspace_id, user_id, name, now, now),
                )
        return WorkspaceRecord(workspace_id=workspace_id, user_id=user_id, name=name, created_at=now, updated_at=now)

    def list_workspaces(self, user_id: str) -> List[WorkspaceRecord]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT workspace_id, user_id, name, created_at, updated_at
                    FROM workspaces
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
                rows = cursor.fetchall()
        return [
            WorkspaceRecord(workspace_id=row[0], user_id=row[1], name=row[2], created_at=row[3], updated_at=row[4])
            for row in rows
        ]

    def create_project(self, user_id: str, workspace_id: str, title: str) -> ProjectRecord:
        project_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT workspace_id FROM workspaces WHERE workspace_id = %s AND user_id = %s",
                    (workspace_id, user_id),
                )
                if not cursor.fetchone():
                    raise KeyError(workspace_id)
                cursor.execute(
                    """
                    INSERT INTO projects (project_id, workspace_id, user_id, title, draft_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NULL, %s, %s)
                    """,
                    (project_id, workspace_id, user_id, title, now, now),
                )
        return ProjectRecord(
            project_id=project_id,
            workspace_id=workspace_id,
            user_id=user_id,
            title=title,
            draft_id=None,
            created_at=now,
            updated_at=now,
        )

    def list_projects(self, user_id: str, workspace_id: str) -> List[ProjectRecord]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT project_id, workspace_id, user_id, title, draft_id, created_at, updated_at
                    FROM projects
                    WHERE user_id = %s AND workspace_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (user_id, workspace_id),
                )
                rows = cursor.fetchall()
        return [
            ProjectRecord(
                project_id=row[0],
                workspace_id=row[1],
                user_id=row[2],
                title=row[3],
                draft_id=row[4],
                created_at=row[5],
                updated_at=row[6],
            )
            for row in rows
        ]

    def update_project(self, user_id: str, project_id: str, title: Optional[str] = None, draft_id: Optional[str] = None) -> ProjectRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT project_id, workspace_id, user_id, title, draft_id, created_at, updated_at
                    FROM projects
                    WHERE project_id = %s AND user_id = %s
                    """,
                    (project_id, user_id),
                )
                row = cursor.fetchone()
                if not row:
                    raise KeyError(project_id)
                next_title = title if title is not None else row[3]
                next_draft_id = draft_id if draft_id is not None else row[4]
                if next_draft_id:
                    cursor.execute(
                        "SELECT 1 FROM drafts WHERE draft_id = %s AND user_id = %s",
                        (next_draft_id, user_id),
                    )
                    if not cursor.fetchone():
                        raise KeyError(next_draft_id)
                cursor.execute(
                    """
                    UPDATE projects
                    SET title = %s, draft_id = %s, updated_at = %s
                    WHERE project_id = %s AND user_id = %s
                    """,
                    (next_title, next_draft_id, now, project_id, user_id),
                )
        return ProjectRecord(
            project_id=row[0],
            workspace_id=row[1],
            user_id=row[2],
            title=next_title,
            draft_id=next_draft_id,
            created_at=row[5],
            updated_at=now,
        )

    def save_otp(self, email: str, code: str, expires_at: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO password_reset_otps (email, otp_code, expires_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (email) DO UPDATE
                    SET otp_code = EXCLUDED.otp_code, expires_at = EXCLUDED.expires_at
                    """,
                    (email.lower(), code, expires_at),
                )

    def get_valid_otp(self, email: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT email, otp_code, expires_at
                    FROM password_reset_otps
                    WHERE email = %s
                    """,
                    (email.lower(),),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return {"email": row[0], "otp_code": row[1], "expires_at": row[2]}

    def delete_otp(self, email: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM password_reset_otps WHERE email = %s", (email.lower(),))

    def update_user_password(self, email: str, password_hash: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE users SET password_hash = %s WHERE email = %s",
                    (password_hash, email.lower()),
                )



def create_draft_store(database_path: str, database_url: str = "") -> DraftStoreProtocol:
    if database_url.strip():
        return PostgresDraftStore(database_url.strip())
    return DraftStore(database_path)
