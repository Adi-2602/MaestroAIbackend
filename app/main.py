import logging
import asyncio
import json
import hashlib
import hmac
import shutil
import subprocess
import secrets
import tempfile
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Response, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .commercial import commercial_review
from .evaluation import evaluate_composition
from .music import (
    composition_to_midi_bytes,
    composition_to_musicxml_bytes,
    composition_to_notation_text,
    composition_to_soundfont_wav_bytes,
    composition_to_wav_bytes,
    composition_validation_report,
    optimize_generated_composition,
    validate_composition,
)
from .nim_client import NimClient, NimTimeoutError
from .schemas import (
    ComposeRequest,
    ComposeResponse,
    Composition,
    CommercialReview,
    DraftRecord,
    DraftSummary,
    EvaluationReport,
    ProviderInfo,
    RefineRequest,
    ValidationReport,
    RegisterRequest,
    LoginRequest,
    UserPublic,
    AuthSession,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    WorkspaceCreate,
    WorkspaceRecord,
    ProjectCreate,
    ProjectUpdate,
    ProjectRecord,
)
from .storage import DraftStoreProtocol as DraftStore, create_draft_store

from datetime import datetime, timezone, timedelta
import smtplib
import httpx
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger("music-composer")
logging.basicConfig(level=logging.INFO)

MAX_OTP_ATTEMPTS = 5
_otp_failed_attempts: dict[str, int] = {}


def _log_dev_otp(email: str, otp: str, settings: Settings, reason: str) -> None:
    """Print the OTP to the console only in development so it never leaks into production logs."""
    if settings.node_env.lower() == "development":
        logger.warning("\n" + "=" * 60 + f"\n[DEV ONLY] {reason}\nEmail: {email}\nOTP Code: {otp}\n" + "=" * 60)
    else:
        logger.error("Password reset email could not be sent to %s (%s).", email, reason)


@lru_cache
def get_store() -> DraftStore:
    settings = get_settings()
    return create_draft_store(settings.app_database_path, settings.database_url)


def get_nim_client(settings: Settings = Depends(get_settings)) -> NimClient:
    return NimClient(settings)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 180_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt, digest = stored_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 180_000).hex()
    return hmac.compare_digest(candidate, digest)


def send_otp_email(email: str, otp: str, settings: Settings) -> bool:
    smtp_configured = bool(settings.smtp_host and settings.smtp_user and settings.smtp_password)
    if not settings.brevo_api_key and not smtp_configured:
        _log_dev_otp(email, otp, settings, "Email delivery not configured")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Verify Your Maestro AI Password Reset"
        msg["From"] = settings.smtp_sender
        msg["To"] = email

        # Plain text fallback
        text_content = f"""Hello,

You requested to reset your password for Maestro AI.
Your 8-digit verification code is: {otp}

This code is valid for 10 minutes. If you did not request this, please ignore this email.

Best regards,
Maestro AI Team
"""

        # HTML Premium Template
        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Maestro AI - Password Reset Verification</title>
</head>
<body style="margin: 0; padding: 0; background-color: #090b0f; font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #e7edf2; -webkit-font-smoothing: antialiased;">
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #090b0f; min-height: 100vh; padding: 40px 20px;">
    <tr>
      <td align="center" valign="top">
        <!-- Main Card Container -->
        <table border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 500px; background-color: #111820; border: 1px solid #2d3842; border-radius: 8px; box-shadow: 0 20px 54px rgba(0, 0, 0, 0.36); overflow: hidden;">
          <!-- Header -->
          <tr>
            <td align="center" style="padding: 30px 20px; background-color: #0b1015; border-bottom: 1px solid #2d3842;">
              <table border="0" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="background: linear-gradient(135deg, #13d6c8, #e357a8); border-radius: 6px; padding: 10px; display: inline-block;">
                    <!-- Stylized Logo Mark (Music Note Symbol) -->
                    <span style="font-size: 24px; color: #081112; font-weight: bold; line-height: 1; display: block; width: 24px; height: 24px;">♪</span>
                  </td>
                  <td style="padding-left: 12px; text-align: left;">
                    <h1 style="margin: 0; font-size: 20px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">Maestro AI</h1>
                    <p style="margin: 0; font-size: 11px; color: #8796a4; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">AI composition workstation</p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Content Body -->
          <tr>
            <td style="padding: 40px 30px; text-align: center;">
              <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 700; color: #ffffff;">Reset Your Password</h2>
              <p style="margin: 0 0 30px 0; font-size: 14px; line-height: 1.5; color: #bac6cf;">
                We received a request to reset your password. Use the verification code below to complete the reset process:
              </p>
              
              <!-- OTP Box Container -->
              <div style="background-color: #0b1015; border: 1px solid #34404a; border-radius: 6px; padding: 20px; display: inline-block; min-width: 260px;">
                <span style="font-size: 11px; color: #8796a4; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; display: block; margin-bottom: 10px;">Verification Code</span>
                <span style="font-size: 32px; font-weight: 800; font-family: 'Courier New', Courier, monospace; letter-spacing: 6px; color: #13d6c8; display: block; line-height: 1.2;">{otp}</span>
              </div>
              
              <p style="margin: 30px 0 0 0; font-size: 12px; line-height: 1.5; color: #ff6262; font-weight: 600;">
                This code is valid for exactly 10 minutes.
              </p>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td align="center" style="padding: 24px 30px; background-color: #0b1015; border-top: 1px solid #2d3842; font-size: 11px; color: #5f6d78; line-height: 1.4;">
              <p style="margin: 0 0 8px 0;">If you did not request a password reset, you can safely ignore this email.</p>
              <p style="margin: 0;">&copy; 2026 Maestro AI. All rights reserved.</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        if settings.brevo_api_key:
            response = httpx.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={"api-key": settings.brevo_api_key, "accept": "application/json"},
                json={
                    "sender": {"name": settings.app_name, "email": settings.smtp_sender},
                    "to": [{"email": email}],
                    "subject": msg["Subject"],
                    "textContent": text_content,
                    "htmlContent": html_content,
                },
                timeout=15,
            )
            response.raise_for_status()
            logger.info("Successfully sent OTP email to %s via Brevo", email)
            return True

        # Send
        if settings.smtp_use_tls:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10)
            server.starttls()
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10)

        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(settings.smtp_sender, email, msg.as_string())
        server.quit()
        logger.info("Successfully sent OTP email to %s", email)
        return True
    except Exception as e:
        logger.exception("Failed to send OTP email to %s: %s", email, e)
        _log_dev_otp(email, otp, settings, "Email delivery failed")
        return False


def create_auth_session(user: UserPublic, store: DraftStore) -> AuthSession:
    token = secrets.token_urlsafe(32)
    store.create_session(user.user_id, token)
    return AuthSession(token=token, user=user)


def get_current_user(
    authorization: str = Header(default=""),
    store: DraftStore = Depends(get_store),
) -> UserPublic:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Login required.")
    user = store.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return user


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.middleware("http")
async def unhandled_error_as_json(request: Request, call_next):
    # Registered before CORSMiddleware so it sits inside it: a crash then
    # still gets CORS headers, and the browser shows the real error instead
    # of a misleading "blocked by CORS policy" / "Failed to fetch".
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Server error ({type(exc).__name__}). Check the backend logs."},
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=settings.origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
api_router = APIRouter()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@api_router.get("/provider", response_model=ProviderInfo)
def provider(settings: Settings = Depends(get_settings)) -> ProviderInfo:
    soundfont_ready = bool(settings.resolved_fluidsynth_path and settings.resolved_soundfont_path)
    return ProviderInfo(
        base_url=settings.normalized_nim_base_url,
        model=settings.nim_model,
        api_key_configured=bool(settings.nim_api_key),
        audio_engine="soundfont_fluidsynth" if soundfont_ready else "procedural",
    )


@api_router.post("/auth/register", response_model=AuthSession)
def register(request: RegisterRequest, store: DraftStore = Depends(get_store)) -> AuthSession:
    if store.get_user_by_email(request.email):
        raise HTTPException(status_code=409, detail="An account already exists for this email.")
    user = store.create_user(request.name.strip(), request.email.strip().lower(), hash_password(request.password))
    return create_auth_session(user, store)


@api_router.post("/auth/login", response_model=AuthSession)
def login(request: LoginRequest, store: DraftStore = Depends(get_store)) -> AuthSession:
    record = store.get_user_by_email(request.email.strip().lower())
    if not record or not verify_password(request.password, record["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user = UserPublic(
        user_id=record["user_id"],
        name=record["name"],
        email=record["email"],
        created_at=record["created_at"],
    )
    return create_auth_session(user, store)


@api_router.get("/auth/me", response_model=UserPublic)
def me(user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return user


@api_router.post("/auth/logout")
def logout(
    authorization: str = Header(default=""),
    store: DraftStore = Depends(get_store),
) -> dict:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        store.delete_session(token)
    return {"status": "ok"}


@api_router.post("/auth/forgot-password")
def forgot_password(
    request: ForgotPasswordRequest,
    store: DraftStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> dict:
    email = request.email.strip().lower()
    generic_response = {
        "status": "ok",
        "message": "If an account exists for this email, a verification code has been sent.",
    }
    user = store.get_user_by_email(email)
    if not user:
        # Do not reveal whether an account is registered.
        return generic_response

    # Generate random 8 digit OTP
    otp = "".join(secrets.choice("0123456789") for _ in range(8))

    # Expiration time is 10 minutes from now
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    # Store OTP in DB
    store.save_otp(email, otp, expires_at)
    _otp_failed_attempts.pop(email, None)

    # Send email
    send_otp_email(email, otp, settings)

    return generic_response


@api_router.post("/auth/reset-password")
def reset_password(
    request: ResetPasswordRequest,
    store: DraftStore = Depends(get_store),
) -> dict:
    email = request.email.strip().lower()
    otp_record = store.get_valid_otp(email)
    if not otp_record:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code.")

    # Check expiration
    try:
        expiry_dt = datetime.fromisoformat(otp_record["expires_at"])
    except (TypeError, ValueError):
        store.delete_otp(email)
        raise HTTPException(status_code=400, detail="Invalid verification code state.")
    if datetime.now(timezone.utc) > expiry_dt:
        store.delete_otp(email)
        raise HTTPException(status_code=400, detail="Verification code has expired.")

    # Check code match (constant-time), locking the code after too many wrong attempts
    if not hmac.compare_digest(otp_record["otp_code"].encode("utf-8"), request.otp.strip().encode("utf-8")):
        attempts = _otp_failed_attempts.get(email, 0) + 1
        if attempts >= MAX_OTP_ATTEMPTS:
            _otp_failed_attempts.pop(email, None)
            store.delete_otp(email)
            raise HTTPException(status_code=429, detail="Too many incorrect attempts. Please request a new code.")
        _otp_failed_attempts[email] = attempts
        raise HTTPException(status_code=400, detail="Invalid or expired verification code.")

    # Code is valid! Update password
    new_password_hash = hash_password(request.new_password)
    store.update_user_password(email, new_password_hash)

    # Delete the OTP code from DB so it cannot be reused
    store.delete_otp(email)
    _otp_failed_attempts.pop(email, None)

    # Sign out every existing session for this account
    user_record = store.get_user_by_email(email)
    if user_record:
        store.delete_user_sessions(user_record["user_id"])

    return {"status": "ok", "message": "Password reset successfully."}



@api_router.get("/workspaces", response_model=list[WorkspaceRecord])
def list_user_workspaces(
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> list[WorkspaceRecord]:
    return store.list_workspaces(user.user_id)


@api_router.post("/workspaces", response_model=WorkspaceRecord)
def create_user_workspace(
    request: WorkspaceCreate,
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> WorkspaceRecord:
    return store.create_workspace(user.user_id, request.name.strip())


@api_router.get("/workspaces/{workspace_id}/projects", response_model=list[ProjectRecord])
def list_workspace_projects(
    workspace_id: str,
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> list[ProjectRecord]:
    return store.list_projects(user.user_id, workspace_id)


@api_router.post("/workspaces/{workspace_id}/projects", response_model=ProjectRecord)
def create_workspace_project(
    workspace_id: str,
    request: ProjectCreate,
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> ProjectRecord:
    try:
        return store.create_project(user.user_id, workspace_id, request.title.strip())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found.") from exc


@api_router.put("/projects/{project_id}", response_model=ProjectRecord)
def update_workspace_project(
    project_id: str,
    request: ProjectUpdate,
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> ProjectRecord:
    try:
        return store.update_project(user.user_id, project_id, title=request.title, draft_id=request.draft_id)
    except KeyError as exc:
        detail = "Draft not found." if str(exc) != project_id else "Project not found."
        raise HTTPException(status_code=404, detail=detail) from exc


@api_router.post("/compose")
async def compose(
    request: ComposeRequest,
    user: UserPublic = Depends(get_current_user),
    nim_client: NimClient = Depends(get_nim_client),
    store: DraftStore = Depends(get_store),
) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in nim_client.stream_compose_langgraph(request):
                if event["type"] == "complete":
                    versions = {}
                    for tier, comp_dict in event.get("compositions", {}).items():
                        if comp_dict:
                            comp = Composition.model_validate(comp_dict)
                            versions[tier] = optimize_generated_composition(comp)
                            
                    composition = versions.get("balanced")
                    if not composition:
                        yield f"data: {json.dumps({'error': 'Composition workflow did not return a balanced variant.'})}\n\n"
                        return

                    warnings = validate_composition(composition)
                    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
                    if blocking:
                        yield f"data: {json.dumps({'error': 'Generated composition failed validation.', 'warnings': warnings})}\n\n"
                        return

                    draft_id = store.create(user.user_id, composition)
                    
                    final_event = {
                        "type": "complete",
                        "draft_id": draft_id,
                        "composition": composition.model_dump(),
                        "versions": {k: v.model_dump() for k, v in versions.items()},
                        "warnings": warnings
                    }
                    yield f"data: {json.dumps(final_event)}\n\n"
                else:
                    yield f"data: {json.dumps(event)}\n\n"
        except NimTimeoutError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except Exception as exc:
            logger.exception("Composition generation failed")
            yield f"data: {json.dumps({'error': f'NVIDIA NIM generation failed: {exc}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@api_router.post("/refine", response_model=ComposeResponse)
async def refine(
    request: RefineRequest,
    user: UserPublic = Depends(get_current_user),
    nim_client: NimClient = Depends(get_nim_client),
    store: DraftStore = Depends(get_store),
) -> ComposeResponse:
    try:
        composition = optimize_generated_composition(await nim_client.refine(request.target, request.composition, request.instructions))
    except NimTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Composition refinement failed")
        raise HTTPException(status_code=502, detail=f"NVIDIA NIM refinement failed: {exc}") from exc

    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Refined composition failed validation.", "warnings": warnings})

    draft_id = store.create(user.user_id, composition)
    return ComposeResponse(draft_id=draft_id, composition=composition, warnings=warnings)


@api_router.post("/validate", response_model=ValidationReport)
def validate_draft(composition: Composition) -> ValidationReport:
    return ValidationReport(**composition_validation_report(composition))


@api_router.post("/evaluate", response_model=EvaluationReport)
def evaluate_draft(composition: Composition) -> EvaluationReport:
    return EvaluationReport(**evaluate_composition(composition))


@api_router.post("/commercial-review", response_model=CommercialReview)
def review_commercial_readiness(composition: Composition) -> CommercialReview:
    return CommercialReview(**commercial_review(composition))


@api_router.get("/drafts", response_model=list[DraftSummary])
def list_drafts(
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> list[DraftSummary]:
    return store.list(user.user_id)


@api_router.post("/drafts", response_model=DraftRecord)
def create_draft(
    composition: Composition,
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> DraftRecord:
    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Draft failed validation.", "warnings": warnings})
    draft_id = store.create(user.user_id, composition)
    record = store.get(user.user_id, draft_id)
    if not record:
        raise HTTPException(status_code=500, detail="Draft could not be created.")
    return record


@api_router.get("/drafts/{draft_id}", response_model=DraftRecord)
def get_draft(
    draft_id: str,
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> DraftRecord:
    record = store.get(user.user_id, draft_id)
    if not record:
        raise HTTPException(status_code=404, detail="Draft not found.")
    return record


@api_router.put("/drafts/{draft_id}", response_model=DraftRecord)
def update_draft(
    draft_id: str,
    composition: Composition,
    user: UserPublic = Depends(get_current_user),
    store: DraftStore = Depends(get_store),
) -> DraftRecord:
    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Draft failed validation.", "warnings": warnings})
    try:
        store.update(user.user_id, draft_id, composition)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Draft not found.") from exc
    record = store.get(user.user_id, draft_id)
    if not record:
        raise HTTPException(status_code=404, detail="Draft not found.")
    return record


@api_router.post("/export/midi")
def export_midi(composition: Composition) -> Response:
    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Cannot export invalid draft.", "warnings": warnings})
    midi_bytes = composition_to_midi_bytes(composition)
    safe_title = "".join(char for char in composition.title if char.isalnum() or char in (" ", "-", "_")).strip()
    filename = (safe_title or "composition").replace(" ", "_") + ".mid"
    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api_router.post("/export/wav")
def export_wav(composition: Composition, settings: Settings = Depends(get_settings)) -> Response:
    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Cannot export invalid draft.", "warnings": warnings})
    try:
        wav_bytes = composition_to_soundfont_wav_bytes(
            composition,
            fluidsynth_path=settings.resolved_fluidsynth_path,
            soundfont_path=settings.resolved_soundfont_path,
        )
    except Exception as exc:
        logger.warning("SoundFont render failed, using procedural fallback: %s", exc)
        wav_bytes = composition_to_wav_bytes(composition)
    safe_title = "".join(char for char in composition.title if char.isalnum() or char in (" ", "-", "_")).strip()
    filename = (safe_title or "composition").replace(" ", "_") + ".wav"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api_router.post("/export/mp3")
def export_mp3(composition: Composition, settings: Settings = Depends(get_settings)) -> Response:
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=503, detail="MP3 export requires FFmpeg. Install FFmpeg or use WAV export.")
    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Cannot export invalid draft.", "warnings": warnings})
    try:
        wav_bytes = composition_to_soundfont_wav_bytes(
            composition,
            fluidsynth_path=settings.resolved_fluidsynth_path,
            soundfont_path=settings.resolved_soundfont_path,
        )
    except Exception:
        wav_bytes = composition_to_wav_bytes(composition)
    safe_title = "".join(char for char in composition.title if char.isalnum() or char in (" ", "-", "_")).strip()
    filename = (safe_title or "composition").replace(" ", "_") + ".mp3"
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "input.wav"
        mp3_path = Path(temp_dir) / "output.mp3"
        wav_path.write_bytes(wav_bytes)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise HTTPException(
                status_code=503,
                detail="MP3 encoding failed. Install a working FFmpeg build or use WAV export.",
            ) from exc
        mp3_bytes = mp3_path.read_bytes()
    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api_router.post("/export/stems")
def export_stems(composition: Composition) -> Response:
    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Cannot export invalid draft.", "warnings": warnings})
    safe_title = "".join(char for char in composition.title if char.isalnum() or char in (" ", "-", "_")).strip()
    base_name = (safe_title or "composition").replace(" ", "_")
    package = BytesIO()
    with ZipFile(package, "w", ZIP_DEFLATED) as zip_file:
        for stem in ("rhythm", "harmony", "melody", "bass", "drums"):
            zip_file.writestr(f"{base_name}_{stem}.wav", composition_to_wav_bytes(composition, stem=stem))
        zip_file.writestr(f"{base_name}_stem_notes.json", json.dumps({
            "drum_pattern": composition.drum_pattern,
            "bassline": composition.bassline,
            "mix_notes": composition.mix_notes,
        }, indent=2))
    package.seek(0)
    return Response(
        content=package.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{base_name}_stems.zip"'},
    )


@api_router.post("/export/musicxml")
def export_musicxml(composition: Composition) -> Response:
    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Cannot export invalid draft.", "warnings": warnings})
    musicxml_bytes = composition_to_musicxml_bytes(composition)
    safe_title = "".join(char for char in composition.title if char.isalnum() or char in (" ", "-", "_")).strip()
    filename = (safe_title or "composition").replace(" ", "_") + ".musicxml"
    return Response(
        content=musicxml_bytes,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api_router.post("/export/notation")
def export_notation(composition: Composition) -> Response:
    notation = composition_to_notation_text(composition)
    safe_title = "".join(char for char in composition.title if char.isalnum() or char in (" ", "-", "_")).strip()
    filename = (safe_title or "composition").replace(" ", "_") + "_notation.txt"
    return Response(
        content=notation,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api_router.post("/export/package")
def export_package(composition: Composition, settings: Settings = Depends(get_settings)) -> Response:
    warnings = validate_composition(composition)
    blocking = [warning for warning in warnings if warning.startswith("Invalid")]
    if blocking:
        raise HTTPException(status_code=422, detail={"message": "Cannot export invalid draft.", "warnings": warnings})

    safe_title = "".join(char for char in composition.title if char.isalnum() or char in (" ", "-", "_")).strip()
    base_name = (safe_title or "composition").replace(" ", "_")
    package = BytesIO()
    evaluation = evaluate_composition(composition)
    review = commercial_review(composition)
    license_notes = (
        f"{composition.disclaimer}\n\n"
        "Review generated chords, melody, lyrics, MIDI, WAV, stems, and MusicXML before commercial release.\n"
        "Do not claim the output is guaranteed unique. Keep human review in the workflow.\n"
    )
    with ZipFile(package, "w", ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"{base_name}.json", json.dumps(composition.model_dump(), indent=2))
        zip_file.writestr(f"{base_name}.mid", composition_to_midi_bytes(composition))
        try:
            zip_file.writestr(
                f"{base_name}.wav",
                composition_to_soundfont_wav_bytes(
                    composition,
                    fluidsynth_path=settings.resolved_fluidsynth_path,
                    soundfont_path=settings.resolved_soundfont_path,
                ),
            )
        except Exception as exc:
            logger.warning("SoundFont package render failed, using procedural fallback: %s", exc)
            zip_file.writestr(f"{base_name}.wav", composition_to_wav_bytes(composition))
        zip_file.writestr(f"{base_name}_rhythm_stem.wav", composition_to_wav_bytes(composition, stem="rhythm"))
        zip_file.writestr(f"{base_name}_harmony_stem.wav", composition_to_wav_bytes(composition, stem="harmony"))
        zip_file.writestr(f"{base_name}_melody_stem.wav", composition_to_wav_bytes(composition, stem="melody"))
        zip_file.writestr(f"{base_name}.musicxml", composition_to_musicxml_bytes(composition))
        zip_file.writestr(f"{base_name}_notation.txt", composition_to_notation_text(composition))
        zip_file.writestr(f"{base_name}_evaluation.json", json.dumps(evaluation, indent=2))
        zip_file.writestr(f"{base_name}_commercial_review.json", json.dumps(review, indent=2))
        zip_file.writestr("LICENSE_AND_REVIEW_NOTES.txt", license_notes)
    package.seek(0)
    return Response(
        content=package.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{base_name}_package.zip"'},
    )


app.include_router(api_router, prefix=settings.normalized_api_prefix)

if settings.normalized_api_prefix != "/api":
    app.include_router(api_router, prefix="/api")
