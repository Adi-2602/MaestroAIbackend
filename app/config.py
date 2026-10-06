import shutil
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    app_name: str = Field(default="Maestro AI", alias="APP_NAME")
    node_env: str = Field(default="development", alias="NODE_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=5050, alias="PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")
    nim_api_key: str = Field(default="", validation_alias=AliasChoices("NVIDIA_API_KEY", "NIM_API_KEY"))
    nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com",
        validation_alias=AliasChoices("NVIDIA_NIM_BASE_URL", "NIM_BASE_URL"),
    )
    nim_model: str = Field(
        default="meta/llama-3.1-8b-instruct",
        validation_alias=AliasChoices("NVIDIA_NIM_MODEL", "NIM_MODEL"),
    )
    nim_timeout_seconds: float = Field(
        default=180,
        ge=30,
        le=600,
        validation_alias=AliasChoices("NVIDIA_NIM_TIMEOUT_SECONDS", "NIM_TIMEOUT_SECONDS"),
    )
    nim_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        validation_alias=AliasChoices("NVIDIA_NIM_RETRIES", "NIM_RETRIES"),
    )
    fluidsynth_path: str = Field(
        default="",
        validation_alias=AliasChoices("FLUIDSYNTH_PATH", "SOUNDFONT_RENDERER_PATH"),
    )
    soundfont_path: str = Field(
        default="",
        validation_alias=AliasChoices("SOUNDFONT_PATH", "SF2_PATH", "SF3_PATH"),
    )
    app_database_path: str = Field(default="./data/drafts.db", alias="APP_DATABASE_PATH")
    database_url: str = Field(default="", alias="DATABASE_URL")
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://localhost:4173,http://localhost",
        validation_alias=AliasChoices("CORS_ORIGIN", "ALLOWED_ORIGINS"),
    )
    smtp_host: str = Field(default="", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_sender: str = Field(default="noreply@maestro.studio", alias="SMTP_SENDER")
    smtp_use_tls: bool = Field(default=True, alias="SMTP_USE_TLS")

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    @property
    def origins(self) -> List[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def normalized_api_prefix(self) -> str:
        prefix = self.api_prefix.strip() or "/api/v1"
        return prefix if prefix.startswith("/") else f"/{prefix}"

    @property
    def normalized_nim_base_url(self) -> str:
        base_url = self.nim_base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return base_url
        return f"{base_url}/v1"

    @property
    def resolved_fluidsynth_path(self) -> str:
        if self.fluidsynth_path:
            path = Path(self.fluidsynth_path)
            return str(path if path.is_absolute() else ENV_FILE.parent / path)
        system_path = shutil.which("fluidsynth")
        if system_path:
            return system_path
        bundled = ENV_FILE.parent / "tools" / "fluidsynth" / "dist" / "fluidsynth-v2.5.4-win10-x64-cpp11" / "bin" / "fluidsynth.exe"
        return str(bundled) if bundled.exists() else ""

    @property
    def resolved_soundfont_path(self) -> str:
        if self.soundfont_path:
            path = Path(self.soundfont_path)
            return str(path if path.is_absolute() else ENV_FILE.parent / path)
        for bundled in (
            ENV_FILE.parent / "assets" / "soundfonts" / "MuseScore_General.sf3",
            ENV_FILE.parent / "assets" / "soundfonts" / "FluidR3_GM.sf2",
        ):
            if bundled.exists():
                return str(bundled)
        return ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
