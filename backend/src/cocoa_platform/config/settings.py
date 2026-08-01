"""Environment and model-registry loading without hard-coded model paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _load_local_dotenv() -> None:
    """Load a local .env without adding another runtime dependency."""
    dotenv = PROJECT_ROOT / ".env"
    if not dotenv.is_file():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    admin_email: str
    admin_password_hash: str
    auth_secret: str
    cors_allowed_origins: tuple[str, ...]
    registry_path: Path


def get_settings() -> Settings:
    _load_local_dotenv()
    origins = os.getenv("COCOA_CORS_ALLOWED_ORIGINS", "http://localhost:5173")
    return Settings(
        admin_email=os.getenv("COCOA_ADMIN_EMAIL", "admin@chula.ac.th"),
        admin_password_hash=os.getenv("COCOA_ADMIN_PASSWORD_HASH", ""),
        auth_secret=os.getenv("COCOA_AUTH_SECRET", ""),
        cors_allowed_origins=tuple(item.strip() for item in origins.split(",") if item.strip()),
        registry_path=Path(os.getenv(
            "COCOA_MODEL_REGISTRY",
            str(PROJECT_ROOT / "backend" / "config" / "model_registry.json"),
        )),
    )


def load_registry(path: Path | None = None) -> dict:
    """Load the version-controlled registry; resolve all relative paths at use time."""
    registry_path = path or get_settings().registry_path
    with registry_path.open("r", encoding="utf-8") as registry_file:
        registry = json.load(registry_file)
    if registry.get("version") != 1:
        raise ValueError("Unsupported model registry version")
    return registry


def resolve_model_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
