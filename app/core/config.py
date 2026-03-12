"""
Application configuration — reads from environment / .env file.
All settings have safe defaults so the app starts without any env vars.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv() -> None:  # type: ignore
        pass

load_dotenv()


def _origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass
class Settings:
    # --- OpenAI / LLM ---
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or None
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.4"))
    )

    # --- Modes ---
    force_mock: bool = field(
        default_factory=lambda: os.getenv("FORCE_MOCK", "").strip().lower() in ("1", "true", "yes")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # --- Server ---
    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    # --- CORS ---
    allowed_origins: List[str] = field(default_factory=_origins)

    # --- Upload limits ---
    max_upload_size_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    )
    allowed_extensions: Tuple[str, ...] = (".txt", ".md", ".pdf")


# Module-level singleton — import and use directly
settings = Settings()
