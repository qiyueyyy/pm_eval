from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    api_url: str = os.getenv("TARGET_API_URL") or os.getenv("BEAUTYAGENT_API_URL", "http://localhost:8000/api/agent/chat")
    api_timeout: int = int(os.getenv("TARGET_API_TIMEOUT") or os.getenv("BEAUTYAGENT_API_TIMEOUT", "90"))
    mock_mode: bool = _bool_env("TARGET_MOCK_MODE", _bool_env("BEAUTYAGENT_MOCK_MODE", True))
    client_mode: str = os.getenv("TARGET_CLIENT_MODE") or os.getenv("BEAUTYAGENT_CLIENT_MODE", "mock")
    target_name: str = os.getenv("TARGET_NAME") or os.getenv("BEAUTYAGENT_TARGET_NAME", "Target API")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    db_path: str = os.getenv("PMEVAL_DB_PATH", str(ROOT_DIR / "data" / "pm_eval.sqlite"))


def get_settings() -> Settings:
    return Settings()
