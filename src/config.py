"""Application configuration, loaded from environment variables / .env.

Nothing sensitive (API keys, secrets) is ever hard-coded here — all such
values come from the environment. See .env.example for the full list.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Medical Imaging AI Copilot"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # --- LLM provider selection (used from Day 9 onward) ---
    llm_provider: str = "ollama"
    # Optional second provider (Section 21: "Invalid -> retry / fallback").
    # If set, the Copilot pipeline tries this provider after the primary
    # exhausts its retries, instead of failing outright. None = no
    # fallback, matches every prior day's single-provider behavior.
    llm_fallback_provider: str | None = None

    groq_api_key: str | None = None
    groq_model: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    anthropic_api_key: str | None = None
    claude_model: str | None = None

    # --- Trained model checkpoints (used from Day 8 onward) ---
    model_2d_checkpoint_path: str = "training/checkpoints/model_2d_best.pth"
    model_3d_checkpoint_path: str = "training/checkpoints/model_3d_best.pth"

    # --- Reports history storage (Day 11) ---
    # Empty string means "use storage/database.py's own default path" --
    # kept configurable (not hardcoded) so tests can point at an
    # isolated temp DB instead of the real one.
    database_url: str = ""


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — import and call this, don't instantiate
    Settings() directly, so the whole app shares one instance."""
    return Settings()
