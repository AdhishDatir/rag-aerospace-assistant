"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings shared across the application."""

    app_name: str = "Aerospace RAG Assistant"
    app_version: str = "0.1.0"
    openai_api_key: str | None = None
    openai_model: str = "gpt-6-astra"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()