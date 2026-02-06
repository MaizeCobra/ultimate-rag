"""Configuration management using Pydantic Settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file="../.env",  # Parent directory 
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
    )
    
    # Database
    database_url: str
    postgres_user: str
    postgres_password: str
    postgres_db: str
    
    # Gemini API
    gemini_api_key: str
    gemini_chat_model: str = "models/gemini-2.5-pro"
    gemini_flash_model: str = "models/gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    
    # Google Drive (optional for Phase 1)
    google_drive_client_id: str = ""
    google_drive_client_secret: str = ""
    google_drive_folder_id: str = ""
    google_drive_refresh_token: str = ""
    
    # Webhook Auth
    webhook_auth_header: str = "X-API-Key"
    webhook_auth_value: str = ""
    
    @property
    def async_database_url(self) -> str:
        """Convert standard URL to asyncpg format."""
        return self.database_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
