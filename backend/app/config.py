"""
GenAI Runtime Risk Research Platform
Application Settings
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_debug: bool = False

    # Database
    database_url: str = "sqlite+aiosqlite:///./genai_runtime_risk.db"
    database_sync_url: str = "sqlite:///./genai_runtime_risk.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # ---- Single LLM Provider Selection ----
    # Set in .env: GENAI_PROVIDER=openai|anthropic|google
    genai_provider: str = "openai"
    genai_model_name: str = "gpt-4o"
    genai_model_version: str = "2024-11-20"

    # Provider API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # GenAI Parameters
    genai_temperature: float = 0.1
    genai_max_tokens: int = 2048

    # RAG
    chroma_persist_directory: str = "./data/chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 5

    # PD Model
    pd_model_version: str = "1.0.0"
    pd_model_frozen: bool = True

    # Paths
    data_raw_dir: str = "./data/raw"
    data_processed_dir: str = "./data/processed"
    reports_dir: str = "./reports"
    logs_dir: str = "./logs"

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/research_platform.log"

    # Research Metadata
    research_project_id: str = "GENAI-RUNTIME-RISK-001"
    research_version: str = "2.0.0"
    research_institution: str = "Research Institution"
    dataset_version: str = "v1.0.0"

    def get_active_api_key(self) -> Optional[str]:
        """Return the API key for the configured provider."""
        if self.genai_provider == "openai":
            return self.openai_api_key
        elif self.genai_provider == "anthropic":
            return self.anthropic_api_key
        elif self.genai_provider == "google":
            return self.google_api_key
        return None

    def is_llm_configured(self) -> bool:
        """Return True if the active provider key is set and not a placeholder."""
        key = self.get_active_api_key()
        if not key:
            return False
        placeholders = ("sk-...", "sk-ant-...", "AIza...", "your-key-here", "")
        return key not in placeholders and not key.endswith("...")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
