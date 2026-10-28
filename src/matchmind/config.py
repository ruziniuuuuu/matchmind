"""Application settings loaded from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    The defaults are safe for local demo mode: no network calls and no secrets.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Model provider
    llm_provider: str = "deepseek"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_model_deep: str = "deepseek-v4-pro"

    # Azure OpenAI / Foundry-resource endpoint (Chat Completions, verified with MCP tools)
    azure_openai_endpoint: str = ""  # https://<resource>.openai.azure.com
    azure_openai_base_url: str = ""  # optional full ".../openai/v1/" URL (overrides endpoint)
    azure_openai_api_key: str = ""  # empty + azure_openai_use_entra=true -> Entra ID auth
    azure_openai_use_entra: bool = False
    azure_openai_api_version: str = "2025-04-01-preview"
    azure_openai_deployment: str = ""  # deployment NAME, not the base model name
    azure_openai_deployment_deep: str = ""
    azure_openai_reasoning_effort: str = "minimal"  # gpt-5/o-series; "" = omit

    # Optional Microsoft Foundry production path (Responses API, Entra ID auth)
    foundry_project_endpoint: str = ""
    foundry_model: str = ""

    # App
    app_env: str = "dev"
    log_level: str = "INFO"
    demo_mode: str = "replay"  # replay | live
    allow_live_generation: bool = True
    use_agent_graph: bool = True
    agent_max_attempts: int = 2
    agent_retry_delay_seconds: float = 0.35
    agent_max_concurrency: int = 0  # 0 = unlimited; set 2-3 for low Azure TPM quotas
    agent_rate_limit_retries: int = 3  # 429 backoff retries (6s, 12s, 24s)
    match_data_dir: str = "./data/demo"
    audit_db_path: str = "./data/audit.sqlite3"
    cache_dir: str = "./data/cache"
    public_base_url: str = "http://localhost:8000"

    # Optional Azure production path
    applicationinsights_connection_string: str = ""
    azure_cosmos_endpoint: str = ""
    azure_cosmos_database: str = "matchmind"
    azure_key_vault_uri: str = ""

    # Safety / budget
    max_llm_calls_per_match: int = 60
    max_cost_usd_per_match: float = 2.0
    live_rate_limit_per_minute: int = 10
    llm_timeout_seconds: int = 60

    @property
    def active_model(self) -> str:
        """Model / deployment name of the active provider (used in traces and audit)."""
        provider = self.llm_provider.lower()
        if provider == "azure_openai":
            return self.azure_openai_deployment
        if provider == "foundry":
            return self.foundry_model
        return self.deepseek_model


settings = Settings()
