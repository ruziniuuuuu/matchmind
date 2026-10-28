"""Model-provider factory for agents.

LLM_PROVIDER = deepseek | azure_openai | foundry
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from agent_framework.openai import OpenAIChatCompletionClient

from matchmind.config import settings


def _is_reasoning_model(name: str) -> bool:
    n = name.lower()
    return n.startswith(("gpt-5", "o1", "o3", "o4"))


class _AzureReasoningChatClient(OpenAIChatCompletionClient):
    """Chat Completions client that is safe for gpt-5 / o-series reasoning deployments.

    Reasoning models reject non-default temperature/top_p and max_tokens. The framework already
    maps max_tokens -> max_completion_tokens; here we also drop sampling params and set
    reasoning_effort (Chat Completions accepts it on gpt-5 / o-series).
    """

    def _prepare_options(self, messages, options):  # type: ignore[override]
        run_options = super()._prepare_options(messages, options)
        if _is_reasoning_model(str(run_options.get("model") or "")):
            for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                run_options.pop(key, None)
            if settings.azure_openai_reasoning_effort:
                run_options.setdefault("reasoning_effort", settings.azure_openai_reasoning_effort)
        return run_options


@lru_cache(maxsize=1)
def _entra_credential() -> Any:
    """One shared Entra ID credential (az login locally, managed identity on Azure)."""
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


@lru_cache(maxsize=1)
def _foundry_credential() -> Any:
    from azure.identity.aio import DefaultAzureCredential

    return DefaultAzureCredential()


def build_chat_client(model: str | None = None) -> Any:
    """Build a chat client for the configured provider.

    DeepSeek's Responses API currently mishandles tool-call outputs in the
    Agent Framework adapter, so DeepSeek and Azure OpenAI use Chat Completions,
    which is verified to work with MCP and function tools. ``foundry`` uses the
    Foundry project Responses endpoint.
    """
    provider = settings.llm_provider.lower()
    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        return OpenAIChatCompletionClient(
            model=model or settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    if provider == "azure_openai":
        deployment = model or settings.azure_openai_deployment
        if not deployment or not (settings.azure_openai_endpoint or settings.azure_openai_base_url):
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_BASE_URL) and "
                "AZURE_OPENAI_DEPLOYMENT are required"
            )
        kwargs: dict[str, Any] = {"model": deployment}
        if settings.azure_openai_base_url:
            kwargs["base_url"] = settings.azure_openai_base_url
        else:
            kwargs["azure_endpoint"] = settings.azure_openai_endpoint
            kwargs["api_version"] = settings.azure_openai_api_version
        if settings.azure_openai_api_key and not settings.azure_openai_use_entra:
            kwargs["api_key"] = settings.azure_openai_api_key
        else:
            kwargs["credential"] = _entra_credential()
        return _AzureReasoningChatClient(**kwargs)
    if provider == "foundry":
        if not settings.foundry_project_endpoint or not settings.foundry_model:
            raise RuntimeError("FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL are required")
        from agent_framework.foundry import FoundryChatClient

        return FoundryChatClient(
            project_endpoint=settings.foundry_project_endpoint,
            model=model or settings.foundry_model,
            credential=_foundry_credential(),  # required; previous code omitted it
        )
    raise RuntimeError(f"unsupported LLM_PROVIDER: {settings.llm_provider}")
