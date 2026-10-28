"""Provider factory wiring (no network)."""

import pytest

from matchmind.agents import llm
from matchmind.config import settings


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setattr(settings, k, v)


def test_azure_openai_key_auth(monkeypatch):
    _set(
        monkeypatch,
        llm_provider="azure_openai",
        azure_openai_endpoint="https://r.openai.azure.com",
        azure_openai_deployment="my-deploy",
        azure_openai_api_key="k",
        azure_openai_use_entra=False,
    )
    client = llm.build_chat_client()
    assert client.model == "my-deploy"
    assert settings.active_model == "my-deploy"


def test_azure_openai_requires_config(monkeypatch):
    _set(monkeypatch, llm_provider="azure_openai", azure_openai_endpoint="")
    with pytest.raises(RuntimeError):
        llm.build_chat_client()


def test_reasoning_client_strips_sampling_and_sets_effort(monkeypatch):
    _set(
        monkeypatch,
        llm_provider="azure_openai",
        azure_openai_endpoint="https://r.openai.azure.com",
        azure_openai_deployment="gpt-5",
        azure_openai_api_key="k",
        azure_openai_reasoning_effort="minimal",
    )
    from agent_framework import Message

    client = llm.build_chat_client()
    opts = client._prepare_options(
        [Message(role="user", contents=["hi"])],
        {"temperature": 0.2, "top_p": 0.9, "max_tokens": 100},
    )
    assert "temperature" not in opts and "top_p" not in opts
    assert opts["max_completion_tokens"] == 100
    assert opts["reasoning_effort"] == "minimal"
