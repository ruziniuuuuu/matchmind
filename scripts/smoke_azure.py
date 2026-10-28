#!/usr/bin/env python3
"""Smoke-test the configured Azure provider (azure_openai or foundry) end to end.

Usage:
    LLM_PROVIDER=azure_openai AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_DEPLOYMENT=... \
      AZURE_OPENAI_API_KEY=... uv run python scripts/smoke_azure.py
Never prints secrets.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


async def main() -> int:
    from agent_framework import Agent

    from matchmind.agents.llm import build_chat_client
    from matchmind.config import settings

    print(f"provider={settings.llm_provider} model={settings.active_model}")
    agent = Agent(
        client=build_chat_client(),
        name="SmokeAgent",
        instructions="Answer in one short sentence.",
    )
    result = await agent.run("Say 'azure connected' and nothing else.")
    print("[agent] PASS:", (getattr(result, "text", None) or str(result))[:200])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
