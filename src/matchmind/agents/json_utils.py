"""Structured-output helpers for agents."""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, ValidationError

from matchmind.agents.budget import consume_llm_call
from matchmind.config import settings

_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _semaphore() -> asyncio.Semaphore | None:
    limit = settings.agent_max_concurrency
    if limit <= 0:
        return None
    loop_id = id(asyncio.get_running_loop())
    return _SEMAPHORES.setdefault(loop_id, asyncio.Semaphore(limit))


def _is_rate_limited(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return "429" in text or "RateLimit" in text or "rate limit" in text.lower()


async def _run_agent(agent, prompt: str):
    """agent.run with optional concurrency cap and 429 backoff (Azure TPM quotas)."""
    sem = _semaphore()
    delay = 6.0
    for attempt in range(settings.agent_rate_limit_retries + 1):
        try:
            if sem is None:
                return await agent.run(prompt)
            async with sem:
                return await agent.run(prompt)
        except Exception as exc:  # noqa: BLE001
            if attempt >= settings.agent_rate_limit_retries or not _is_rate_limited(exc):
                raise
            await asyncio.sleep(delay)
            delay *= 2


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(cleaned[start : end + 1])


async def run_json_agent[T: BaseModel](
    agent,
    prompt: str,
    schema: type[T],
    *,
    repair: bool = True,
) -> T:
    """Run an agent and validate its JSON output, with one repair attempt."""
    consume_llm_call()
    result = await _run_agent(agent, prompt)
    text = getattr(result, "text", None) or str(result)
    try:
        return schema.model_validate(extract_json(text))
    except (ValueError, ValidationError) as first_error:
        if not repair:
            raise
        repair_prompt = (
            f"{prompt}\n\nYour previous response was invalid: {first_error}.\n"
            f"Return only valid JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
        consume_llm_call()
        result = await _run_agent(agent, repair_prompt)
        text = getattr(result, "text", None) or str(result)
        return schema.model_validate(extract_json(text))
