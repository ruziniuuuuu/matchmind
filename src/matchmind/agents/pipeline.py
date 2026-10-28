"""Five-agent MatchMind pipeline with shared evidence and verification."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any

from agent_framework import Agent, MCPStdioTool

from matchmind.agents.cache import load_cached, save_cached
from matchmind.agents.context import window
from matchmind.agents.contracts import (
    Claim,
    FanCard,
    InsightCandidate,
    MetricRef,
    OverlayCue,
    PipelineRequest,
    PipelineResult,
    RundownDecision,
    TraceStep,
    VerificationResult,
)
from matchmind.agents.director import (
    build_rundown,
    director_options,
    normalize_decision,
)
from matchmind.agents.grounding import ensure_grounded
from matchmind.agents.json_utils import run_json_agent
from matchmind.agents.llm import build_chat_client
from matchmind.agents.prompts import (
    ANALYST_INSTRUCTIONS,
    CHECKER_INSTRUCTIONS,
    DIRECTOR_INSTRUCTIONS,
    EDITOR_INSTRUCTIONS,
    PRODUCER_INSTRUCTIONS,
    PROMPT_VERSION,
    analyst_prompt,
    checker_prompt,
    director_prompt,
    editor_prompt,
    producer_prompt,
)
from matchmind.agents.retry import retry_async
from matchmind.agents.verifier import (
    fallback_card,
    fallback_cue,
    fallback_decision,
    fallback_insight,
    merge_verifications,
    verify_cue,
    verify_deterministic,
)
from matchmind.analytics import build_snapshot, detect_moments, load_match
from matchmind.analytics.models import FeatureSnapshot, Moment
from matchmind.config import settings
from matchmind.models.event import Event, MatchBundle

logger = logging.getLogger(__name__)


def _compact_metrics(snapshot: FeatureSnapshot) -> dict[str, Any]:
    return {
        name: {
            "value": metric.value,
            "unit": metric.unit,
            "formula_version": metric.formula_version,
            "event_ids": metric.event_ids,
        }
        for name, metric in snapshot.metrics.items()
    }


def build_checker_evidence(
    snapshot: FeatureSnapshot, moment: Moment, events: list[Event]
) -> dict[str, Any]:
    return {
        "metrics": _compact_metrics(snapshot),
        "moment": moment.model_dump(mode="json"),
        "event_window": [_compact_event(event) for event in events[-8:]],
    }


def _compact_event(event: Event) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "clock": event.clock.model_dump(),
        "type": event.type,
        "team_id": event.team_id,
        "player_id": event.player_id,
        "recipient_id": event.recipient_id,
        "location": event.location.model_dump(),
        "end_location": event.end_location.model_dump() if event.end_location else None,
        "pressure": event.pressure,
        "outcome": event.outcome,
        "ball_speed_kmh": event.ball_speed_kmh,
        "shot_speed_kmh": event.shot_speed_kmh,
        "speed_kmh": event.speed_kmh,
        "distance_m": event.distance_m,
        "difficulty": event.difficulty,
        "tags": event.tags,
        "score": event.score.model_dump(),
    }


class MatchMindPipeline:
    """Run the analyst → writers → verifier → director graph for one moment."""

    def __init__(self, use_mcp: bool = True) -> None:
        self.use_mcp = use_mcp

    async def _analyst(
        self,
        match_id: str,
        moment: Moment,
        snapshot: FeatureSnapshot,
        events: list[Event],
        tools: list[Any] | None,
    ) -> InsightCandidate:
        agent = Agent(
            client=build_chat_client(),
            name="TacticalAnalyst",
            instructions=ANALYST_INSTRUCTIONS,
            tools=tools,
        )
        prompt = analyst_prompt(
            match_id,
            moment.model_dump(mode="json"),
            {"metrics": _compact_metrics(snapshot), "clock": snapshot.clock.model_dump()},
            [_compact_event(event) for event in events],
        )
        return await asyncio.wait_for(
            run_json_agent(agent, prompt, InsightCandidate),
            timeout=settings.llm_timeout_seconds,
        )

    async def _producer(
        self, candidate: InsightCandidate, feedback: list[dict[str, Any]] | None = None
    ) -> OverlayCue:
        async def attempt() -> OverlayCue:
            agent = Agent(
                client=build_chat_client(),
                name="StoryProducer",
                instructions=PRODUCER_INSTRUCTIONS,
            )
            return await asyncio.wait_for(
                run_json_agent(
                    agent,
                    producer_prompt(candidate.model_dump(mode="json"), feedback),
                    OverlayCue,
                ),
                timeout=settings.llm_timeout_seconds,
            )

        return await retry_async(
            attempt,
            attempts=settings.agent_max_attempts,
            delay_seconds=settings.agent_retry_delay_seconds,
        )

    async def _editor(
        self, candidate: InsightCandidate, profile: str, language: str
    ) -> FanCard:
        async def attempt() -> FanCard:
            agent = Agent(
                client=build_chat_client(),
                name="AudienceEditor",
                instructions=EDITOR_INSTRUCTIONS,
            )
            return await asyncio.wait_for(
                run_json_agent(
                    agent,
                    editor_prompt(candidate.model_dump(mode="json"), profile, language),
                    FanCard,
                ),
                timeout=settings.llm_timeout_seconds,
            )

        return await retry_async(
            attempt,
            attempts=settings.agent_max_attempts,
            delay_seconds=settings.agent_retry_delay_seconds,
        )

    async def _checker(
        self, candidate: InsightCandidate, evidence: dict[str, Any]
    ) -> VerificationResult:
        async def attempt() -> VerificationResult:
            agent = Agent(
                client=build_chat_client(),
                name="FactChecker",
                instructions=CHECKER_INSTRUCTIONS,
            )
            return await asyncio.wait_for(
                run_json_agent(
                    agent,
                    checker_prompt(candidate.model_dump(mode="json"), evidence),
                    VerificationResult,
                    repair=False,
                ),
                timeout=min(35, settings.llm_timeout_seconds),
            )

        return await retry_async(
            attempt,
            attempts=settings.agent_max_attempts,
            delay_seconds=settings.agent_retry_delay_seconds,
        )

    async def _director(
        self,
        candidate: InsightCandidate,
        verification: VerificationResult,
        options: list[dict[str, Any]],
    ) -> RundownDecision:
        async def attempt() -> RundownDecision:
            agent = Agent(
                client=build_chat_client(),
                name="Director",
                instructions=DIRECTOR_INSTRUCTIONS,
            )
            return await asyncio.wait_for(
                run_json_agent(
                    agent,
                    director_prompt(
                        candidate.model_dump(mode="json"),
                        verification.model_dump(mode="json"),
                        options,
                    ),
                    RundownDecision,
                ),
                timeout=settings.llm_timeout_seconds,
            )

        return await retry_async(
            attempt,
            attempts=settings.agent_max_attempts,
            delay_seconds=settings.agent_retry_delay_seconds,
        )

    @staticmethod
    def _select_moment(bundle: MatchBundle, moment_id: str | None) -> Moment:
        moments = detect_moments(bundle)
        if not moments:
            raise ValueError("no moments detected")
        if moment_id:
            moment = next((item for item in moments if item.moment_id == moment_id), None)
            if moment is None:
                raise ValueError(f"unknown moment_id: {moment_id}")
            return moment
        return max(moments, key=lambda item: item.importance)

    async def _generate_cards(
        self,
        candidate: InsightCandidate,
        profiles: list[str],
        languages: list[str],
        trace: list[TraceStep],
    ) -> list[FanCard]:
        tasks = [
            self._editor(candidate, profile, language)
            for profile in profiles
            for language in languages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        cards: list[FanCard] = []
        for (profile, language), result in zip(
            [(p, lang) for p in profiles for lang in languages], results, strict=False
        ):
            if isinstance(result, Exception):
                trace.append(
                    TraceStep(
                        agent=f"AudienceEditor:{profile}:{language}",
                        model=settings.active_model,
                        prompt_version=PROMPT_VERSION,
                        latency_ms=0,
                        status="fallback",
                        error=str(result)[:300],
                    )
                )
                cards.append(fallback_card(candidate, profile, language))
            else:
                cards.append(result)
        return cards

    async def run(self, match_id: str, request: PipelineRequest) -> PipelineResult:
        bundle = load_match(match_id)
        moment = self._select_moment(bundle, request.moment_id)
        snapshot = build_snapshot(bundle, until_event_id=moment.event_id)
        events = window(bundle, moment.event_id)
        checker_evidence = build_checker_evidence(snapshot, moment, events)
        trace: list[TraceStep] = []
        fallback = False

        async def run_with_tool(tool: MCPStdioTool | None) -> InsightCandidate:
            start = time.perf_counter()
            try:
                candidate = await self._analyst(match_id, moment, snapshot, events, tool)
                trace.append(
                    TraceStep(
                        agent="TacticalAnalyst",
                        model=settings.active_model,
                        prompt_version=PROMPT_VERSION,
                        latency_ms=round((time.perf_counter() - start) * 1000, 1),
                        status="ok",
                    )
                )
                return candidate
            except Exception as exc:  # noqa: BLE001 - degrade to deterministic fallback
                trace.append(
                    TraceStep(
                        agent="TacticalAnalyst",
                        model=settings.active_model,
                        prompt_version=PROMPT_VERSION,
                        latency_ms=round((time.perf_counter() - start) * 1000, 1),
                        status="error",
                        error=str(exc)[:300],
                    )
                )
                raise

        try:
            if self.use_mcp and request.use_mcp:
                async with MCPStdioTool(
                    name="analytics",
                    command=sys.executable,
                    args=["-m", "matchmind.mcp.server"],
                    approval_mode="never_require",
                    allowed_tools=[
                        "get_event_window",
                        "get_tactical_chain",
                        "get_pressure_comparison",
                        "get_player_summary",
                    ],
                    request_timeout=settings.llm_timeout_seconds,
                ) as tool:
                    candidate = await run_with_tool(tool)
            else:
                candidate = await run_with_tool(None)
        except Exception:
            fallback = True
            try:
                candidate = await run_with_tool(None)
            except Exception:
                candidate = fallback_insight(moment, snapshot)
        candidate = ensure_grounded(candidate, bundle, moment, snapshot)

        if request.adversarial_claim:
            speed = moment.metrics.get("event_speed") or snapshot.metrics.get("top_speed")
            candidate.claims.append(
                Claim(
                    text=request.adversarial_claim,
                    event_ids=[moment.event_id],
                    metric_refs=(
                        [
                            MetricRef(
                                name=speed.name,
                                value=speed.value,
                                formula_version=speed.formula_version,
                                event_ids=speed.event_ids,
                            )
                        ]
                        if speed
                        else []
                    ),
                    confidence=0.5,
                )
            )

        producer_start = time.perf_counter()
        checker_start = time.perf_counter()
        cards_start = time.perf_counter()
        producer_task = asyncio.create_task(self._producer(candidate))
        checker_task = asyncio.create_task(self._checker(candidate, checker_evidence))
        cards_task = asyncio.create_task(
            self._generate_cards(candidate, request.profiles, request.languages, trace)
        )
        try:
            cue = await producer_task
            trace.append(
                TraceStep(
                    agent="StoryProducer",
                    model=settings.active_model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=round((time.perf_counter() - producer_start) * 1000, 1),
                    status="ok",
                )
            )
        except Exception as exc:  # noqa: BLE001
            fallback = True
            cue = fallback_cue(candidate)
            trace.append(
                TraceStep(
                    agent="StoryProducer",
                    model=settings.active_model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=0,
                    status="fallback",
                    error=str(exc)[:300],
                )
            )
        try:
            model_verification = await checker_task
            trace.append(
                TraceStep(
                    agent="FactChecker",
                    model=settings.active_model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=round((time.perf_counter() - checker_start) * 1000, 1),
                    status="ok",
                    error="" if model_verification.status == "pass" else model_verification.notes,
                )
            )
        except Exception as exc:  # noqa: BLE001
            fallback = True
            model_verification = None
            trace.append(
                TraceStep(
                    agent="FactChecker",
                    model=settings.active_model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=0,
                    status="fallback",
                    error=str(exc)[:300],
                )
            )
        fan_cards = await cards_task
        card_fallback = any(
            step.agent.startswith("AudienceEditor:") and step.status == "fallback"
            for step in trace
        )
        trace.append(
            TraceStep(
                agent="AudienceEditor",
                model=settings.active_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - cards_start) * 1000, 1),
                status="fallback" if card_fallback else "ok",
            )
        )

        events_by_id = {event.event_id: event for event in bundle.events}
        deterministic = verify_deterministic(candidate, snapshot, moment, events_by_id)
        cue_check = verify_cue(cue, candidate)
        deterministic = VerificationResult(
            status="reject" if "reject" in (deterministic.status, cue_check.status) else "pass",
            issues=deterministic.issues + cue_check.issues,
            checked_claims=deterministic.checked_claims,
            notes="deterministic candidate and cue verifier",
        )
        verification = merge_verifications(deterministic, model_verification)

        if verification.status == "revise":
            revise_start = time.perf_counter()
            try:
                cue = await self._producer(
                    candidate,
                    [issue.model_dump() for issue in model_verification.issues]
                    if model_verification
                    else [],
                )
                trace.append(
                    TraceStep(
                        agent="StoryProducer:revise",
                        model=settings.active_model,
                        prompt_version=PROMPT_VERSION,
                        latency_ms=round((time.perf_counter() - revise_start) * 1000, 1),
                        status="ok",
                    )
                )
                revised_evidence = {**checker_evidence, "cue": cue.model_dump(mode="json")}
                model_verification = await self._checker(candidate, revised_evidence)
                trace.append(
                    TraceStep(
                        agent="FactChecker:recheck",
                        model=settings.active_model,
                        prompt_version=PROMPT_VERSION,
                        latency_ms=round((time.perf_counter() - revise_start) * 1000, 1),
                        status="ok",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                fallback = True
                cue = fallback_cue(candidate)
                model_verification = None
                trace.append(
                    TraceStep(
                        agent="FactChecker:recheck",
                        model=settings.active_model,
                        prompt_version=PROMPT_VERSION,
                        latency_ms=0,
                        status="fallback",
                        error=str(exc)[:300],
                    )
                )
            cue_check = verify_cue(cue, candidate)
            deterministic = VerificationResult(
                status="reject" if cue_check.status == "reject" else deterministic.status,
                issues=deterministic.issues + cue_check.issues,
                checked_claims=deterministic.checked_claims,
                notes="deterministic revised cue verifier",
            )
            verification = merge_verifications(
                deterministic, model_verification, after_revision=True
            )

        if verification.status != "pass":
            decision = fallback_decision(verification)
            return PipelineResult(
                match_id=match_id,
                moment=moment,
                clock=moment.clock,
                insight=candidate,
                cue=None,
                fan_cards=[],
                verification=verification,
                decision=decision,
                trace=trace,
                fallback=fallback,
            )

        director_start = time.perf_counter()
        moments = detect_moments(bundle)
        try:
            decision = await self._director(
                candidate, verification, director_options(moments, moment, bundle.events)
            )
            trace.append(
                TraceStep(
                    agent="Director",
                    model=settings.active_model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=round((time.perf_counter() - director_start) * 1000, 1),
                    status="ok",
                )
            )
        except Exception as exc:  # noqa: BLE001
            fallback = True
            decision = fallback_decision(verification)
            trace.append(
                TraceStep(
                    agent="Director",
                    model=settings.active_model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=0,
                    status="fallback",
                    error=str(exc)[:300],
                )
            )
        decision = normalize_decision(decision, verification, moments, moment, bundle.events)

        return PipelineResult(
            match_id=match_id,
            moment=moment,
            clock=moment.clock,
            insight=candidate,
            cue=cue,
            fan_cards=fan_cards,
            verification=verification,
            decision=decision,
            rundown=build_rundown(decision, moments, moment),
            trace=trace,
            fallback=fallback,
        )


async def run_pipeline(match_id: str, request: PipelineRequest) -> PipelineResult:
    if request.use_cache and not request.adversarial_claim:
        cached = load_cached(match_id, request)
        if cached is not None:
            return cached
    if request.cache_only:
        raise FileNotFoundError("no cached insight for this moment")
    if settings.use_agent_graph and not request.adversarial_claim:
        try:
            from matchmind.agents.workflow import run_graph_pipeline

            result = await run_graph_pipeline(match_id, request)
        except Exception as exc:  # noqa: BLE001 - keep the local demo available
            logger.warning("agent graph failed, falling back to sequential pipeline: %s", exc)
            result = await MatchMindPipeline(use_mcp=request.use_mcp).run(match_id, request)
    else:
        result = await MatchMindPipeline(use_mcp=request.use_mcp).run(match_id, request)
    if (
        request.use_cache
        and not request.adversarial_claim
        and result.verification.status == "pass"
    ):
        save_cached(match_id, request, result)
    return result
