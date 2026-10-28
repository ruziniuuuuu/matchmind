"""Real Agent Framework workflow graph for the five MatchMind agents.

The graph is intentionally typed:

    prepare -> analyst -> fan-out(producer, audience, checker) -> fan-in(finalize)

The same agent methods used by the sequential pipeline are wrapped as graph
executors, so the product can run either path without duplicating prompts.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from agent_framework import MCPStdioTool, WorkflowBuilder, WorkflowContext, executor

from matchmind.agents.context import window
from matchmind.agents.contracts import (
    FanCard,
    InsightCandidate,
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
from matchmind.agents.pipeline import MatchMindPipeline, build_checker_evidence
from matchmind.agents.prompts import PROMPT_VERSION
from matchmind.agents.verifier import (
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

_PIPELINE = MatchMindPipeline()


@dataclass
class GraphInput:
    match_id: str
    request: PipelineRequest


@dataclass
class GraphState:
    match_id: str
    request: PipelineRequest
    bundle: MatchBundle
    moment: Moment
    snapshot: FeatureSnapshot
    events: list[Event]
    trace: list[TraceStep] = field(default_factory=list)


@dataclass
class GraphEnvelope:
    state: GraphState
    candidate: InsightCandidate
    cue: OverlayCue | None = None
    cards: list[FanCard] | None = None
    verification: VerificationResult | None = None
    revision_count: int = 0


@executor
async def prepare(input: GraphInput, ctx: WorkflowContext[GraphState]) -> None:
    bundle = load_match(input.match_id)
    moments = detect_moments(bundle)
    if not moments:
        raise ValueError("no moments detected")
    if input.request.moment_id:
        moment = next(
            (item for item in moments if item.moment_id == input.request.moment_id), None
        )
        if moment is None:
            raise ValueError(f"unknown moment_id: {input.request.moment_id}")
    else:
        moment = next(
            (
                item
                for item in moments
                if item.type == "line_break" and item.clock.minute >= 70
            ),
            max(moments, key=lambda item: item.importance),
        )
    snapshot = build_snapshot(bundle, until_event_id=moment.event_id)
    events = window(bundle, moment.event_id)
    await ctx.send_message(
        GraphState(
            match_id=input.match_id,
            request=input.request,
            bundle=bundle,
            moment=moment,
            snapshot=snapshot,
            events=events,
        )
    )


@executor
async def analyst(state: GraphState, ctx: WorkflowContext[GraphEnvelope]) -> None:
    started = time.perf_counter()
    candidate: InsightCandidate | None = None
    if state.request.use_mcp:
        try:
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
                candidate = await _PIPELINE._analyst(
                    state.match_id, state.moment, state.snapshot, state.events, tool
                )
        except Exception:
            candidate = None
    if candidate is None:
        try:
            candidate = await _PIPELINE._analyst(
                state.match_id, state.moment, state.snapshot, state.events, None
            )
        except Exception as exc:  # noqa: BLE001 - deterministic fallback
            state.trace.append(
                TraceStep(
                    agent="TacticalAnalyst",
                    model=settings.active_model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                    status="error",
                    error=str(exc)[:300],
                )
            )
            candidate = fallback_insight(state.moment, state.snapshot)
    candidate = ensure_grounded(candidate, state.bundle, state.moment, state.snapshot)
    if candidate is not None and not state.trace:
        state.trace.append(
            TraceStep(
                agent="TacticalAnalyst",
                model=settings.active_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                status="ok",
            )
        )
    await ctx.send_message(GraphEnvelope(state=state, candidate=candidate))


@executor
async def producer(envelope: GraphEnvelope, ctx: WorkflowContext[GraphEnvelope]) -> None:
    started = time.perf_counter()
    try:
        cue = await _PIPELINE._producer(envelope.candidate)
        envelope.state.trace.append(
            TraceStep(
                agent="StoryProducer",
                model=settings.active_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                status="ok",
            )
        )
    except Exception as exc:  # noqa: BLE001
        cue = fallback_cue(envelope.candidate)
        envelope.state.trace.append(
            TraceStep(
                agent="StoryProducer",
                model=settings.active_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                status="fallback",
                error=str(exc)[:300],
            )
        )
    envelope.cue = cue
    await ctx.send_message(envelope)


@executor
async def audience(envelope: GraphEnvelope, ctx: WorkflowContext[GraphEnvelope]) -> None:
    started = time.perf_counter()
    cards = await _PIPELINE._generate_cards(
        envelope.candidate,
        envelope.state.request.profiles,
        envelope.state.request.languages,
        envelope.state.trace,
    )
    envelope.cards = cards
    envelope.state.trace.append(
        TraceStep(
            agent="AudienceEditor",
            model=settings.active_model,
            prompt_version=PROMPT_VERSION,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            status="ok",
        )
    )
    await ctx.send_message(envelope)


@executor
async def checker(envelope: GraphEnvelope, ctx: WorkflowContext[GraphEnvelope]) -> None:
    started = time.perf_counter()
    evidence = build_checker_evidence(
        envelope.state.snapshot, envelope.state.moment, envelope.state.events
    )
    try:
        result = await _PIPELINE._checker(envelope.candidate, evidence)
        envelope.state.trace.append(
            TraceStep(
                agent="FactChecker",
                model=settings.active_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                status="ok",
                error="" if result.status == "pass" else result.notes,
            )
        )
    except Exception as exc:  # noqa: BLE001
        result = None
        envelope.state.trace.append(
            TraceStep(
                agent="FactChecker",
                model=settings.active_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                status="fallback",
                error=str(exc)[:300],
            )
        )
    envelope.verification = result
    await ctx.send_message(envelope)


def deterministic_review(envelope: GraphEnvelope) -> VerificationResult:
    state = envelope.state
    candidate_check = verify_deterministic(
        envelope.candidate,
        state.snapshot,
        state.moment,
        {event.event_id: event for event in state.bundle.events},
    )
    cue_check = (
        verify_cue(envelope.cue, envelope.candidate)
        if envelope.cue
        else VerificationResult(status="reject", notes="missing broadcast cue")
    )
    return VerificationResult(
        status="reject" if "reject" in (candidate_check.status, cue_check.status) else "pass",
        issues=candidate_check.issues + cue_check.issues,
        checked_claims=candidate_check.checked_claims,
        notes="deterministic candidate and cue verifier",
    )


@executor
async def review(
    envelopes: list[GraphEnvelope], ctx: WorkflowContext[GraphEnvelope]
) -> None:
    if not envelopes:
        raise RuntimeError("workflow produced no envelopes")
    first = envelopes[0]
    model_verification = next(
        (item.verification for item in envelopes if item.verification is not None), None
    )
    merged = GraphEnvelope(
        state=first.state,
        candidate=first.candidate,
        cue=next((item.cue for item in envelopes if item.cue is not None), None),
        cards=next((item.cards for item in envelopes if item.cards is not None), []),
        revision_count=first.revision_count,
    )
    merged.verification = merge_verifications(
        deterministic_review(merged), model_verification
    )
    await ctx.send_message(merged)


@executor
async def revise(envelope: GraphEnvelope, ctx: WorkflowContext[GraphEnvelope]) -> None:
    started = time.perf_counter()
    feedback = envelope.verification.issues if envelope.verification else []
    try:
        envelope.cue = await _PIPELINE._producer(
            envelope.candidate, [issue.model_dump() for issue in feedback]
        )
        status = "ok"
        error = ""
    except Exception as exc:  # noqa: BLE001
        envelope.cue = fallback_cue(envelope.candidate)
        status = "fallback"
        error = str(exc)[:300]
    envelope.state.trace.append(
        TraceStep(
            agent="StoryProducer:revise",
            model=settings.active_model,
            prompt_version=PROMPT_VERSION,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            status=status,
            error=error,
        )
    )
    envelope.revision_count += 1
    await ctx.send_message(envelope)


@executor
async def recheck(envelope: GraphEnvelope, ctx: WorkflowContext[GraphEnvelope]) -> None:
    started = time.perf_counter()
    evidence = build_checker_evidence(
        envelope.state.snapshot, envelope.state.moment, envelope.state.events
    )
    evidence["cue"] = envelope.cue.model_dump(mode="json") if envelope.cue else None
    try:
        model_verification = await _PIPELINE._checker(envelope.candidate, evidence)
        status = "ok"
        error = ""
    except Exception as exc:  # noqa: BLE001
        model_verification = None
        status = "fallback"
        error = str(exc)[:300]
    envelope.state.trace.append(
        TraceStep(
            agent="FactChecker:recheck",
            model=settings.active_model,
            prompt_version=PROMPT_VERSION,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            status=status,
            error=error,
        )
    )
    envelope.verification = merge_verifications(
        deterministic_review(envelope), model_verification, after_revision=True
    )
    await ctx.send_message(envelope)


@executor
async def finalize(envelope: GraphEnvelope, ctx: WorkflowContext[PipelineResult]) -> None:
    state = envelope.state
    verification = envelope.verification or deterministic_review(envelope)
    if verification.status != "pass":
        await ctx.yield_output(
            PipelineResult(
                match_id=state.match_id,
                moment=state.moment,
                clock=state.moment.clock,
                insight=envelope.candidate,
                cue=None,
                fan_cards=[],
                verification=verification,
                decision=fallback_decision(verification),
                trace=state.trace,
                fallback=any(step.status != "ok" for step in state.trace),
            )
        )
        return

    director_started = time.perf_counter()
    moments = detect_moments(state.bundle)
    try:
        decision: RundownDecision = await _PIPELINE._director(
            envelope.candidate,
            verification,
            director_options(moments, state.moment, state.bundle.events),
        )
        state.trace.append(
            TraceStep(
                agent="Director",
                model=settings.active_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - director_started) * 1000, 1),
                status="ok",
            )
        )
    except Exception as exc:  # noqa: BLE001
        decision = fallback_decision(verification)
        state.trace.append(
            TraceStep(
                agent="Director",
                model=settings.active_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=round((time.perf_counter() - director_started) * 1000, 1),
                status="fallback",
                error=str(exc)[:300],
            )
        )
    decision = normalize_decision(
        decision, verification, moments, state.moment, state.bundle.events
    )

    await ctx.yield_output(
        PipelineResult(
            match_id=state.match_id,
            moment=state.moment,
            clock=state.moment.clock,
            insight=envelope.candidate,
            cue=envelope.cue,
            fan_cards=envelope.cards or [],
            verification=verification,
            decision=decision,
            rundown=build_rundown(decision, moments, state.moment),
            trace=state.trace,
            fallback=any(step.status != "ok" for step in state.trace),
        )
    )


def build_workflow():
    builder = WorkflowBuilder(start_executor=prepare)
    builder.add_edge(prepare, analyst)
    builder.add_fan_out_edges(analyst, [producer, audience, checker])
    builder.add_fan_in_edges([producer, audience, checker], review)
    builder.add_edge(
        review,
        revise,
        condition=lambda envelope: envelope.verification.status == "revise"
        and envelope.revision_count < 1,
    )
    builder.add_edge(
        review,
        finalize,
        condition=lambda envelope: envelope.verification.status != "revise",
    )
    builder.add_edge(revise, recheck)
    builder.add_edge(recheck, finalize)
    return builder.build()


_WORKFLOW = build_workflow()


async def run_graph_pipeline(match_id: str, request: PipelineRequest) -> PipelineResult:
    result = await _WORKFLOW.run(GraphInput(match_id=match_id, request=request))
    outputs = result.get_outputs()
    if not outputs:
        raise RuntimeError("agent workflow produced no output")
    return outputs[0]
