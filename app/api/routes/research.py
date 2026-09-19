"""
FastAPI research routes for multi-agent workflows, synchronous evaluation,
and the measured benchmark suite.

Concurrency design: research runs are CPU/IO-bound synchronous pipelines.
They execute in a bounded process-wide thread pool (sized from settings)
via anyio, so the event loop stays responsive and burst load degrades
gracefully instead of oversubscribing threads.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anyio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request

from app.api.schemas.requests import BenchmarkRunRequest, ResearchQuery
from app.api.schemas.responses import (
    AgentTelemetryInfo,
    BenchmarkResponse,
    CritiqueInfo,
    ResearchResponse,
    SourceInfo,
    VerificationInfo,
)
from app.config import get_settings
from app.utils.logger import setup_logger
from app.utils.task_store import TaskStore, create_task_store

logger = setup_logger(__name__, "INFO")
router = APIRouter(prefix="/api/v1/research", tags=["Research"])

# Shared task store (Redis when configured, in-memory otherwise).
TASKS: TaskStore = create_task_store()

# Bounded concurrency for pipeline executions: prevents thread
# oversubscription under burst load while keeping multiple pipelines
# genuinely parallel. Sized from settings.
_PIPELINE_LIMITER: Optional[anyio.CapacityLimiter] = None


def _get_limiter() -> anyio.CapacityLimiter:
    """Lazily create the capacity limiter inside the running event loop."""
    global _PIPELINE_LIMITER
    if _PIPELINE_LIMITER is None:
        settings = get_settings()
        _PIPELINE_LIMITER = anyio.CapacityLimiter(
            max(4, min(settings.MAX_CONCURRENT_REQUESTS, 16))
        )
    return _PIPELINE_LIMITER


def _build_sources(state_sources: Optional[List[dict]]) -> List[SourceInfo]:
    """Format raw source dicts into SourceInfo objects."""
    sources = []
    for s in state_sources or []:
        sources.append(
            SourceInfo(
                title=s.get("title", "Reference Source"),
                url_or_path=s.get("source", s.get("url_or_path", "")),
                relevance_score=float(s.get("relevance_score", 0.0)),
                source_type=s.get("source_type", "document"),
                snippet=s.get("snippet", ""),
            )
        )
    return sources


def _build_telemetry(raw_tel: dict) -> AgentTelemetryInfo:
    """Format measured agent telemetry."""
    return AgentTelemetryInfo(
        planner_time_ms=raw_tel.get("planner_time_ms", 0.0),
        retriever_time_ms=raw_tel.get("retriever_time_ms", 0.0),
        analyzer_time_ms=raw_tel.get("analyzer_time_ms", 0.0),
        writer_time_ms=raw_tel.get("writer_time_ms", 0.0),
        verifier_time_ms=raw_tel.get("verifier_time_ms", 0.0),
        critic_time_ms=raw_tel.get("critic_time_ms", 0.0),
        total_latency_ms=raw_tel.get("total_latency_ms", 0.0),
        sequential_retrieval_baseline_ms=raw_tel.get(
            "sequential_retrieval_baseline_ms", 0.0
        ),
        parallel_retrieval_actual_ms=raw_tel.get("parallel_retrieval_actual_ms", 0.0),
        synthesis_reduction_pct=raw_tel.get("synthesis_reduction_pct", 0.0),
    )


def _build_verification(raw: dict) -> Optional[VerificationInfo]:
    """Format verifier output, if present."""
    if not raw:
        return None
    return VerificationInfo(
        grounded_ratio=raw.get("grounded_ratio", 0.0),
        total_sentences=raw.get("total_sentences", 0),
        grounded_sentences=raw.get("grounded_sentences", 0),
        citation_integrity=raw.get("citation_integrity", 0.0),
        measured_accuracy=raw.get("measured_accuracy", 0.0),
        method=raw.get("method", ""),
        ungrounded_sentences=[str(u) for u in raw.get("ungrounded_sentences", [])],
    )


def _build_critique(raw: dict) -> Optional[CritiqueInfo]:
    """Format critic output, if present."""
    if not raw:
        return None
    return CritiqueInfo(
        quality_score=raw.get("quality_score", 0.0),
        grounding_component=raw.get("grounding_component", 0.0),
        structure_component=raw.get("structure_component", 0.0),
        evidence_component=raw.get("evidence_component", 0.0),
        revision_notes=[str(n) for n in raw.get("revision_notes", [])],
        needs_revision=bool(raw.get("needs_revision", False)),
        method=raw.get("method", ""),
    )


def _state_to_response(
    task_id: str,
    query: str,
    orchestrator: str,
    state: Dict[str, Any],
    elapsed_s: float,
) -> ResearchResponse:
    """Convert a final pipeline state into a ResearchResponse."""
    return ResearchResponse(
        task_id=task_id,
        status="completed",
        query=query,
        report=state.get("final_report", ""),
        sources=_build_sources(state.get("sources_cited", state.get("sources", []))),
        confidence_score=float(state.get("confidence_score", 0.0)),
        response_accuracy_score=float(state.get("response_accuracy_score", 0.0)),
        synthesis_speedup_ratio=float(state.get("synthesis_speedup_ratio", 0.0)),
        processing_time_seconds=round(elapsed_s, 2),
        orchestrator=orchestrator,
        telemetry=_build_telemetry(state.get("agent_telemetry", {})),
        verification=_build_verification(state.get("verification", {})),
        critique=_build_critique(state.get("critique", {})),
        revision_count=int(state.get("revision_count", 0)),
        created_at=datetime.now(timezone.utc),
    )


def _execute_pipeline(
    query: str, rag_mode: str, orchestrator: str, app_state: Any
) -> Dict[str, Any]:
    """Run the selected orchestrator synchronously and return final state."""
    if orchestrator == "crewai":
        from app.agents.crew import ResearchCrew

        vstore = getattr(app_state, "vector_store", None)
        crew = ResearchCrew(vstore)
        return crew.run(query)

    graph = getattr(app_state, "research_graph", None)
    if graph is None:
        from app.agents.graph import ResearchGraph

        vstore = getattr(app_state, "vector_store", None)
        graph = ResearchGraph(vstore)
    return dict(graph.run(query, rag_mode=rag_mode))


def perform_research_task(
    task_id: str,
    query: str,
    rag_mode: str,
    orchestrator: str,
    app_state: Any,
) -> None:
    """Background task executing the multi-agent workflow."""
    start_time = time.perf_counter()
    try:
        state = _execute_pipeline(query, rag_mode, orchestrator, app_state)
        elapsed = time.perf_counter() - start_time
        TASKS.set(
            task_id,
            _state_to_response(task_id, query, orchestrator, state, elapsed),
        )
    except Exception as exc:  # noqa: BLE001 - record failure for the client
        logger.error("Research task %s failed: %s", task_id, exc, exc_info=True)
        TASKS.set(
            task_id,
            ResearchResponse(
                task_id=task_id,
                status="failed",
                query=query,
                report=f"Execution error: {exc}",
                created_at=datetime.now(timezone.utc),
            ),
        )


@router.post("/", response_model=ResearchResponse)
async def create_research_task(
    query_obj: ResearchQuery,
    request: Request,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    """Initiates an asynchronous multi-agent research workflow."""
    task_id = str(uuid.uuid4())
    response = ResearchResponse(
        task_id=task_id,
        status="pending",
        query=query_obj.query,
        orchestrator=query_obj.orchestrator,
        created_at=datetime.now(timezone.utc),
    )
    TASKS.set(task_id, response)
    background_tasks.add_task(
        perform_research_task,
        task_id,
        query_obj.query,
        query_obj.rag_mode,
        query_obj.orchestrator,
        request.app.state,
    )
    return response


@router.post("/sync", response_model=ResearchResponse)
async def execute_research_sync(
    query_obj: ResearchQuery, request: Request
) -> ResearchResponse:
    """
    Synchronously runs the multi-agent research workflow.

    Executes in a bounded worker pool so concurrent requests are processed
    in parallel without oversubscribing the host.
    """
    task_id = str(uuid.uuid4())
    start_t = time.perf_counter()
    try:
        state = await anyio.to_thread.run_sync(
            _execute_pipeline,
            query_obj.query,
            query_obj.rag_mode,
            query_obj.orchestrator,
            request.app.state,
            limiter=_get_limiter(),
        )
        elapsed = time.perf_counter() - start_t
        return _state_to_response(
            task_id, query_obj.query, query_obj.orchestrator, state, elapsed
        )
    except Exception as exc:  # noqa: BLE001 - surface as failed response
        logger.error("Sync research error: %s", exc, exc_info=True)
        return ResearchResponse(
            task_id=task_id,
            status="failed",
            query=query_obj.query,
            report=f"Execution error: {exc}",
            processing_time_seconds=round(time.perf_counter() - start_t, 2),
            created_at=datetime.now(timezone.utc),
        )


def _summary_to_response(summary: Any) -> BenchmarkResponse:
    """Convert a BenchmarkSummary into the API response model."""
    return BenchmarkResponse(
        total_test_cases=summary.total_test_cases,
        passed_test_cases=summary.passed_test_cases,
        pass_rate_percentage=summary.pass_rate_percentage,
        average_accuracy_percentage=summary.average_accuracy_percentage,
        target_accuracy_percentage=summary.target_accuracy_percentage,
        average_latency_seconds=summary.average_latency_seconds,
        p95_latency_seconds=summary.p95_latency_seconds,
        target_latency_seconds=summary.target_latency_seconds,
        average_synthesis_speedup_percentage=(
            summary.average_synthesis_speedup_percentage
        ),
        target_synthesis_speedup_percentage=(
            summary.target_synthesis_speedup_percentage
        ),
        categories_evaluated=summary.categories_evaluated,
        llm_mode=summary.llm_mode,
        measurement_note=summary.measurement_note,
        sample_results=[r.model_dump() for r in summary.sample_results],
    )


@router.get("/benchmark", response_model=BenchmarkResponse)
async def get_benchmark_report(
    max_cases: int = Query(default=10, ge=1, le=300),
) -> BenchmarkResponse:
    """
    Runs the measured benchmark on a subset of cases and returns results.

    Note: each case executes the real pipeline, so large max_cases values
    take proportionally longer.
    """
    from app.evaluation.benchmark_runner import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator()
    summary = await anyio.to_thread.run_sync(
        lambda: evaluator.run_benchmark(max_cases=max_cases)
    )
    return _summary_to_response(summary)


@router.post("/benchmark/run", response_model=BenchmarkResponse)
async def run_benchmark_suite(
    request_body: Optional[BenchmarkRunRequest] = None,
) -> BenchmarkResponse:
    """Executes the measured benchmark suite (optionally capped)."""
    from app.evaluation.benchmark_runner import BenchmarkEvaluator

    max_cases = request_body.max_cases if request_body else None
    evaluator = BenchmarkEvaluator()
    summary = await anyio.to_thread.run_sync(
        lambda: evaluator.run_benchmark(max_cases=max_cases)
    )
    return _summary_to_response(summary)


@router.get("/{task_id}", response_model=ResearchResponse)
async def get_research_task(task_id: str) -> ResearchResponse:
    """Retrieves status and report of a background research task."""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
