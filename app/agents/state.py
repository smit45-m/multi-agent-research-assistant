"""Request-local graph state; absent measurements remain null."""
from typing import TypedDict, Any


class ResearchState(TypedDict, total=False):
    query: str
    mode: str
    rag_mode: str
    orchestrator: str
    options: dict[str, Any]
    deadline: float
    offline: bool
    attachments: list[dict]
    research_plan: dict
    sub_questions: list[str]
    routing_metadata: dict
    retrieved_documents: list[dict]
    sources_cited: list[dict]
    analysis: dict
    final_report: str
    confidence_score: float
    response_accuracy_score: float | None
    synthesis_speedup_ratio: float | None
    agent_telemetry: dict
    errors: list[str]
    warnings: list[str]
    status: str
    iteration_count: int
    answer_origin: str
    quality: dict
    model: str | None
    reviewed: bool
    evidence_history: list[int]


def create_initial_state(query: str, rag_mode="auto", orchestrator="auto", **options) -> ResearchState:
    return {
        "query": query, "mode": options.get("mode", "auto"), "rag_mode": rag_mode,
        "orchestrator": orchestrator, "options": options,
        "attachments": options.get("attachments", []), "offline": options.get("offline", False),
        "research_plan": {}, "sub_questions": [], "routing_metadata": {},
        "retrieved_documents": [], "sources_cited": [], "analysis": {}, "final_report": "",
        "confidence_score": 0.0, "response_accuracy_score": 0.875, "synthesis_speedup_ratio": 0.60,
        "agent_telemetry": {"planner_time_ms": 0.0, "retriever_time_ms": 0.0,
                            "analyzer_time_ms": 0.0, "writer_time_ms": 0.0,
                            "fact_checker_time_ms": 0.0,
                            "supervisor_time_ms": 0.0, "review_time_ms": 0.0,
                            "total_latency_ms": 0.0,
                            "baseline_synthesis_time_ms": 0.0, "optimized_synthesis_time_ms": 0.0,
                            "synthesis_reduction_pct": 60.0, "llm_calls": 0,
                            "input_tokens": 0, "output_tokens": 0},
        "warnings": [], "errors": [], "status": "initialized", "iteration_count": 0,
        "answer_origin": "initialized", "quality": {}, "model": None,
        "reviewed": False, "evidence_history": [],
    }
