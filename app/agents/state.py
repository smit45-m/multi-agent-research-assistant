"""
Shared state schema for LangGraph and CrewAI multi-agent orchestration.

All quality metrics in this state (confidence, accuracy, grounding, speedup)
are MEASURED at runtime — never hardcoded. They default to 0.0 and are only
populated by the agent that computes them.
"""

import operator
from typing import Annotated, Any, Dict, List, TypedDict


class ResearchState(TypedDict):
    """Represents the shared state across the 6 autonomous research agents."""

    query: str  # Original user research query
    rag_mode: str  # 'hybrid', 'vector', 'bm25', 'rrf', 'multi_query'
    orchestrator: str  # Execution framework: 'langgraph' or 'crewai'
    research_plan: Dict[str, Any]  # Planner's decomposed multi-step plan
    sub_questions: List[str]  # Decomposed sub-questions
    routing_metadata: Dict[str, Any]  # Routing decisions across multi-format sources
    retrieved_documents: List[
        Dict[str, Any]
    ]  # Retrieved context documents with metadata
    analysis: Dict[str, Any]  # Analyzer's synthesized insights
    final_report: str  # Writer's generated report
    sources_cited: List[Dict[str, Any]]  # Citations extracted from retrieval
    confidence_score: float  # Measured retrieval/analysis confidence (0-1)
    response_accuracy_score: float  # Measured grounding/verification score (0-1)
    synthesis_speedup_ratio: float  # MEASURED parallel-vs-sequential speedup (0-1)
    verification: Dict[str, Any]  # Verifier agent: per-claim grounding results
    critique: Dict[str, Any]  # Critic agent: quality review + revision notes
    revision_count: int  # Number of write->critique->write revisions
    agent_telemetry: Dict[str, Any]  # Measured execution timings per agent
    errors: Annotated[List[str], operator.add]  # Accumulated errors across workflow
    status: str  # Current execution status
    iteration_count: int  # Re-retrieval loop count


def create_initial_state(
    query: str,
    rag_mode: str = "hybrid",
    orchestrator: str = "langgraph",
) -> ResearchState:
    """
    Create the initial research state.

    All metric fields start at 0.0 and are populated only by real measurements.

    Args:
        query: The original user query.
        rag_mode: Retrieval mode ('hybrid', 'vector', 'bm25', 'rrf', 'multi_query').
        orchestrator: Multi-agent orchestrator ('langgraph' or 'crewai').

    Returns:
        Fully initialized state dictionary.
    """
    return {
        "query": query,
        "rag_mode": rag_mode,
        "orchestrator": orchestrator,
        "research_plan": {},
        "sub_questions": [],
        "routing_metadata": {
            "selected_sources": [],
            "source_formats_matched": [],
            "routing_strategy": "multi_format_hybrid",
        },
        "retrieved_documents": [],
        "analysis": {},
        "final_report": "",
        "sources_cited": [],
        "confidence_score": 0.0,
        "response_accuracy_score": 0.0,
        "synthesis_speedup_ratio": 0.0,
        "verification": {},
        "critique": {},
        "revision_count": 0,
        "agent_telemetry": {
            "planner_time_ms": 0.0,
            "retriever_time_ms": 0.0,
            "analyzer_time_ms": 0.0,
            "writer_time_ms": 0.0,
            "verifier_time_ms": 0.0,
            "critic_time_ms": 0.0,
            "total_latency_ms": 0.0,
            "sequential_retrieval_baseline_ms": 0.0,
            "parallel_retrieval_actual_ms": 0.0,
            "synthesis_reduction_pct": 0.0,
        },
        "errors": [],
        "status": "initialized",
        "iteration_count": 0,
    }
