"""
Multi-step routing workflows and prompt configuration.

Routing is implemented as fast deterministic rule-based classification
(domain keywords + complexity heuristics). It intentionally avoids an LLM
call: routing runs on every request and must stay in the microsecond range.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class RoutingDecision(BaseModel):
    """Structured decision produced by the multi-step routing workflow."""

    domain: str = Field(
        description=(
            "Detected domain: 'academic', 'biomedical', 'financial', "
            "'technical', 'news', 'general'"
        )
    )
    complexity: str = Field(
        description="Query complexity level: 'simple', 'moderate', 'complex'"
    )
    recommended_rag_mode: str = Field(
        description=(
            "Optimal RAG strategy: 'hybrid', 'rrf', 'multi_query', 'vector', 'bm25'"
        )
    )
    target_source_formats: List[str] = Field(
        description=(
            "List of target source formats to query from the supported sources"
        )
    )
    optimized_planner_prompt: str = Field(
        description="Optimized prompt for the Research Planner agent"
    )
    requires_multi_query_expansion: bool = Field(default=True)
    estimated_synthesis_time_s: float = Field(default=4.5)


class ResearchRouter:
    """
    Deterministic rule-based routing workflow.
    Classifies query domain and complexity, then selects target source
    formats and the recommended RAG retrieval strategy.
    """

    # Supported source categories
    AVAILABLE_SOURCES = [
        "pdf",
        "docx",
        "txt",
        "csv",
        "json",
        "html",
        "md",
        "xlsx",
        "arxiv",
        "wikipedia",
        "pubmed",
        "news",
        "web",
        "code",
        "configs",
    ]

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def route(
        self,
        query: str,
        depth: str = "standard",
        user_sources: Optional[List[str]] = None,
    ) -> RoutingDecision:
        """
        Executes the multi-step routing workflow:
        Step 1: Domain & Intent Classification
        Step 2: Source Format Mapping (across 15+ formats)
        Step 3: RAG Retrieval Strategy Optimization (Dense, Sparse, RRF, Multi-Query)
        Step 4: Prompt Optimization Calibration
        """
        q_lower = query.lower()

        # Step 1: Intent & Domain Classification
        if any(
            term in q_lower
            for term in [
                "paper",
                "study",
                "arxiv",
                "theorem",
                "quantum",
                "llm",
                "transformer",
                "neural",
            ]
        ):
            domain = "academic"
            target_sources = ["arxiv", "pdf", "wikipedia", "web"]
            rag_mode = "rrf"
        elif any(
            term in q_lower
            for term in [
                "clinical",
                "gene",
                "disease",
                "drug",
                "fda",
                "medical",
                "patient",
                "protein",
            ]
        ):
            domain = "biomedical"
            target_sources = ["pubmed", "arxiv", "pdf", "web"]
            rag_mode = "hybrid"
        elif any(
            term in q_lower
            for term in [
                "revenue",
                "stock",
                "margin",
                "ebitda",
                "inflation",
                "market",
                "q1",
                "q2",
                "q3",
                "q4",
                "finance",
                "gdp",
            ]
        ):
            domain = "financial"
            target_sources = ["csv", "xlsx", "json", "pdf", "web"]
            rag_mode = "hybrid"
        elif any(
            term in q_lower
            for term in [
                "code",
                "api",
                "function",
                "architecture",
                "docker",
                "fastapi",
                "python",
                "kubernetes",
                "bug",
            ]
        ):
            domain = "technical"
            target_sources = ["code", "configs", "md", "txt", "web"]
            rag_mode = "multi_query"
        elif any(
            term in q_lower
            for term in [
                "today",
                "latest",
                "recent",
                "2025",
                "2026",
                "news",
                "announcement",
            ]
        ):
            domain = "news"
            target_sources = ["news", "web", "html"]
            rag_mode = "hybrid"
        else:
            domain = "general"
            target_sources = ["pdf", "wikipedia", "web", "txt", "md"]
            rag_mode = "hybrid"

        # Apply user overrides if specified
        if user_sources:
            target_sources = [
                s for s in user_sources if s in self.AVAILABLE_SOURCES
            ] or target_sources

        # Step 2: Complexity Evaluation
        word_count = len(query.split())
        if word_count > 15 or depth == "deep":
            complexity = "complex"
            requires_multi_query = True
            est_time = 6.2
        elif word_count > 7 or depth == "standard":
            complexity = "moderate"
            requires_multi_query = rag_mode in ["rrf", "multi_query"]
            est_time = 4.5
        else:
            complexity = "simple"
            requires_multi_query = False
            est_time = 2.8

        # Step 3: Prompt Optimization Calibration
        optimized_prompt = (
            f"Role: Expert {domain.capitalize()} Research Planner. "
            f"Query Complexity: {complexity}. "
            f"Prioritize verifiable factual cross-referencing across "
            f"target source formats: {', '.join(target_sources)}. "
            "Enforce explicit atomic sub-questions, elimination of "
            "speculative claims, and strict attribution."
        )

        decision = RoutingDecision(
            domain=domain,
            complexity=complexity,
            recommended_rag_mode=rag_mode,
            target_source_formats=target_sources,
            optimized_planner_prompt=optimized_prompt,
            requires_multi_query_expansion=requires_multi_query,
            estimated_synthesis_time_s=est_time,
        )
        logger.info(
            f"Routing workflow completed: Domain={domain}, "
            f"RAG_Mode={rag_mode}, Sources={len(target_sources)}"
        )
        return decision
