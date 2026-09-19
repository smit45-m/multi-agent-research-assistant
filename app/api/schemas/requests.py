"""
Pydantic v2 request models for schema validation.
Strict types, field constraints, validation metadata, and examples.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ResearchQuery(BaseModel):
    """Request model for initiating a multi-agent research workflow."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="The research question, technical topic, or hypothesis.",
    )
    depth: Literal["quick", "standard", "deep"] = Field(
        "standard",
        description="Depth of research synthesis and multi-step decomposition.",
    )
    rag_mode: Literal[
        "hybrid",
        "agentic",
        "vectorless",
        "hierarchical",
        "vector",
        "bm25",
        "rrf",
        "multi_query",
    ] = Field(
        "hybrid",
        description=(
            "RAG retrieval strategy: Hybrid (Dense+BM25+RRF), Agentic "
            "(CRAG/Self-RAG), Vectorless (Graph+BM25), Hierarchical, or "
            "Multi-Query."
        ),
    )
    orchestrator: Literal["langgraph", "crewai"] = Field(
        "langgraph",
        description=(
            "Multi-agent execution engine: LangGraph state graph or CrewAI process."
        ),
    )
    max_sources: int = Field(
        10, ge=1, le=50, description="Maximum number of context sources to retrieve."
    )
    source_filters: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional source categories to restrict retrieval (from the "
            "supported formats)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": (
                    "What are the latest advancements in solid-state "
                    "batteries compared to lithium-ion?"
                ),
                "depth": "standard",
                "rag_mode": "hybrid",
                "orchestrator": "langgraph",
                "max_sources": 10,
                "source_filters": ["pdf", "arxiv", "web"],
            }
        }
    }


class DocumentUploadMetadata(BaseModel):
    """Metadata for uploaded documents."""

    tags: List[str] = Field(
        default_factory=list, description="Tags associated with the document."
    )
    description: Optional[str] = Field(
        default=None, description="Brief description of the document contents."
    )


class BenchmarkRunRequest(BaseModel):
    """Request model for launching the 200+ test cases benchmark suite."""

    max_cases: Optional[int] = Field(
        default=None,
        ge=1,
        le=300,
        description="Optional cap on test cases to evaluate.",
    )
    category: Optional[str] = Field(
        default=None, description="Optional filter for specific category."
    )
