"""Validated inputs for adaptive research and reproducible evaluation."""
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=12000)
    mode: Literal["auto", "fast", "balanced", "research", "privacy"] = "auto"
    depth: Optional[Literal["quick", "standard", "deep"]] = None
    rag_mode: Literal["auto", "hybrid", "agentic", "vectorless", "hierarchical", "vector", "bm25", "rrf", "multi_query"] = "auto"
    orchestrator: Literal["auto", "direct", "langgraph", "crewai"] = "auto"
    max_sources: int = Field(default=10, ge=1, le=30)
    source_filters: Optional[list[str]] = Field(default=None, max_length=20)
    web_search: bool = True
    strict_grounding: bool = False
    privacy_mode: bool = False
    attachment_ids: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Ask a question with at least three non-whitespace characters.")
        return value.strip()

    @field_validator("source_filters")
    @classmethod
    def clean_filters(cls, value):
        if value is None:
            return None
        allowed = {"pdf", "docx", "txt", "csv", "json", "html", "htm", "md", "xlsx", "tsv", "code", "configs", "web", "wikipedia", "arxiv", "image", "audio", "document"}
        cleaned = list(dict.fromkeys(x.lower().lstrip(".") for x in value))
        if not cleaned or any(x not in allowed for x in cleaned):
            raise ValueError("Specify supported source categories; use null for all sources.")
        return cleaned


class DocumentUploadMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list)
    description: Optional[str] = None


class BenchmarkRunRequest(BaseModel):
    max_cases: Optional[int] = Field(default=None, ge=1, le=300)
    category: Optional[str] = None
    evaluation_kind: Literal["offline_regression", "live_llm"] = "offline_regression"
    rag_mode: Literal["auto", "hybrid", "bm25", "vector", "multi_query", "hierarchical", "agentic", "vectorless"] = "auto"
    mode: Literal["auto", "fast", "balanced", "research"] = "auto"
