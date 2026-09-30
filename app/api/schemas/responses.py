"""Response contracts distinguish evidence diagnostics from measured accuracy."""
from typing import Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    citation_id: int = 0
    title: str
    url_or_path: str
    relevance_score: float = 0.0
    source_type: str = "document"
    snippet: Optional[str] = None


class AgentTelemetryInfo(BaseModel):
    planner_time_ms: float = 0.0
    retriever_time_ms: float = 0.0
    analyzer_time_ms: float = 0.0
    writer_time_ms: float = 0.0
    fact_checker_time_ms: float = 0.0
    supervisor_time_ms: float = 0.0
    review_time_ms: float = 0.0
    total_latency_ms: float = 0.0
    baseline_synthesis_time_ms: Optional[float] = None
    optimized_synthesis_time_ms: Optional[float] = None
    synthesis_reduction_pct: Optional[float] = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_rounds: int = 0


class ResearchResponse(BaseModel):
    task_id: str
    status: str
    query: str
    report: Optional[str] = None
    sources: list[SourceInfo] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, description="Lexical evidence coverage; NOT factual accuracy.")
    response_accuracy_score: Optional[float] = None
    synthesis_speedup_ratio: Optional[float] = None
    processing_time_seconds: float = 0.0
    mode: str = "auto"
    rag_mode: str = "auto"
    orchestrator: str = "auto"
    model: Optional[str] = None
    answer_origin: str = "pending"
    warnings: list[str] = Field(default_factory=list)
    routing: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    research_plan: dict[str, Any] = Field(default_factory=dict)
    telemetry: Optional[AgentTelemetryInfo] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    format: str = "document"
    chunk_count: int
    status: str
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    vector_store_documents: int
    active_concurrent_capacity: int = 8
    latency_sla_seconds: float = 25.0
    accuracy_benchmark_target: float = 85.0
    llm_configured: bool = False
    embedding_status: str = "not_loaded"
    note: str = "Budgets and targets are configuration, not achieved benchmark results."


class BenchmarkResponse(BaseModel):
    total_test_cases: int
    passed_test_cases: int
    pass_rate_percentage: float
    average_accuracy_percentage: float
    target_accuracy_percentage: float = 85.0
    average_latency_seconds: float
    target_latency_seconds: float = 25.0
    average_synthesis_speedup_percentage: Optional[float] = None
    target_synthesis_speedup_percentage: Optional[float] = None
    categories_evaluated: list[str]
    sample_results: list[dict[str, Any]]
    evaluation_kind: str = "offline_regression"
    methodology: str = ""
    limitations: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    configurations: list[dict[str, Any]] = Field(default_factory=list)
    dataset_sha256: str = ""
    model: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
