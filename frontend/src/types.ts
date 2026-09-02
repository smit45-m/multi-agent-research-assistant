export interface SourceInfo {
  title: string;
  url_or_path: string;
  relevance_score: number;
  source_type: string;
  snippet?: string;
}

export interface AgentTelemetryInfo {
  planner_time_ms: number;
  retriever_time_ms: number;
  analyzer_time_ms: number;
  writer_time_ms: number;
  total_latency_ms: number;
  baseline_synthesis_time_ms: number;
  optimized_synthesis_time_ms: number;
  synthesis_reduction_pct: number;
}

export interface ResearchResponse {
  task_id: string;
  status: string;
  query: string;
  report?: string;
  sources: SourceInfo[];
  confidence_score: number;
  response_accuracy_score: number;
  synthesis_speedup_ratio: number;
  processing_time_seconds: number;
  orchestrator: string;
  telemetry?: AgentTelemetryInfo;
  created_at: string;
}

export interface BenchmarkResult {
  test_id: string;
  category: string;
  query: string;
  target_source_format: string;
  accuracy_score: number;
  latency_seconds: number;
  synthesis_speedup: number;
  passed: boolean;
  details: string;
}

export interface BenchmarkResponse {
  total_test_cases: number;
  passed_test_cases: number;
  pass_rate_percentage: number;
  average_accuracy_percentage: number;
  target_accuracy_percentage: number;
  average_latency_seconds: number;
  target_latency_seconds: number;
  average_synthesis_speedup_percentage: number;
  target_synthesis_speedup_percentage: number;
  categories_evaluated: string[];
  sample_results: BenchmarkResult[];
}

export interface DocumentResponse {
  document_id: string;
  filename: string;
  format: string;
  chunk_count: number;
  status: string;
  uploaded_at: string;
}

export interface DocumentListResponse {
  documents: DocumentResponse[];
  total_count: number;
}
