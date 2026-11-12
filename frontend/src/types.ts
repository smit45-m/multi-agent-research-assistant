export interface SourceInfo {
  title: string;
  url_or_path: string;
  relevance_score: number;
  source_type: string;
  snippet?: string;
}

export interface AgentTelemetryInfo {
  planner_time_ms?: number;
  retriever_time_ms?: number;
  analyzer_time_ms?: number;
  writer_time_ms?: number;
  fact_checker_time_ms?: number;
  supervisor_time_ms?: number;
  review_time_ms?: number;
  total_latency_ms?: number;
  baseline_synthesis_time_ms?: number;
  optimized_synthesis_time_ms?: number;
  synthesis_reduction_pct?: number;
  [key: string]: any;
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
  mode?: string;
  rag_mode?: string;
  model?: string;
  answer_origin?: string;
  telemetry?: AgentTelemetryInfo;
  thread_id?: string;
  chat_id?: string;
  memory_context_used?: {
    short_term_count?: number;
    long_term_count?: number;
    short_term_queries?: string[];
    long_term_titles?: string[];
    [key: string]: unknown;
  };
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

export interface DocumentQaLink {
  chat_id: string;
  thread_id: string;
  query: string;
  answer_summary?: string;
  created_at: string;
}

export interface DocumentResponse {
  document_id: string;
  filename: string;
  format: string;
  chunk_count: number;
  size_bytes?: number;
  qa_count?: number;
  qa_links?: DocumentQaLink[];
  status: string;
  uploaded_at: string;
}

export interface DocumentListResponse {
  documents: DocumentResponse[];
  total_count: number;
}

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'user' | 'admin';
  db_name?: string;
  is_active?: boolean;
  created_at: string;
  last_login?: string | null;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface SessionItem {
  jti: string;
  user_agent: string;
  created_at: string;
  expires_at: string;
  revoked: boolean;
  is_current: boolean;
}

export interface WorkspaceStats {
  db_name: string;
  backend: string;
  total_chats: number;
  total_threads: number;
  uploaded_files: number;
  indexed_chunks: number;
  short_term_memories: number;
  long_term_memories: number;
}

export interface ChatTurnItem {
  chat_id: string;
  thread_id?: string;
  query: string;
  report: string;
  report_summary?: string;
  sources: SourceInfo[];
  mode?: string;
  rag_mode?: string;
  orchestrator?: string;
  model?: string | null;
  answer_origin?: string;
  attachment_names?: string[];
  confidence_score?: number;
  response_accuracy_score?: number;
  processing_time_seconds?: number;
  memory_context_used?: {
    short_term_count?: number;
    long_term_count?: number;
    short_term_queries?: string[];
    long_term_titles?: string[];
    [key: string]: unknown;
  };
  created_at: string;
}

export interface ChatThreadItem {
  thread_id: string;
  title: string;
  latest_chat_id: string;
  latest_query: string;
  latest_summary: string;
  turn_count: number;
  mode: string;
  model?: string | null;
  file_names: string[];
  created_at: string;
  updated_at: string;
  turns: ChatTurnItem[];
}

export interface HistoryThreadsPayload {
  user_id: string;
  db_name: string;
  threads: ChatThreadItem[];
  total_threads: number;
  stats: WorkspaceStats;
}

export interface UserFilesHistoryPayload {
  user_id: string;
  db_name: string;
  files: DocumentResponse[];
  total_files: number;
}

export interface ShortTermMemoryItem {
  memory_id: string;
  user_id: string;
  thread_id: string;
  chat_id: string;
  query: string;
  answer_summary: string;
  keywords: string[];
  files_used: string[];
  created_at: string;
}

export interface LongTermMemoryItem {
  memory_id: string;
  user_id: string;
  topic_key: string;
  kind: string;
  title: string;
  summary: string;
  keywords: string[];
  source_files: string[];
  chat_ids: string[];
  access_count: number;
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserMemoryPayload {
  user_id: string;
  db_name: string;
  stats: WorkspaceStats;
  short_term: ShortTermMemoryItem[];
  long_term: LongTermMemoryItem[];
  short_term_count: number;
  long_term_count: number;
}
