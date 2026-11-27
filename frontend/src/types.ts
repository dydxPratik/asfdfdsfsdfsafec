export type StageStatus = 'idle' | 'waiting' | 'running' | 'completed' | 'failed' | 'skipped';
export type PipelineType = 'ingestion' | 'query';

export interface StageState {
  stage: string;
  label: string;
  group: string;
  status: StageStatus;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  message: string;
  metadata: Record<string, any>;
}

export interface PipelineEvent {
  event_id: string;
  run_id: string;
  stage: string;
  status: StageStatus;
  message: string;
  timestamp: string;
  duration_ms: number | null;
  metadata: Record<string, any>;
}

export interface Citation {
  source_file: string;
  quote: string;
}

export interface RAGResponse {
  answer: string;
  citations: Citation[];
  is_supported: boolean;
  confidence: 'high' | 'medium' | 'low';
}

export interface RetrievedChunk {
  chunk_id: string;
  text: string;
  source_file: string;
  page_or_section: string;
  dense_rank: number | null;
  dense_score: number | null;
  bm25_rank: number | null;
  bm25_score: number | null;
  rrf_score: number;
  rerank_score: number;
}

export interface RunRecord {
  run_id: string;
  pipeline_type: PipelineType;
  title: string;
  status: 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at: string | null;
  total_duration_ms: number | null;
  params: Record<string, any>;
  stages: Record<string, StageState>;
  events: PipelineEvent[];
  result: any;
  error: string | null;
}

export interface RunSummary {
  run_id: string;
  pipeline_type: PipelineType;
  title: string;
  status: 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at: string | null;
  total_duration_ms: number | null;
  params: Record<string, any>;
  error: string | null;
  result_preview: any;
}

export interface RuntimeSettings {
  default_groq_model: string;
  embedding_profile?: string;
  response_language?: string;
  dense_top_k: number;
  bm25_top_k: number;
  rrf_k: number;
  rerank_top_n: number;
  chunk_size: number;
  chunk_overlap: number;
}

export interface HealthResponse {
  status: string;
  environment: string;
  groq_api_configured: boolean;
  embedding_model: string;
  reranker_model: string;
  vector_collection: string;
  chroma_chunks: number;
  bm25_docs: number;
  source_files: number;
  supported_extensions: string[];
  available_groq_models: string[];
  embedding_profiles?: Record<string, { model_name: string; label: string }>;
  supported_response_languages?: string[];
  settings: RuntimeSettings;
}

export interface DocumentItem {
  name: string;
  extension: string;
  category: 'document' | 'spreadsheet' | 'image';
  size_bytes: number;
  modified_at: string;
  indexed_chunks: number;
}

export interface OverviewResponse {
  health: HealthResponse;
  documents_count: number;
  format_breakdown: Record<string, number>;
  telemetry: {
    total_runs: number;
    completed_query_runs: number;
    grounded_answers: number;
    strict_abstentions: number;
    avg_e2e_ms: number | null;
  };
  recent_runs: RunSummary[];
}
