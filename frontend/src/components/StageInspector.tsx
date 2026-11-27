import React, { useState } from 'react';
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  Clock,
  Code2,
  Layers,
  Minus,
  Sparkles,
  X,
} from 'lucide-react';
import { RunRecord, StageState } from '../types';

interface StageInspectorProps {
  stageId: string | null;
  activeRun: RunRecord | null;
  onClose: () => void;
}

const STAGE_DESCRIPTIONS: Record<string, { title: string; module: string; summary: string }> = {
  file_upload: {
    title: 'File Discovery & Upload',
    module: 'src/ingestion/indexer.py',
    summary: 'Scans ./data and newly uploaded multipart files matching SUPPORTED_EXTENSIONS.',
  },
  file_validation: {
    title: 'Header & Format Validation',
    module: 'src/ingestion/parsers.py',
    summary: 'Verifies non-empty payloads and inspects PNG/JPEG/WEBP magic headers via Pillow.',
  },
  document_parser: {
    title: 'Document & Tabular Parser',
    module: 'src/ingestion/parsers.py',
    summary: 'Extracts page-tracked PDF text (PyMuPDF), heading-aware Markdown/TXT, and header-annotated CSV/Excel rows.',
  },
  ocr_extraction: {
    title: 'Spatial Image OCR',
    module: 'src/ingestion/parsers.py',
    summary: 'Runs RapidOCR ONNX on CPU and reconstructs 2D rows & columns from bounding boxes (skipped if no images).',
  },
  recursive_chunking: {
    title: 'Recursive Text Chunker',
    module: 'src/ingestion/chunker.py',
    summary: 'Splits sections hierarchically with word-aligned trailing overlap and deterministic SHA-1 chunk IDs.',
  },
  embedding_generation: {
    title: 'Dense Vector Generation',
    module: 'src/ingestion/indexer.py',
    summary: 'Encodes chunks into normalized 384-d float32 vectors locally on CPU via BAAI/bge-small-en-v1.5.',
  },
  vector_db_indexing: {
    title: 'ChromaDB HNSW Indexing',
    module: 'src/ingestion/indexer.py',
    summary: 'Upserts vectors, text, and provenance metadata into ./chroma_db persistent HNSW cosine store.',
  },
  bm25_indexing: {
    title: 'BM25 Lexical Corpus Sync',
    module: 'src/ingestion/indexer.py',
    summary: 'Tokenizes hyphenated company codes + sub-parts and syncs bm25_corpus.json 1:1 with ChromaDB.',
  },
  query_input: {
    title: 'Query Input & Config',
    module: 'src/retrieval/hybrid_retriever.py',
    summary: 'Validates query string, retrieval K parameters, and active ChromaDB collection document count.',
  },
  tabular_sql: {
    title: 'DuckDB Tabular SQL Router',
    module: 'src/retrieval/tabular_sql.py',
    summary: 'Routes CSV/Excel counting & aggregation questions to an in-memory read-only DuckDB SQL query.',
  },
  query_embedding: {
    title: 'Asymmetric Query Embedding',
    module: 'src/retrieval/hybrid_retriever.py',
    summary: 'Prepends BGE retrieval instruction prefix and encodes a normalized 384-d query vector.',
  },
  dense_retrieval: {
    title: 'ChromaDB Dense Search',
    module: 'src/retrieval/hybrid_retriever.py',
    summary: 'Retrieves Top-K semantic candidates ranked by cosine similarity (1 - cosine_distance).',
  },
  bm25_retrieval: {
    title: 'BM25Okapi Keyword Search',
    module: 'src/retrieval/hybrid_retriever.py',
    summary: 'Scores exact company codes, IDs, and rare terms across bm25_corpus.json.',
  },
  rrf_fusion: {
    title: 'Reciprocal Rank Fusion (RRF)',
    module: 'src/retrieval/hybrid_retriever.py',
    summary: 'Merges Dense and BM25 rank lists via scale-free RRF: sum(1 / (k + rank)).',
  },
  cross_encoder_rerank: {
    title: 'FlashRank Cross-Encoder Reranking',
    module: 'src/retrieval/hybrid_retriever.py',
    summary: 'Scores [Query + Passage] pairs jointly via quantized ms-marco-MiniLM-L-12-v2 ONNX on CPU.',
  },
  context_assembly: {
    title: 'Escaped XML Context Assembly',
    module: 'src/generation/generator.py',
    summary: 'Wraps Top-N reranked chunks in escaped <chunk> XML boundaries with untrusted-data security rules.',
  },
  llm_generation: {
    title: 'Groq Structured LLM Generation',
    module: 'src/generation/generator.py',
    summary: 'Executes grammar-constrained JSON Schema generation (temp=0.0) with automatic retry recovery.',
  },
  citation_verification: {
    title: 'Pydantic & Quote Substring Verifier',
    module: 'src/generation/generator.py',
    summary: 'Validates RAGResponse schema and verifies every cited quote exists verbatim in its source chunk.',
  },
};

function KV({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-800/90 bg-zinc-50/70 dark:bg-zinc-900/60 px-2.5 py-2">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">{label}</div>
      <div className="mt-0.5 font-mono text-xs font-medium text-zinc-900 dark:text-zinc-100 break-words">
        {value ?? '—'}
      </div>
    </div>
  );
}

export const StageInspector: React.FC<StageInspectorProps> = ({ stageId, activeRun, onClose }) => {
  const [tab, setTab] = useState<'formatted' | 'raw'>('formatted');

  if (!stageId) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6 text-center text-zinc-500 dark:text-zinc-400">
        <Layers className="h-8 w-8 mb-2 opacity-50" />
        <p className="text-xs font-medium text-zinc-700 dark:text-zinc-300">Stage Inspector</p>
        <p className="mt-1 text-xs leading-relaxed">
          Click any node on the pipeline canvas to inspect its live execution metadata, intermediate chunks, scores, and timings.
        </p>
      </div>
    );
  }

  const info = STAGE_DESCRIPTIONS[stageId] || {
    title: stageId,
    module: 'src/',
    summary: '',
  };
  const stageState: StageState | undefined = activeRun?.stages?.[stageId];
  const meta = stageState?.metadata || {};

  return (
    <div className="flex h-full flex-col bg-white dark:bg-[#101014]">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 border-b border-zinc-200 dark:border-zinc-800 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">{info.title}</h3>
          </div>
          <div className="mt-0.5 font-mono text-[11px] text-indigo-600 dark:text-indigo-400">{info.module}</div>
        </div>
        <button
          onClick={onClose}
          className="rounded-md p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Mode Switcher */}
      <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800 px-4 py-2 text-xs">
        <div className="flex gap-1 rounded-lg bg-zinc-100 dark:bg-zinc-900 p-0.5 border border-zinc-200 dark:border-zinc-800">
          <button
            onClick={() => setTab('formatted')}
            className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition ${
              tab === 'formatted'
                ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                : 'text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-300'
            }`}
          >
            Contextual Details
          </button>
          <button
            onClick={() => setTab('raw')}
            className={`flex items-center gap-1 rounded-md px-2.5 py-1 text-[11px] font-medium transition ${
              tab === 'raw'
                ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                : 'text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-300'
            }`}
          >
            <Code2 className="h-3 w-3" /> Raw JSON
          </button>
        </div>

        {stageState?.duration_ms !== null && stageState?.duration_ms !== undefined && (
          <span className="flex items-center gap-1 font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
            <Clock className="h-3 w-3" />
            {stageState.duration_ms.toFixed(1)} ms
          </span>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
        <p className="text-zinc-600 dark:text-zinc-400 leading-relaxed">{info.summary}</p>

        {!stageState ? (
          <div className="rounded-lg border border-dashed border-zinc-300 dark:border-zinc-800 p-4 text-center text-zinc-500">
            No active telemetry for this stage in the selected run. Trigger an{' '}
            <span className="font-semibold text-zinc-700 dark:text-zinc-300">
              {stageId.startsWith('file_') || stageId.includes('indexing') || stageId.includes('parser') || stageId.includes('ocr') || stageId.includes('chunking') || stageId === 'embedding_generation'
                ? 'Ingestion Run'
                : 'Query Run'}
            </span>{' '}
            to populate live execution metrics.
          </div>
        ) : tab === 'raw' ? (
          <pre className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] text-zinc-200">
            {JSON.stringify(stageState, null, 2)}
          </pre>
        ) : (
          <>
            {/* Status Banner */}
            <div className="rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/70 p-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] uppercase font-semibold text-zinc-700 dark:text-zinc-300">
                  Status: {stageState.status}
                </span>
                {stageState.started_at && (
                  <span className="font-mono text-[10px] text-zinc-500">
                    {new Date(stageState.started_at).toLocaleTimeString()}
                  </span>
                )}
              </div>
              <div className="mt-1 text-zinc-700 dark:text-zinc-300">{stageState.message}</div>
            </div>

            {/* Stage-Specific Rich Views */}
            {stageId === 'tabular_sql' && (
              <div className="space-y-2.5">
                <div className="grid grid-cols-2 gap-2">
                  <KV label="Engine" value={meta.engine || 'DuckDB In-Memory'} />
                  <KV label="Rows Returned" value={meta.row_count ?? '—'} />
                </div>
                {meta.sql_query && (
                  <>
                    <div className="font-semibold text-zinc-800 dark:text-zinc-200">Executed SQL</div>
                    <pre className="whitespace-pre-wrap rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-950 p-2.5 font-mono text-[10px] text-emerald-300">
                      {meta.sql_query}
                    </pre>
                  </>
                )}
                {Array.isArray(meta.rows) && (
                  <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-950 p-2.5 font-mono text-[10px] text-zinc-300">
                    {JSON.stringify(meta.rows, null, 2)}
                  </pre>
                )}
              </div>
            )}

            {stageId === 'document_parser' && Array.isArray(meta.files_parsed) && (
              <div className="space-y-2">
                <div className="font-semibold text-zinc-800 dark:text-zinc-200">Parsed Source Files</div>
                {meta.files_parsed.map((f: any, i: number) => (
                  <div key={i} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-2.5 space-y-1.5">
                    <div className="flex items-center justify-between font-mono text-[11px]">
                      <span className="font-semibold text-zinc-900 dark:text-zinc-100">{f.source_file}</span>
                      <span className="text-emerald-600 dark:text-emerald-400">{f.sections_extracted} sec · {f.duration_ms}ms</span>
                    </div>
                    <div className="text-[11px] text-zinc-500">Engine: {f.parser}</div>
                    {f.preview && (
                      <pre className="whitespace-pre-wrap rounded bg-zinc-100 dark:bg-zinc-950 p-2 font-mono text-[10px] text-zinc-700 dark:text-zinc-300">
                        {f.preview}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            )}

            {stageId === 'ocr_extraction' && (
              <div className="space-y-2">
                <KV label="OCR Engine" value={meta.engine || 'RapidOCR ONNX'} />
                {Array.isArray(meta.images_processed) &&
                  meta.images_processed.map((img: any, i: number) => (
                    <div key={i} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-2.5 space-y-1">
                      <div className="flex justify-between font-mono text-[11px]">
                        <span className="font-semibold">{img.source_file}</span>
                        <span>{img.duration_ms}ms</span>
                      </div>
                      <div className="text-[10px] text-zinc-500">{img.location_info}</div>
                      <pre className="whitespace-pre-wrap rounded bg-zinc-100 dark:bg-zinc-950 p-2 font-mono text-[10px]">
                        {img.extracted_preview}
                      </pre>
                    </div>
                  ))}
              </div>
            )}

            {stageId === 'recursive_chunking' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <KV label="Chunk Size" value={`${meta.chunk_size} chars`} />
                  <KV label="Overlap" value={`${meta.chunk_overlap} chars`} />
                  <KV label="Total Chunks" value={meta.total_chunks} />
                  <KV label="Avg / Max Len" value={`${meta.avg_chars} / ${meta.max_chars} c`} />
                </div>
                {Array.isArray(meta.chunk_previews) && (
                  <div className="space-y-2">
                    <div className="font-semibold text-zinc-800 dark:text-zinc-200">
                      Generated Chunk Previews ({meta.chunk_previews.length})
                    </div>
                    {meta.chunk_previews.slice(0, 6).map((ch: any) => (
                      <div key={ch.chunk_id} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-2.5 space-y-1">
                        <div className="flex items-center justify-between font-mono text-[10px] text-indigo-600 dark:text-indigo-400">
                          <span className="truncate">{ch.source_file}</span>
                          <span>{ch.char_len}c</span>
                        </div>
                        <div className="text-[10px] text-zinc-500">{ch.page_or_section} · type={ch.file_type}</div>
                        <pre className="whitespace-pre-wrap rounded bg-zinc-100 dark:bg-zinc-950 p-2 font-mono text-[10px] text-zinc-700 dark:text-zinc-300">
                          {ch.text}
                        </pre>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {(stageId === 'embedding_generation' || stageId === 'query_embedding') && (
              <div className="grid grid-cols-2 gap-2">
                <KV label="Model" value={meta.model} />
                <KV label="Dimensions" value={meta.dimensions || 384} />
                {meta.vectors_generated !== undefined && <KV label="Vectors Generated" value={meta.vectors_generated} />}
                {meta.query_prefix && (
                  <div className="col-span-2">
                    <KV label="BGE Asymmetric Query Prefix" value={meta.query_prefix} />
                  </div>
                )}
                {Array.isArray(meta.vector_preview_first_8) && (
                  <div className="col-span-2">
                    <KV label="First 8 Vector Dimensions" value={`[${meta.vector_preview_first_8.join(', ')}, ...]`} />
                  </div>
                )}
              </div>
            )}

            {(stageId === 'dense_retrieval' || stageId === 'bm25_retrieval') && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <KV label="Technique" value={meta.technique || stageId} />
                  <KV label="Retrieved Count" value={meta.retrieved_count} />
                </div>
                {Array.isArray(meta.query_tokens) && (
                  <KV label="Tokenized Query Terms" value={meta.query_tokens.join(', ')} />
                )}
                {Array.isArray(meta.hits) &&
                  meta.hits.slice(0, 8).map((h: any) => (
                    <div key={h.chunk_id} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-2.5 space-y-1">
                      <div className="flex items-center justify-between font-mono text-[11px]">
                        <span className="font-semibold">#{h.rank} · {h.source_file}</span>
                        <span className="text-emerald-600 dark:text-emerald-400">
                          {h.cosine_score !== undefined ? `cos=${h.cosine_score}` : `bm25=${h.bm25_score}`}
                        </span>
                      </div>
                      <div className="text-[10px] text-zinc-500">{h.page_or_section}</div>
                      <p className="font-mono text-[10px] text-zinc-600 dark:text-zinc-300 line-clamp-3">{h.preview}</p>
                    </div>
                  ))}
              </div>
            )}

            {stageId === 'rrf_fusion' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <KV label="Dense / BM25 In" value={`${meta.dense_input_count} / ${meta.bm25_input_count}`} />
                  <KV label="Unique Fused" value={`${meta.unique_fused_count} (${meta.dual_retriever_overlap_count} overlap)`} />
                </div>
                {Array.isArray(meta.candidates) &&
                  meta.candidates.slice(0, 8).map((c: any) => (
                    <div key={c.chunk_id} className="flex items-center justify-between rounded-lg border border-zinc-200 dark:border-zinc-800 px-2.5 py-2 font-mono text-[11px]">
                      <div className="min-w-0 pr-2">
                        <div className="truncate font-semibold">RRF #{c.rrf_rank} · {c.source_file}</div>
                        <div className="text-[10px] text-zinc-500">
                          Dense: {c.dense_rank ?? '—'} | BM25: {c.bm25_rank ?? '—'}
                        </div>
                      </div>
                      <span className="shrink-0 text-indigo-600 dark:text-indigo-400">{c.rrf_score}</span>
                    </div>
                  ))}
              </div>
            )}

            {stageId === 'cross_encoder_rerank' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <KV label="Reranker Model" value={meta.model} />
                  <KV label="Funnel" value={`${meta.input_count} in → Top ${meta.output_count}`} />
                </div>
                <div className="font-semibold text-zinc-800 dark:text-zinc-200">
                  Before vs. After Reranking Comparison
                </div>
                {Array.isArray(meta.comparisons) &&
                  meta.comparisons.slice(0, 8).map((c: any) => (
                    <div
                      key={c.chunk_id}
                      className={`rounded-lg border p-2.5 space-y-1.5 ${
                        c.kept_in_top_n
                          ? 'border-emerald-500/35 bg-emerald-500/5'
                          : 'border-zinc-200 dark:border-zinc-800 opacity-65'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 font-mono text-[11px]">
                        <span className="font-semibold truncate">
                          #{c.final_rank} · {c.source_file}
                        </span>
                        <div className="flex items-center gap-1.5 shrink-0">
                          {c.rank_delta > 0 && (
                            <span className="inline-flex items-center text-emerald-600 dark:text-emerald-400 text-[10px]">
                              <ArrowUp className="h-3 w-3" />+{c.rank_delta} (was #{c.rrf_rank_before})
                            </span>
                          )}
                          {c.rank_delta < 0 && (
                            <span className="inline-flex items-center text-amber-600 dark:text-amber-400 text-[10px]">
                              <ArrowDown className="h-3 w-3" />{c.rank_delta} (was #{c.rrf_rank_before})
                            </span>
                          )}
                          {c.rank_delta === 0 && (
                            <span className="inline-flex items-center text-zinc-500 text-[10px]">
                              <Minus className="h-3 w-3" /> #{c.rrf_rank_before}
                            </span>
                          )}
                          <span className="rounded bg-zinc-900 text-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5">
                            {Number(c.rerank_score).toFixed(4)}
                          </span>
                        </div>
                      </div>
                      <div className="text-[10px] text-zinc-500">{c.page_or_section}</div>
                      <pre className="whitespace-pre-wrap rounded bg-white/80 dark:bg-zinc-950 p-2 font-mono text-[10px] max-h-28 overflow-y-auto">
                        {c.text}
                      </pre>
                    </div>
                  ))}
              </div>
            )}

            {stageId === 'context_assembly' && (
              <div className="space-y-2.5">
                <div className="grid grid-cols-2 gap-2">
                  <KV label="Chunks Injected" value={meta.chunks_injected} />
                  <KV label="XML Chars" value={meta.xml_context_chars} />
                </div>
                <div className="font-semibold text-zinc-800 dark:text-zinc-200">Assembled XML Context</div>
                <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-950 p-2.5 font-mono text-[10px] text-zinc-300">
                  {meta.assembled_xml}
                </pre>
              </div>
            )}

            {stageId === 'llm_generation' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <KV label="Model" value={meta.model} />
                  <KV label="Schema Mode" value={meta.response_format_mode} />
                  <KV label="Prompt Tokens" value={meta.token_usage?.prompt_tokens ?? '—'} />
                  <KV label="Completion Tokens" value={meta.token_usage?.completion_tokens ?? '—'} />
                  <KV label="Total Tokens" value={meta.token_usage?.total_tokens ?? '—'} />
                  <KV label="Attempts" value={meta.attempts} />
                </div>
                {meta.raw_json_output && (
                  <>
                    <div className="font-semibold text-zinc-800 dark:text-zinc-200">Raw Structured LLM Output</div>
                    <pre className="max-h-60 overflow-y-auto whitespace-pre-wrap rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-950 p-2.5 font-mono text-[10px] text-emerald-300">
                      {meta.raw_json_output}
                    </pre>
                  </>
                )}
              </div>
            )}

            {stageId === 'citation_verification' && meta.final_response && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <KV
                    label="Grounding Verdict"
                    value={meta.is_supported ? 'SUPPORTED' : 'ABSTAINED'}
                  />
                  <KV label="Confidence" value={String(meta.confidence).toUpperCase()} />
                  <KV label="Verified Quotes" value={meta.verified_citations_count} />
                  <KV label="Stripped Quotes" value={meta.stripped_citations?.length ?? 0} />
                </div>
                <div className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3 space-y-2">
                  <div className="flex items-center gap-1.5 font-semibold text-zinc-900 dark:text-zinc-100">
                    {meta.is_supported ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-amber-500" />
                    )}
                    Final Grounded Answer
                  </div>
                  <p className="text-zinc-700 dark:text-zinc-300 leading-relaxed">
                    {meta.final_response.answer}
                  </p>
                </div>
              </div>
            )}

            {/* Generic fallback KV grid for remaining stages (file_upload, file_validation, vector_db_indexing, bm25_indexing, query_input) */}
            {['file_upload', 'file_validation', 'vector_db_indexing', 'bm25_indexing', 'query_input'].includes(
              stageId
            ) && (
              <div className="grid grid-cols-1 gap-2">
                {Object.entries(meta).map(([k, v]) => (
                  <KV
                    key={k}
                    label={k.replace(/_/g, ' ')}
                    value={typeof v === 'object' ? JSON.stringify(v, null, 2) : String(v)}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
