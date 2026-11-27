import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Compass,
  FileText,
  Loader2,
  Maximize2,
  Minimize2,
  Moon,
  RefreshCw,
  Send,
  ShieldAlert,
  Sparkles,
  Sun,
  Trash2,
  Upload,
  Workflow,
} from 'lucide-react';
import {
  fetchDocuments,
  fetchHealth,
  startIngestionRun,
  startQueryRun,
  subscribeToRunStream,
} from './api';
import {
  DocumentItem,
  HealthResponse,
  PipelineEvent,
  PipelineType,
  RAGResponse,
  RetrievedChunk,
  RunRecord,
} from './types';
import { PipelineCanvas } from './components/PipelineCanvas';
import { StageInspector } from './components/StageInspector';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content?: string;
  response?: RAGResponse;
  chunks?: RetrievedChunk[];
  durationMs?: number | null;
  error?: string;
}

const SAMPLE_QUERIES = [
  'What is the lodging cap in Zurich under POL-EXP-402 and when must VaultSpend receipts be submitted?',
  'Who is employee NV-EMP-1001, what project do they lead, and how long is the Helios-9 warranty?',
  'What are the SEV-1 incident notification SLAs in SEC-RUN-880?',
  "What is NexaVault's cryptocurrency 401(k) match and stock ticker?",
];

export const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [showFilesList, setShowFilesList] = useState(false);

  // Sidebar RAG controls (matching original app.py)
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('openai/gpt-oss-120b');
  const [topN, setTopN] = useState<number>(5);
  const [showDebug, setShowDebug] = useState<boolean>(true);
  const [statusBanner, setStatusBanner] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Live Visualizer state
  const [activeRun, setActiveRun] = useState<RunRecord | null>(null);
  const [visualizerPipeline, setVisualizerPipeline] = useState<PipelineType>('query');
  const [visualizerExpanded, setVisualizerExpanded] = useState<boolean>(false);
  const [visualizerVisible, setVisualizerVisible] = useState<boolean>(true);
  const [selectedStageId, setSelectedStageId] = useState<string | null>(null);

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [promptInput, setPromptInput] = useState<string>('');
  const [openDebugIds, setOpenDebugIds] = useState<Record<string, boolean>>({});

  const unsubRef = useRef<(() => void) | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  const refreshStats = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([fetchHealth(), fetchDocuments()]);
      setHealth(h);
      setDocuments(d.documents);
      if (h.settings?.default_groq_model) {
        setSelectedModel((prev) => prev || h.settings.default_groq_model);
      }
    } catch {
      // ignore transient startup poll error
    }
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [messages]);

  useEffect(() => {
    return () => {
      unsubRef.current?.();
    };
  }, []);

  const isRunning = activeRun?.status === 'running';

  const attachStream = useCallback(
    (runId: string, onDone?: (finalRun: RunRecord) => void) => {
      unsubRef.current?.();
      unsubRef.current = subscribeToRunStream(runId, {
        onSnapshot: (snap) => {
          setActiveRun(snap);
        },
        onStageEvent: (ev: PipelineEvent) => {
          setActiveRun((prev) => {
            if (!prev || prev.run_id !== ev.run_id) return prev;
            const nextStages = { ...prev.stages };
            if (nextStages[ev.stage]) {
              nextStages[ev.stage] = {
                ...nextStages[ev.stage],
                status: ev.status,
                message: ev.message,
                duration_ms: ev.duration_ms ?? nextStages[ev.stage].duration_ms,
                metadata: { ...nextStages[ev.stage].metadata, ...ev.metadata },
              };
            }
            return {
              ...prev,
              stages: nextStages,
              events: [...prev.events, ev],
            };
          });
        },
        onComplete: (finalRun) => {
          setActiveRun(finalRun);
          refreshStats();
          onDone?.(finalRun);
        },
      });
    },
    [refreshStats]
  );

  const handleSaveAndReindex = async () => {
    if (isRunning) return;
    setStatusBanner(null);
    setSelectedStageId(null);
    setVisualizerVisible(true);
    setVisualizerPipeline('ingestion');

    try {
      const filesToSend = [...pendingFiles];
      const { run_id, run } = await startIngestionRun(filesToSend, true);
      setPendingFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      setActiveRun(run);

      attachStream(run_id, (finalRun) => {
        if (finalRun.status === 'completed') {
          const totalChunks = finalRun.result?.total_chunks_indexed ?? 0;
          setStatusBanner({
            type: 'success',
            text:
              filesToSend.length > 0
                ? `Saved ${filesToSend.length} file(s) and indexed ${totalChunks} chunks.`
                : `Re-indexed knowledge base (${totalChunks} chunks).`,
          });
        } else {
          setStatusBanner({
            type: 'error',
            text: finalRun.error || 'Ingestion failed.',
          });
        }
      });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err?.message || 'Failed to start indexing.' });
    }
  };

  const executeQuery = async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed || isRunning) return;

    setPromptInput('');
    setSelectedStageId(null);
    setVisualizerVisible(true);
    setVisualizerPipeline('query');

    const userMsgId = `u_${Date.now()}`;
    setMessages((prev) => [...prev, { id: userMsgId, role: 'user', content: trimmed }]);

    try {
      const { run_id, run } = await startQueryRun({
        query: trimmed,
        model: selectedModel,
        top_n: topN,
      });
      setActiveRun(run);

      attachStream(run_id, (finalRun) => {
        const asstId = `a_${Date.now()}`;
        if (finalRun.status === 'completed' && finalRun.result?.response) {
          setMessages((prev) => [
            ...prev,
            {
              id: asstId,
              role: 'assistant',
              response: finalRun.result.response,
              chunks: finalRun.result.chunks || [],
              durationMs: finalRun.total_duration_ms,
            },
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            {
              id: asstId,
              role: 'assistant',
              error: finalRun.error || 'RAG pipeline error while processing query.',
            },
          ]);
        }
      });
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: `a_${Date.now()}`,
          role: 'assistant',
          error: err?.message || 'Failed to execute query.',
        },
      ]);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executeQuery(promptInput);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-zinc-50 dark:bg-[#09090b] text-zinc-900 dark:text-zinc-100">
      {/* ---------------- LEFT SIDEBAR ---------------- */}
      <aside className="flex w-[260px] shrink-0 flex-col border-r border-zinc-200 dark:border-zinc-800/90 bg-white dark:bg-[#0d0d11]">
        {/* Brand Header */}
        <div className="flex h-11 items-center justify-between border-b border-zinc-200 dark:border-zinc-800/90 px-3.5">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-600 text-white shadow-sm">
              <Compass className="h-3.5 w-3.5" />
            </div>
            <div>
              <div className="text-xs font-bold tracking-tight">RAGQuest</div>
              <div className="font-mono text-[9px] text-zinc-500">Local Hybrid RAG</div>
            </div>
          </div>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="rounded-md border border-zinc-200 dark:border-zinc-800 p-1 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition"
            title="Toggle Dark / Light Theme"
          >
            {darkMode ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          {/* Company Knowledge Base */}
          <section className="space-y-3">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              Company Knowledge Base
            </h2>

            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/70 p-2.5">
                <div className="text-[11px] text-zinc-500">Source Files</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {health?.source_files ?? documents.length}
                </div>
              </div>
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/70 p-2.5">
                <div className="text-[11px] text-zinc-500">Indexed Chunks</div>
                <div className="mt-0.5 text-lg font-bold tabular-nums">
                  {health?.chroma_chunks ?? '—'}
                </div>
              </div>
            </div>

            {/* File Uploader */}
            <div>
              <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-zinc-300 dark:border-zinc-700 bg-zinc-50/60 dark:bg-zinc-900/40 px-3 py-3 text-center hover:border-indigo-500/60 transition">
                <Upload className="h-4 w-4 text-zinc-500 mb-1" />
                <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                  Upload Company Files
                </span>
                <span className="mt-0.5 text-[10px] text-zinc-500">
                  .pdf, .md, .txt, .csv, .xlsx, .png, .jpg, .webp
                </span>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.md,.txt,.csv,.xlsx,.xlsm,.png,.jpg,.jpeg,.webp"
                  onChange={(e) => {
                    if (e.target.files) {
                      setPendingFiles(Array.from(e.target.files));
                    }
                  }}
                  className="hidden"
                />
              </label>

              {pendingFiles.length > 0 && (
                <div className="mt-2 rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-1.5 text-[11px] text-indigo-600 dark:text-indigo-300">
                  {pendingFiles.length} file(s) staged: {pendingFiles.map((f) => f.name).join(', ')}
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={handleSaveAndReindex}
              disabled={isRunning}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 transition shadow-sm"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${
                  isRunning && activeRun?.pipeline_type === 'ingestion' ? 'animate-spin' : ''
                }`}
              />
              Save &amp; Re-Index Knowledge Base
            </button>

            {statusBanner && (
              <div
                className={`rounded-lg border px-2.5 py-2 text-[11px] ${
                  statusBanner.type === 'success'
                    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                    : 'border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-300'
                }`}
              >
                {statusBanner.text}
              </div>
            )}

            {/* Collapsible Indexed Files */}
            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
              <button
                type="button"
                onClick={() => setShowFilesList(!showFilesList)}
                className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-zinc-700 dark:text-zinc-300"
              >
                <span className="flex items-center gap-1.5">
                  <FileText className="h-3.5 w-3.5 text-zinc-400" />
                  Indexed Files in ./data ({documents.length})
                </span>
                {showFilesList ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
              </button>
              {showFilesList && (
                <div className="border-t border-zinc-200 dark:border-zinc-800 px-3 py-2 space-y-1 max-h-40 overflow-y-auto font-mono text-[11px] text-zinc-600 dark:text-zinc-400">
                  {documents.length === 0 ? (
                    <div>No files in ./data yet.</div>
                  ) : (
                    documents.map((doc) => (
                      <div key={doc.name} className="flex justify-between gap-2 truncate">
                        <span className="truncate">{doc.name}</span>
                        <span className="shrink-0 text-zinc-400">{(doc.size_bytes / 1024).toFixed(1)} KB</span>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </section>

          <hr className="border-zinc-200 dark:border-zinc-800" />

          {/* Retrieval & Generation Settings */}
          <section className="space-y-3.5">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              Retrieval &amp; Generation
            </h2>

            <div>
              <label className="block text-xs font-medium text-zinc-700 dark:text-zinc-300 mb-1">
                Groq Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full rounded-lg border border-zinc-300 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-900 dark:text-zinc-100 focus:border-indigo-500 focus:outline-none"
              >
                {(health?.available_groq_models || ['openai/gpt-oss-120b', 'llama-3.3-70b-versatile']).map(
                  (m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  )
                )}
              </select>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-medium text-zinc-700 dark:text-zinc-300">Top-N Reranked Chunks</span>
                <span className="font-mono font-semibold text-indigo-500">{topN}</span>
              </div>
              <input
                type="range"
                min={3}
                max={10}
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
                className="w-full accent-indigo-600"
              />
            </div>

            <label className="flex items-center justify-between cursor-pointer text-xs">
              <span className="font-medium text-zinc-700 dark:text-zinc-300">Sources &amp; Retrieval Debug</span>
              <input
                type="checkbox"
                checked={showDebug}
                onChange={(e) => setShowDebug(e.target.checked)}
                className="h-4 w-4 rounded accent-indigo-600"
              />
            </label>

            <button
              type="button"
              onClick={() => setMessages([])}
              className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-zinc-200 dark:border-zinc-800 px-3 py-2 text-xs font-medium text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-900 transition"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear Chat History
            </button>
          </section>
        </div>
      </aside>

      {/* ---------------- MAIN COLUMN: COMPACT LIVE VISUALIZER + SPACIOUS CHAT ---------------- */}
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        {/* Top Bar */}
        <header className="flex h-11 shrink-0 items-center justify-between border-b border-zinc-200 dark:border-zinc-800/90 bg-white dark:bg-[#0d0d11] px-4">
          <div className="flex items-baseline gap-2.5 min-w-0">
            <h1 className="truncate text-xs font-bold text-zinc-900 dark:text-zinc-100">
              RAGQuest — Local Hybrid Enterprise RAG
            </h1>
            <p className="hidden lg:block truncate text-[11px] text-zinc-500 dark:text-zinc-400">
              ChromaDB + bge-small + BM25 (RRF) + FlashRank → Grounded Groq
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setVisualizerVisible((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-medium transition ${
                visualizerVisible
                  ? 'border-indigo-500/40 bg-indigo-500/10 text-indigo-600 dark:text-indigo-300'
                  : 'border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400'
              }`}
            >
              <Workflow className="h-3.5 w-3.5" />
              {visualizerVisible ? 'Visualizer On' : 'Show Visualizer'}
            </button>
          </div>
        </header>

        {/* ---------------- BALANCED LIVE RAG PIPELINE VISUALIZER ---------------- */}
        {visualizerVisible && (
          <section
            className={`relative flex shrink-0 flex-col border-b border-zinc-200 dark:border-zinc-800 transition-all duration-200 ${
              visualizerExpanded ? 'h-[62vh]' : 'h-[230px]'
            }`}
          >
            {/* Slim Top Strip (does not overlap diagram nodes) */}
            <div className="flex h-8 shrink-0 items-center justify-between border-b border-zinc-200/70 dark:border-zinc-800/80 bg-zinc-100/70 dark:bg-[#111116] px-3 text-[11px]">
              <div className="flex items-center gap-2">
                <div className="flex rounded-md bg-zinc-200/80 dark:bg-zinc-800/90 p-0.5 text-[11px]">
                  <button
                    type="button"
                    onClick={() => {
                      setVisualizerPipeline('query');
                      setSelectedStageId(null);
                    }}
                    className={`rounded px-2 py-0.5 font-medium transition ${
                      visualizerPipeline === 'query'
                        ? 'bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 shadow-sm'
                        : 'text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-300'
                    }`}
                  >
                    Query Pipeline
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setVisualizerPipeline('ingestion');
                      setSelectedStageId(null);
                    }}
                    className={`rounded px-2 py-0.5 font-medium transition ${
                      visualizerPipeline === 'ingestion'
                        ? 'bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 shadow-sm'
                        : 'text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-300'
                    }`}
                  >
                    Ingestion Pipeline
                  </button>
                </div>

                {activeRun && activeRun.pipeline_type === visualizerPipeline && (
                  <span className="flex items-center gap-1.5 font-mono text-[11px] text-zinc-600 dark:text-zinc-300">
                    {activeRun.status === 'running' ? (
                      <>
                        <Loader2 className="h-3 w-3 animate-spin text-indigo-500" />
                        <span className="text-indigo-500 font-semibold">Running live…</span>
                      </>
                    ) : activeRun.status === 'completed' ? (
                      <>
                        <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                        <span>
                          Completed in{' '}
                          {activeRun.total_duration_ms
                            ? `${(activeRun.total_duration_ms / 1000).toFixed(2)}s`
                            : '—'}
                        </span>
                      </>
                    ) : (
                      <span className="text-rose-500">Failed</span>
                    )}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                <span className="hidden md:inline text-[10px] text-zinc-500">
                  Click any node for details
                </span>
                <button
                  type="button"
                  onClick={() => setVisualizerExpanded((v) => !v)}
                  className="inline-flex items-center gap-1 rounded border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-2 py-0.5 text-[10px] font-medium text-zinc-600 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800"
                  title={visualizerExpanded ? 'Shrink Visualizer' : 'Expand Visualizer'}
                >
                  {visualizerExpanded ? (
                    <>
                      <Minimize2 className="h-3 w-3" /> Compact
                    </>
                  ) : (
                    <>
                      <Maximize2 className="h-3 w-3" /> Expand
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="relative flex-1 min-h-0">
              <PipelineCanvas
                key={`${visualizerPipeline}-${visualizerExpanded ? 'exp' : 'cmp'}`}
                activeRun={activeRun}
                pipeline={visualizerPipeline}
                selectedStageId={selectedStageId}
                onSelectStage={(sid) =>
                  setSelectedStageId((prev) => (prev === sid ? null : sid))
                }
                darkMode={darkMode}
              />

              {/* Slide-Over Contextual Inspector ONLY when a node is clicked */}
              {selectedStageId && (
                <div className="absolute right-0 top-0 bottom-0 z-20 w-[360px] max-w-full border-l border-zinc-200 dark:border-zinc-800 shadow-2xl">
                  <StageInspector
                    stageId={selectedStageId}
                    activeRun={activeRun}
                    onClose={() => setSelectedStageId(null)}
                  />
                </div>
              )}
            </div>
          </section>
        )}

        {/* ---------------- CHAT CONVERSATION AREA ---------------- */}
        <main className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 ? (
            <div className="mx-auto max-w-4xl py-2">
              <div className="mb-2.5 flex items-center gap-1.5 text-xs font-medium text-zinc-500 dark:text-zinc-400">
                <Sparkles className="h-3.5 w-3.5 text-indigo-500" />
                Try a sample company query:
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                {SAMPLE_QUERIES.map((sq, idx) => (
                  <button
                    key={idx}
                    type="button"
                    disabled={isRunning}
                    onClick={() => executeQuery(sq)}
                    className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/70 p-3 text-left text-xs text-zinc-700 dark:text-zinc-300 hover:border-indigo-500/60 hover:bg-zinc-50 dark:hover:bg-zinc-900 disabled:opacity-50 transition shadow-sm"
                  >
                    {sq}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-4xl space-y-3">
              {messages.map((msg, mIdx) => (
                <div
                  key={msg.id}
                  ref={mIdx === messages.length - 1 ? chatEndRef : undefined}
                  className={`rounded-2xl border p-3.5 ${
                    msg.role === 'user'
                      ? 'ml-auto max-w-2xl border-indigo-500/25 bg-indigo-500/10 text-zinc-900 dark:text-zinc-100'
                      : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/80 shadow-sm'
                  }`}
                >
                  {msg.role === 'user' ? (
                    <p className="text-sm">{msg.content}</p>
                  ) : msg.error ? (
                    <p className="text-sm text-rose-500">{msg.error}</p>
                  ) : msg.response ? (
                    <div className="space-y-3">
                      {/* Grounding Badge */}
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        {msg.response.is_supported ? (
                          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 font-medium text-emerald-700 dark:text-emerald-300">
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            {msg.response.confidence === 'high'
                              ? 'Grounded in Company Documents · HIGH'
                              : 'Partially Grounded · MEDIUM'}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 font-medium text-amber-700 dark:text-amber-300">
                            <ShieldAlert className="h-3.5 w-3.5" />
                            Not Found in Company Documents (Strict Abstention) · LOW
                          </span>
                        )}
                        {msg.durationMs && (
                          <span className="font-mono text-[11px] text-zinc-400">
                            {(msg.durationMs / 1000).toFixed(2)}s
                          </span>
                        )}
                      </div>

                      {/* Answer */}
                      <div className="whitespace-pre-wrap text-sm leading-relaxed text-zinc-800 dark:text-zinc-200">
                        {msg.response.answer}
                      </div>

                      {/* Verified Citations */}
                      {msg.response.citations && msg.response.citations.length > 0 && (
                        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950/60 p-3 space-y-1.5">
                          <div className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">
                            Verified Citations:
                          </div>
                          {msg.response.citations.map((cit, cIdx) => (
                            <div key={cIdx} className="text-xs text-zinc-600 dark:text-zinc-400">
                              <span className="font-mono font-semibold text-indigo-600 dark:text-indigo-400">
                                {cit.source_file}
                              </span>{' '}
                              — <span className="italic">"{cit.quote}"</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Collapsible Sources & Retrieval Debug */}
                      {showDebug && msg.chunks && (
                        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden">
                          <button
                            type="button"
                            onClick={() =>
                              setOpenDebugIds((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }))
                            }
                            className="flex w-full items-center justify-between bg-zinc-50 dark:bg-zinc-950/50 px-3 py-2 text-xs font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
                          >
                            <span>
                              Sources &amp; Retrieval Debug ({msg.chunks.length} reranked chunks)
                            </span>
                            {openDebugIds[msg.id] ? (
                              <ChevronDown className="h-3.5 w-3.5" />
                            ) : (
                              <ChevronRight className="h-3.5 w-3.5" />
                            )}
                          </button>

                          {openDebugIds[msg.id] && (
                            <div className="divide-y divide-zinc-200 dark:divide-zinc-800 p-3 space-y-3 text-xs">
                              {msg.chunks.map((ch, rIdx) => (
                                <div key={ch.chunk_id} className="pt-2 first:pt-0 space-y-1">
                                  <div className="font-semibold text-zinc-800 dark:text-zinc-200">
                                    Rank #{rIdx + 1} —{' '}
                                    <span className="font-mono text-indigo-500">{ch.source_file}</span>{' '}
                                    <span className="font-mono text-zinc-400">({ch.page_or_section})</span>
                                  </div>
                                  <div className="font-mono text-[11px] text-zinc-500">
                                    FlashRank: {ch.rerank_score.toFixed(4)} | RRF: {ch.rrf_score.toFixed(6)} | Dense
                                    Rank: {ch.dense_rank ?? '—'} | BM25 Rank: {ch.bm25_rank ?? '—'}
                                  </div>
                                  <pre className="mt-1 whitespace-pre-wrap rounded-lg bg-zinc-100 dark:bg-zinc-950 p-2.5 font-mono text-[11px] text-zinc-700 dark:text-zinc-300">
                                    {ch.text}
                                  </pre>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : null}
                </div>
              ))}

              {isRunning && activeRun?.pipeline_type === 'query' && (
                <div className="flex items-center gap-2 text-xs text-indigo-500 font-medium px-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Executing Hybrid Retrieval &amp; Grounded Groq Generation — watch live stages above…
                </div>
              )}
            </div>
          )}
        </main>

        {/* ---------------- CHAT INPUT BAR ---------------- */}
        <footer className="shrink-0 border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-[#0d0d11] px-5 py-2.5">
          <form onSubmit={handleFormSubmit} className="mx-auto flex max-w-4xl items-center gap-2">
            <input
              type="text"
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              disabled={isRunning}
              placeholder="Ask a question about company policies, warranties, security SLAs, or employees..."
              className="flex-1 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 px-3.5 py-2 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:border-indigo-500 focus:outline-none disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={isRunning || !promptInput.trim()}
              className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 transition shadow-sm"
            >
              {isRunning && activeRun?.pipeline_type === 'query' ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              Ask
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
};
