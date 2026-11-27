import {
  DocumentItem,
  HealthResponse,
  OverviewResponse,
  PipelineEvent,
  RunRecord,
  RunSummary,
  RuntimeSettings,
} from './types';

async function checkJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return checkJson<HealthResponse>(await fetch('/api/health'));
}

export async function fetchOverview(): Promise<OverviewResponse> {
  return checkJson<OverviewResponse>(await fetch('/api/overview'));
}

export async function fetchDocuments(): Promise<{ documents: DocumentItem[]; total_chunks: number }> {
  return checkJson(await fetch('/api/documents'));
}

export async function deleteDocument(filename: string): Promise<void> {
  await checkJson(await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' }));
}

export async function startIngestionRun(files: File[], resetIndex: boolean): Promise<{ run_id: string; run: RunRecord }> {
  const form = new FormData();
  form.append('reset_index', String(resetIndex));
  for (const f of files) {
    form.append('files', f);
  }
  return checkJson(await fetch('/api/ingest/start', { method: 'POST', body: form }));
}

export async function startQueryRun(params: {
  query: string;
  model?: string;
  response_language?: string;
  embedding_profile?: string;
  top_n?: number;
  dense_top_k?: number;
  bm25_top_k?: number;
  rrf_k?: number;
}): Promise<{ run_id: string; run: RunRecord }> {
  return checkJson(
    await fetch('/api/query/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
  );
}

export async function fetchRuns(limit = 30): Promise<{ runs: RunSummary[] }> {
  return checkJson(await fetch(`/api/runs?limit=${limit}`));
}

export async function fetchRun(runId: string): Promise<RunRecord> {
  return checkJson(await fetch(`/api/runs/${encodeURIComponent(runId)}`));
}

export async function exploreRetrieval(params: {
  query: string;
  dense_top_k?: number;
  bm25_top_k?: number;
  rrf_k?: number;
  top_n?: number;
}): Promise<any> {
  return checkJson(
    await fetch('/api/retrieval/explore', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
  );
}

export async function fetchEvaluation(): Promise<{ dataset: any[]; latest_report: any | null }> {
  return checkJson(await fetch('/api/evaluation'));
}

export async function runEvaluationBenchmark(): Promise<{ report: any }> {
  return checkJson(await fetch('/api/evaluation/run', { method: 'POST' }));
}

export async function updateRuntimeSettings(patch: Partial<RuntimeSettings>): Promise<{ settings: RuntimeSettings }> {
  return checkJson(
    await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    })
  );
}

/**
 * Subscribe to a live backend SSE stream for `runId` with automatic state recovery on disconnect.
 */
export function subscribeToRunStream(
  runId: string,
  handlers: {
    onSnapshot: (run: RunRecord) => void;
    onStageEvent: (ev: PipelineEvent) => void;
    onComplete: (run: RunRecord) => void;
    onConnectionChange?: (connected: boolean) => void;
  }
): () => void {
  let closed = false;
  let es: EventSource | null = null;
  let reconnectTimer: number | null = null;

  const connect = () => {
    if (closed) return;
    es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream`);

    es.onopen = () => {
      handlers.onConnectionChange?.(true);
    };

    es.onmessage = (msgEvent) => {
      if (closed || !msgEvent.data) return;
      try {
        const payload = JSON.parse(msgEvent.data);
        if (payload.type === 'snapshot' && payload.run) {
          handlers.onSnapshot(payload.run);
          if (payload.run.status === 'completed' || payload.run.status === 'failed') {
            es?.close();
          }
        } else if (payload.type === 'stage_event' && payload.event) {
          handlers.onStageEvent(payload.event);
        } else if (payload.type === 'run_complete' && payload.run) {
          handlers.onComplete(payload.run);
          es?.close();
        }
      } catch {
        // ignore malformed keep-alive
      }
    };

    es.onerror = () => {
      handlers.onConnectionChange?.(false);
      es?.close();
      if (closed) return;
      // Recover authoritative run state via HTTP and reconnect only if still running
      fetchRun(runId)
        .then((latest) => {
          if (closed) return;
          handlers.onSnapshot(latest);
          if (latest.status === 'running') {
            reconnectTimer = window.setTimeout(connect, 1200);
          } else {
            handlers.onComplete(latest);
          }
        })
        .catch(() => {
          if (!closed) {
            reconnectTimer = window.setTimeout(connect, 2000);
          }
        });
    };
  };

  connect();

  return () => {
    closed = true;
    if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
    es?.close();
  };
}
