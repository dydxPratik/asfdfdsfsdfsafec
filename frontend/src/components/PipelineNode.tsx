import React from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { motion } from 'framer-motion';
import {
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  FileCode2,
  FileSearch,
  FileText,
  Filter,
  GitMerge,
  Loader2,
  MinusCircle,
  ScanText,
  ShieldCheck,
  Sparkles,
  SplitSquareVertical,
  UploadCloud,
  XCircle,
} from 'lucide-react';
import { StageStatus } from '../types';

export interface PipelineNodeData extends Record<string, unknown> {
  stageId: string;
  label: string;
  subtitle: string;
  pipelineType: 'ingestion' | 'query';
  status: StageStatus;
  durationMs: number | null;
  message: string;
  badgeText: string | null;
  selected: boolean;
  targetPos?: Position;
  sourcePos?: Position;
}

const STAGE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  file_upload: UploadCloud,
  file_validation: ShieldCheck,
  document_parser: FileText,
  ocr_extraction: ScanText,
  recursive_chunking: SplitSquareVertical,
  embedding_generation: Cpu,
  vector_db_indexing: Database,
  bm25_indexing: FileCode2,
  query_input: FileSearch,
  tabular_sql: Database,
  query_embedding: Cpu,
  dense_retrieval: Database,
  bm25_retrieval: Filter,
  rrf_fusion: GitMerge,
  cross_encoder_rerank: Sparkles,
  context_assembly: FileCode2,
  llm_generation: Sparkles,
  citation_verification: ShieldCheck,
};

function statusStyles(status: StageStatus, selected: boolean) {
  const ring = selected ? 'ring-2 ring-indigo-500/80 dark:ring-indigo-400/80' : '';
  switch (status) {
    case 'running':
      return {
        card: `border-indigo-500 bg-white dark:bg-zinc-900 shadow-[0_0_24px_-6px_rgba(99,102,241,0.45)] ${ring}`,
        badge: 'bg-indigo-500/15 text-indigo-600 dark:text-indigo-300 border-indigo-500/30',
        dot: 'text-indigo-500 dark:text-indigo-400',
        label: 'Running',
      };
    case 'completed':
      return {
        card: `border-emerald-500/45 bg-white dark:bg-zinc-900/95 hover:border-emerald-500/80 ${ring}`,
        badge: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30',
        dot: 'text-emerald-500 dark:text-emerald-400',
        label: 'Completed',
      };
    case 'failed':
      return {
        card: `border-rose-500/70 bg-rose-950/10 dark:bg-rose-950/20 ${ring}`,
        badge: 'bg-rose-500/15 text-rose-600 dark:text-rose-300 border-rose-500/30',
        dot: 'text-rose-500 dark:text-rose-400',
        label: 'Failed',
      };
    case 'skipped':
      return {
        card: `border-dashed border-zinc-300 dark:border-zinc-800 bg-zinc-100/60 dark:bg-zinc-950/60 opacity-60 ${ring}`,
        badge: 'bg-zinc-500/10 text-zinc-500 dark:text-zinc-400 border-zinc-500/20',
        dot: 'text-zinc-400 dark:text-zinc-500',
        label: 'Skipped',
      };
    case 'waiting':
      return {
        card: `border-dashed border-zinc-300 dark:border-zinc-700/80 bg-white/90 dark:bg-zinc-900/70 ${ring}`,
        badge: 'bg-amber-500/10 text-amber-600 dark:text-amber-300/90 border-amber-500/20',
        dot: 'text-amber-500/80 dark:text-amber-400/80',
        label: 'Waiting',
      };
    default:
      return {
        card: `border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/90 hover:border-zinc-400 dark:hover:border-zinc-700 ${ring}`,
        badge: 'bg-zinc-500/10 text-zinc-600 dark:text-zinc-400 border-zinc-500/20',
        dot: 'text-zinc-400 dark:text-zinc-500',
        label: 'Idle',
      };
  }
}

export const PipelineNode: React.FC<NodeProps> = ({ data }) => {
  const d = data as unknown as PipelineNodeData;
  const Icon = STAGE_ICONS[d.stageId] || Cpu;
  const st = statusStyles(d.status, d.selected);
  const targetPosition = d.targetPos || Position.Left;
  const sourcePosition = d.sourcePos || Position.Right;

  return (
    <motion.div
      initial={{ opacity: 0.92, scale: 0.98 }}
      animate={{
        opacity: d.status === 'skipped' ? 0.58 : 1,
        scale: d.status === 'running' ? 1.03 : 1,
      }}
      transition={{ duration: 0.18 }}
      className={`relative w-[204px] rounded-xl border px-3 py-2 transition-colors cursor-pointer select-none shadow-sm ${st.card}`}
    >
      <Handle
        type="target"
        position={targetPosition}
        className="!w-2 !h-2 !bg-zinc-400 dark:!bg-zinc-600 !border-0"
      />

      {d.status === 'running' && (
        <motion.div
          className="pointer-events-none absolute -inset-[1px] rounded-xl border-2 border-indigo-400/75"
          animate={{ opacity: [0.3, 0.95, 0.3] }}
          transition={{ duration: 1.3, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}

      {/* Header Row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-zinc-100 dark:bg-zinc-800/90 border border-zinc-200/80 dark:border-zinc-700/60">
            <Icon className={`h-3.5 w-3.5 ${st.dot}`} />
          </div>
          <div className="min-w-0">
            <div className="truncate text-xs font-semibold text-zinc-900 dark:text-zinc-100">
              {d.label}
            </div>
            <div className="truncate text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
              {d.subtitle}
            </div>
          </div>
        </div>

        {/* Status Icon */}
        <div className="shrink-0 pt-0.5">
          {d.status === 'running' && <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-500" />}
          {d.status === 'completed' && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />}
          {d.status === 'failed' && <XCircle className="h-3.5 w-3.5 text-rose-500" />}
          {d.status === 'skipped' && <MinusCircle className="h-3.5 w-3.5 text-zinc-400" />}
          {d.status === 'waiting' && <Clock className="h-3.5 w-3.5 text-amber-400/80" />}
        </div>
      </div>

      {/* Footer Status & Telemetry */}
      <div className="mt-1.5 flex items-center justify-between gap-1.5 pt-1.5 border-t border-zinc-100 dark:border-zinc-800/80">
        <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium ${st.badge}`}>
          {st.label}
        </span>

        <div className="flex items-center gap-1 font-mono text-[10px] text-zinc-500 dark:text-zinc-400 truncate">
          {d.badgeText && (
            <span className="truncate rounded bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 text-zinc-700 dark:text-zinc-300">
              {d.badgeText}
            </span>
          )}
          {d.durationMs !== null && d.status !== 'skipped' && (
            <span className="shrink-0 tabular-nums font-medium">
              {d.durationMs >= 1000 ? `${(d.durationMs / 1000).toFixed(2)}s` : `${Math.round(d.durationMs)}ms`}
            </span>
          )}
        </div>
      </div>

      <Handle
        type="source"
        position={sourcePosition}
        className="!w-2 !h-2 !bg-zinc-400 dark:!bg-zinc-600 !border-0"
      />
    </motion.div>
  );
};
