import React, { useMemo } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  Edge,
  MarkerType,
  Node,
  Position,
  ReactFlow,
} from '@xyflow/react';
import { PipelineNode, PipelineNodeData } from './PipelineNode';
import { PipelineType, RunRecord, StageState, StageStatus } from '../types';

const nodeTypes = {
  pipelineStage: PipelineNode,
};

interface PipelineCanvasProps {
  activeRun: RunRecord | null;
  pipeline: PipelineType;
  selectedStageId: string | null;
  onSelectStage: (stageId: string) => void;
  darkMode: boolean;
}

interface BlueprintNode {
  id: string;
  label: string;
  subtitle: string;
  pipeline: PipelineType;
  x: number;
  y: number;
  targetPos?: Position;
  sourcePos?: Position;
}

/**
 * Compact 2-row Ingestion DAG (~1,390px x 185px) with zero wasted vertical whitespace.
 */
const INGESTION_BLUEPRINT: BlueprintNode[] = [
  { id: 'file_upload', label: '1. File Upload', subtitle: './data scanner', pipeline: 'ingestion', x: 15, y: 52 },
  { id: 'file_validation', label: '2. File Validation', subtitle: 'Magic bytes & size', pipeline: 'ingestion', x: 250, y: 52 },
  { id: 'document_parser', label: '3a. Document Parser', subtitle: 'PyMuPDF / OpenPyXL', pipeline: 'ingestion', x: 485, y: 0 },
  { id: 'ocr_extraction', label: '3b. Spatial OCR', subtitle: 'RapidOCR (images)', pipeline: 'ingestion', x: 485, y: 108 },
  { id: 'recursive_chunking', label: '4. Recursive Chunker', subtitle: '700c / 120c overlap', pipeline: 'ingestion', x: 720, y: 52 },
  { id: 'embedding_generation', label: '5a. BGE Embeddings', subtitle: 'bge-small (384-d)', pipeline: 'ingestion', x: 955, y: 0 },
  { id: 'vector_db_indexing', label: '6. ChromaDB Index', subtitle: 'HNSW cosine upsert', pipeline: 'ingestion', x: 1190, y: 0 },
  { id: 'bm25_indexing', label: '5b. BM25 Corpus Sync', subtitle: 'Exact ID tokens', pipeline: 'ingestion', x: 955, y: 108 },
];

const INGESTION_EDGES: Array<[string, string]> = [
  ['file_upload', 'file_validation'],
  ['file_validation', 'document_parser'],
  ['file_validation', 'ocr_extraction'],
  ['document_parser', 'recursive_chunking'],
  ['ocr_extraction', 'recursive_chunking'],
  ['recursive_chunking', 'embedding_generation'],
  ['embedding_generation', 'vector_db_indexing'],
  ['recursive_chunking', 'bm25_indexing'],
];

/**
 * Compact 2-row S-curve Query DAG (~1,180px x 210px) so it fits cleanly in a ~215px banner above chat.
 */
const QUERY_BLUEPRINT: BlueprintNode[] = [
  { id: 'query_input', label: '1. User Query', subtitle: 'Prompt & Top-N', pipeline: 'query', x: 15, y: 45 },
  { id: 'query_embedding', label: '2a. Query Embedding', subtitle: 'BGE 384-d vector', pipeline: 'query', x: 250, y: 0 },
  { id: 'dense_retrieval', label: '3a. ChromaDB Search', subtitle: 'Dense cosine Top-K', pipeline: 'query', x: 485, y: 0 },
  { id: 'bm25_retrieval', label: '2b. BM25 Search', subtitle: 'Exact keyword Top-K', pipeline: 'query', x: 368, y: 98 },
  { id: 'rrf_fusion', label: '4. RRF Rank Fusion', subtitle: '1 / (60 + rank)', pipeline: 'query', x: 720, y: 45 },
  { id: 'cross_encoder_rerank', label: '5. FlashRank Reranker', subtitle: 'MiniLM ONNX CPU', pipeline: 'query', x: 955, y: 45, sourcePos: Position.Bottom },
  { id: 'context_assembly', label: '6. Context Assembly', subtitle: 'Escaped XML chunks', pipeline: 'query', x: 955, y: 192, targetPos: Position.Top, sourcePos: Position.Left },
  { id: 'llm_generation', label: '7. Groq LLM', subtitle: 'Strict JSON Schema', pipeline: 'query', x: 720, y: 192, targetPos: Position.Right, sourcePos: Position.Left },
  { id: 'citation_verification', label: '8. Quote Verifier', subtitle: 'Pydantic + verbatim', pipeline: 'query', x: 485, y: 192, targetPos: Position.Right, sourcePos: Position.Left },
];

const QUERY_EDGES: Array<[string, string]> = [
  ['query_input', 'query_embedding'],
  ['query_embedding', 'dense_retrieval'],
  ['query_input', 'bm25_retrieval'],
  ['dense_retrieval', 'rrf_fusion'],
  ['bm25_retrieval', 'rrf_fusion'],
  ['rrf_fusion', 'cross_encoder_rerank'],
  ['cross_encoder_rerank', 'context_assembly'],
  ['context_assembly', 'llm_generation'],
  ['llm_generation', 'citation_verification'],
];

function extractBadge(stageId: string, st?: StageState): string | null {
  if (!st || !st.metadata) return null;
  const m = st.metadata;
  switch (stageId) {
    case 'file_upload':
      return m.discovered_files ? `${m.discovered_files.length} files` : null;
    case 'file_validation':
      return m.valid_documents !== undefined
        ? `${(m.valid_documents?.length || 0) + (m.valid_images?.length || 0)} valid`
        : null;
    case 'document_parser':
      return m.files_parsed ? `${m.files_parsed.length} parsed` : null;
    case 'ocr_extraction':
      return m.images_processed?.length ? `${m.images_processed.length} OCR` : null;
    case 'recursive_chunking':
      return m.total_chunks !== undefined ? `${m.total_chunks} chunks` : null;
    case 'embedding_generation':
      if (st.status === 'running' && m.total_chunks) {
        return `${m.chunks_processed || 0}/${m.total_chunks}`;
      }
      return m.vectors_generated ? `${m.vectors_generated} vecs` : null;
    case 'vector_db_indexing':
      return m.total_persisted_vectors !== undefined ? `${m.total_persisted_vectors} stored` : null;
    case 'bm25_indexing':
      return m.bm25_doc_count !== undefined ? `${m.bm25_doc_count} docs` : null;
    case 'query_input':
      return m.indexed_chunks_available !== undefined ? `${m.indexed_chunks_available} idx` : null;
    case 'query_embedding':
      return m.dimensions ? `${m.dimensions}-d` : null;
    case 'dense_retrieval':
      return m.retrieved_count !== undefined ? `${m.retrieved_count} hits` : null;
    case 'bm25_retrieval':
      return m.retrieved_count !== undefined ? `${m.retrieved_count} hits` : null;
    case 'rrf_fusion':
      return m.unique_fused_count !== undefined ? `${m.unique_fused_count} fused` : null;
    case 'cross_encoder_rerank':
      return m.output_count !== undefined ? `Top ${m.output_count}` : null;
    case 'context_assembly':
      return m.chunks_injected !== undefined ? `${m.chunks_injected} XML` : null;
    case 'llm_generation':
      return m.token_usage?.total_tokens ? `${m.token_usage.total_tokens} tok` : null;
    case 'citation_verification':
      if (m.is_supported === false) return 'Abstained';
      return m.verified_citations_count !== undefined ? `${m.verified_citations_count} cited` : null;
    default:
      return null;
  }
}

export const PipelineCanvas: React.FC<PipelineCanvasProps> = ({
  activeRun,
  pipeline,
  selectedStageId,
  onSelectStage,
  darkMode,
}) => {
  const { nodes, edges } = useMemo(() => {
    const blueprints = pipeline === 'ingestion' ? INGESTION_BLUEPRINT : QUERY_BLUEPRINT;
    const rawEdges = pipeline === 'ingestion' ? INGESTION_EDGES : QUERY_EDGES;
    const stageMap: Record<string, StageState> = activeRun?.stages || {};
    const belongsToRun = activeRun && activeRun.pipeline_type === pipeline;

    const builtNodes: Node[] = blueprints.map((bp) => {
      const stObj = belongsToRun ? stageMap[bp.id] : undefined;
      const status: StageStatus = stObj ? stObj.status : 'idle';

      const nodeData: PipelineNodeData = {
        stageId: bp.id,
        label: bp.label,
        subtitle: bp.subtitle,
        pipelineType: bp.pipeline,
        status,
        durationMs: stObj?.duration_ms ?? null,
        message: stObj?.message || 'Idle',
        badgeText: extractBadge(bp.id, stObj),
        selected: selectedStageId === bp.id,
        targetPos: bp.targetPos,
        sourcePos: bp.sourcePos,
      };

      return {
        id: bp.id,
        type: 'pipelineStage',
        position: { x: bp.x, y: bp.y },
        data: nodeData,
      };
    });

    const builtEdges: Edge[] = rawEdges.map(([src, tgt]) => {
      const srcState = belongsToRun ? stageMap[src]?.status : 'idle';
      const tgtState = belongsToRun ? stageMap[tgt]?.status : 'idle';

      const isFlowing = tgtState === 'running' || (srcState === 'running' && tgtState === 'waiting');
      const isDone = srcState === 'completed' && tgtState === 'completed';
      const isSkipped = srcState === 'skipped' || tgtState === 'skipped';

      let stroke = darkMode ? '#3f3f46' : '#cbd5e1';
      let strokeWidth = 2.0;
      if (isFlowing) {
        stroke = '#6366f1';
        strokeWidth = 2.8;
      } else if (isDone) {
        stroke = '#10b981';
        strokeWidth = 2.3;
      } else if (isSkipped) {
        stroke = darkMode ? '#27272a' : '#e2e8f0';
      }

      return {
        id: `${src}->${tgt}`,
        source: src,
        target: tgt,
        type: 'smoothstep',
        animated: isFlowing,
        style: {
          stroke,
          strokeWidth,
          strokeDasharray: isSkipped ? '5 5' : undefined,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: stroke,
          width: 16,
          height: 16,
        },
      };
    });

    return { nodes: builtNodes, edges: builtEdges };
  }, [activeRun, pipeline, selectedStageId, darkMode]);

  return (
    <div className="relative h-full w-full bg-zinc-50 dark:bg-[#09090b]">
      <ReactFlow
        key={pipeline}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.08 }}
        minZoom={0.35}
        maxZoom={1.5}
        onNodeClick={(_, node) => {
          const d = node.data as unknown as PipelineNodeData;
          onSelectStage(d.stageId);
        }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1.2}
          color={darkMode ? '#27272a' : '#d4d4d8'}
        />
        <Controls position="bottom-left" showInteractive={false} />
      </ReactFlow>
    </div>
  );
};
