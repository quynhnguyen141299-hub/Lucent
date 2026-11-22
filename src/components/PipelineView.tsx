import React, { useEffect, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { lucentApi } from '../services/api';
import type {
  CrossEncoderServiceStatus,
  EngineSettings,
  PublicRuntimeConfig,
  SearchPipelineTrace,
  VectorIndexStats,
} from '../types/lucent';

interface PipelineState {
  runtime: PublicRuntimeConfig;
  settings: EngineSettings;
  crossEncoder: CrossEncoderServiceStatus;
  vectorIndex: VectorIndexStats;
  indexSummary: { totalDocuments: number; totalChunks: number; embeddedChunks: number };
  lastTrace: SearchPipelineTrace | null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function PipelineView() {
  const [data, setData] = useState<PipelineState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await lucentApi.getPipelineStatus();
      setData(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to inspect pipeline status.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  if (loading && !data) {
    return (
      <div className="p-8 text-center text-sm text-[#94A3B8] flex items-center justify-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Loading retrieval pipeline diagnostics...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-4 rounded-lg border border-red-500/40 bg-red-950/30 text-sm text-red-200">
        {error || 'Pipeline status unavailable.'}
      </div>
    );
  }

  const stages = [
    {
      num: '01',
      title: 'Document Ingestion & Segmentation',
      module: 'server/modules/ingestion.ts',
      status: 'Active',
      statusTone: 'text-emerald-400',
      summary: `PDF (per-page), DOCX, TXT, and Markdown extraction with text cleaning, whitespace normalization, and ${data.settings.chunkSize}-word overlapping windows (${data.settings.chunkOverlap}-word overlap).`,
      metric: `${data.indexSummary.totalDocuments} docs · ${data.indexSummary.totalChunks} chunks`,
    },
    {
      num: '02',
      title: 'Query Intent Classification',
      module: 'server/modules/generation.ts',
      status: data.runtime.hasGeminiApiKey ? 'Active' : 'API Key Required',
      statusTone: data.runtime.hasGeminiApiKey ? 'text-emerald-400' : 'text-amber-400',
      summary:
        'Classifies natural-language questions into Factual, Comparative, Historical, Analytical, or Explanatory intents via validated structured JSON output.',
      metric: data.runtime.generationModel,
    },
    {
      num: '03',
      title: 'Lexical Retrieval (Okapi BM25)',
      module: 'server/modules/retrieval.ts',
      status: 'Active',
      statusTone: 'text-emerald-400',
      summary: `Deterministic sparse matching with stopword filtering, smoothed IDF, term saturation k1=${data.settings.bm25K1}, and length normalization b=${data.settings.bm25B}.`,
      metric: `Top ${data.settings.bm25Candidates} candidates`,
    },
    {
      num: '04',
      title: 'Dense Vector Retrieval & FlatIP Index',
      module: 'server/modules/vectorIndex.ts',
      status: data.runtime.hasGeminiApiKey ? 'Active' : 'API Key Required',
      statusTone: data.runtime.hasGeminiApiKey ? 'text-emerald-400' : 'text-amber-400',
      summary: `${data.vectorIndex.backend}. Exact inner product over L2-normalized Float32 embeddings persisted to ${data.vectorIndex.persistedPath} with SHA-256 unchanged-chunk deduplication.`,
      metric: `${data.vectorIndex.vectorCount} vectors (${formatBytes(data.vectorIndex.storageBytes)})`,
    },
    {
      num: '05',
      title: 'Hybrid Rank Fusion (RRF / Weighted)',
      module: 'server/modules/ranking.ts',
      status: 'Active',
      statusTone: 'text-emerald-400',
      summary: `Combines BM25 and dense rankings via ${
        data.settings.hybridFusionMethod === 'rrf'
          ? `Reciprocal Rank Fusion (k=${data.settings.rrfK}, w_bm25=${data.settings.bm25Weight}, w_sem=${data.settings.semanticWeight})`
          : 'weighted score interpolation'
      } and deduplicates chunk hits to document rank.`,
      metric: `Pool ${data.settings.hybridCandidates} docs`,
    },
    {
      num: '06',
      title: 'Second-Stage Neural Reranking',
      module: 'server/modules/reranking.ts + services/cross_encoder',
      status: data.crossEncoder.healthy
        ? 'Cross-Encoder Online'
        : data.runtime.hasGeminiApiKey
          ? 'LLM Reranker Ready · Cross-Encoder Offline'
          : 'Offline',
      statusTone: data.crossEncoder.healthy ? 'text-emerald-400' : 'text-amber-400',
      summary: data.crossEncoder.healthy
        ? `Connected to Python Sentence Transformers service (${data.crossEncoder.modelName} on ${data.crossEncoder.device}).`
        : data.crossEncoder.errorMessage ||
          'Connect external Python FastAPI MS MARCO Cross-Encoder service via CROSS_ENCODER_SERVICE_URL, or use optional Gemini LLM-based reranking.',
      metric: `Batch ${data.settings.rerankBatchSize} · Max ${data.settings.rerankMaxSeqLength} tok`,
    },
    {
      num: '07',
      title: 'Evidence-Grounded RAG Synthesis',
      module: 'server/modules/generation.ts',
      status: data.runtime.hasGeminiApiKey ? 'Active' : 'API Key Required',
      statusTone: data.runtime.hasGeminiApiKey ? 'text-emerald-400' : 'text-amber-400',
      summary:
        'Generates concise, detailed, bullet, or comparative research synthesis strictly grounded in top retrieved chunks with programmatic citation validation and fact/interpretation separation.',
      metric: data.runtime.generationModel,
    },
  ];

  const trace = data.lastTrace;
  const maxStageMs = trace
    ? Math.max(1, ...trace.stages.map((s) => s.latencyMs))
    : 1;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-semibold text-white">Retrieval Pipeline</h1>
          <p className="text-sm text-[#94A3B8] mt-1">
            Inspect active server modules, persistent vector index state, cross-encoder adapter
            connectivity, and stage-by-stage execution traces.
          </p>
        </div>
        <button
          type="button"
          onClick={loadStatus}
          disabled={loading}
          className="h-9 px-3.5 rounded-md border border-[#24324D] bg-[#0D1321] hover:bg-[#131B2E] text-xs font-medium text-[#E6EDF7] flex items-center gap-1.5 whitespace-nowrap"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Diagnostics</span>
        </button>
      </div>

      {/* Modular Architecture Table */}
      <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] overflow-hidden">
        <div className="px-5 py-3.5 border-b border-[#1E293B] flex items-center justify-between">
          <h2 className="font-display text-sm font-semibold text-white">
            Implemented Backend Stages
          </h2>
          <span className="text-xs text-[#64748B]">Server-side TypeScript + optional Python ML service</span>
        </div>

        <div className="divide-y divide-[#1E293B]">
          {stages.map((st) => (
            <div
              key={st.num}
              className="p-4 sm:px-5 grid grid-cols-1 lg:grid-cols-12 gap-3 items-start"
            >
              <div className="lg:col-span-4 space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs text-[#64748B] tabular-nums">{st.num}.</span>
                  <h3 className="text-sm font-medium text-white">{st.title}</h3>
                </div>
                <div className="pl-6 font-mono text-[11px] text-[#64748B]">{st.module}</div>
              </div>

              <div className="lg:col-span-5 text-xs text-[#94A3B8] leading-relaxed">
                {st.summary}
              </div>

              <div className="lg:col-span-3 lg:text-right space-y-0.5">
                <div className={`text-xs font-medium ${st.statusTone}`}>{st.status}</div>
                <div className="font-mono text-[11px] text-[#64748B] tabular-nums">{st.metric}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Last Real Query Execution Trace */}
      <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2 pb-3 border-b border-[#1E293B]">
          <div>
            <h2 className="font-display text-sm font-semibold text-white">
              Most Recent Query Execution Trace
            </h2>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              Measured stage latencies and candidate counts from actual server execution.
            </p>
          </div>
          {trace && (
            <div className="font-mono text-xs text-[#CBD5E1] tabular-nums">
              Total {trace.totalLatencyMs}ms · {trace.mode} · reranker: {trace.rerankerMode}
            </div>
          )}
        </div>

        {!trace ? (
          <div className="py-8 text-center text-xs text-[#94A3B8]">
            No search query has been executed in this session yet. Run a query on the Search page to
            inspect its real stage-by-stage trace.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-xs text-[#CBD5E1] flex flex-wrap items-center gap-2">
              <span className="text-[#64748B]">Query:</span>
              <span className="font-medium text-white">“{trace.query}”</span>
              <span aria-hidden="true" className="text-[#334155]">·</span>
              <span className="font-mono tabular-nums text-[#94A3B8]">
                {new Date(trace.timestamp).toLocaleTimeString()}
              </span>
            </div>

            <div className="space-y-3">
              {trace.stages.map((st) => {
                const pct =
                  st.latencyMs > 0 ? Math.max(3, Math.round((st.latencyMs / maxStageMs) * 100)) : 0;
                return (
                  <div key={st.stage} className="space-y-1.5">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                      <div className="flex items-center gap-2.5">
                        <span className="font-medium text-[#E6EDF7]">{st.label}</span>
                        <span
                          className={`text-[11px] ${
                            st.status === 'executed'
                              ? 'text-emerald-400'
                              : st.status === 'error'
                                ? 'text-red-400'
                                : 'text-[#64748B]'
                          }`}
                        >
                          {st.status}
                        </span>
                      </div>
                      <div className="font-mono tabular-nums text-[#94A3B8]">
                        {st.candidateCount !== null ? `${st.candidateCount} candidates · ` : ''}
                        {st.latencyMs}ms
                      </div>
                    </div>

                    <div className="h-1.5 w-full rounded bg-[#070B14] overflow-hidden">
                      <div
                        className={`h-full rounded ${
                          st.status === 'error'
                            ? 'bg-red-500'
                            : st.stage === 'rag_generation' || st.stage === 'neural_reranking'
                              ? 'bg-[#8B5CF6]'
                              : 'bg-[#3B82F6]'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <p className="text-[11px] text-[#64748B]">{st.details}</p>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
