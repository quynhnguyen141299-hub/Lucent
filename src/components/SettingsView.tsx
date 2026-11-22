import React, { useEffect, useState } from 'react';
import { Loader2, Save } from 'lucide-react';
import { lucentApi } from '../services/api';
import type {
  AnswerStyle,
  CrossEncoderServiceStatus,
  EngineSettings,
  HybridFusionMethod,
  PublicRuntimeConfig,
  RerankerMode,
  SearchMode,
} from '../types/lucent';

interface SettingsViewProps {
  onSettingsUpdated: (updated: EngineSettings) => void;
}

export function SettingsView({ onSettingsUpdated }: SettingsViewProps) {
  const [settings, setSettings] = useState<EngineSettings | null>(null);
  const [runtime, setRuntime] = useState<PublicRuntimeConfig | null>(null);
  const [crossEncoder, setCrossEncoder] = useState<CrossEncoderServiceStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    lucentApi
      .getSettings()
      .then((res) => {
        setSettings(res.settings);
        setRuntime(res.runtime);
        setCrossEncoder(res.crossEncoder);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to load settings.');
      })
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!settings) return;
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const res = await lucentApi.updateSettings(settings);
      setSettings(res.settings);
      onSettingsUpdated(res.settings);
      setNotice(
        res.rechunked
          ? `Settings saved and corpus re-segmented (${res.rechunked.documentsRechunked} docs → ${res.rechunked.totalChunks} chunks).`
          : 'Engine configuration saved to server.'
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to save settings.');
    } finally {
      setSaving(false);
    }
  };

  if (loading || !settings) {
    return (
      <div className="p-8 text-center text-sm text-[#94A3B8] flex items-center justify-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Loading engine settings...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="font-display text-2xl font-semibold text-white">Settings</h1>
        <p className="text-sm text-[#94A3B8] mt-1">
          Configure retrieval hyperparameters, hybrid fusion weights, chunk segmentation,
          reranking parameters, and RAG synthesis defaults.
        </p>
      </div>

      {error && (
        <div className="p-3.5 rounded-lg border border-red-500/40 bg-red-950/30 text-xs text-red-200">
          {error}
        </div>
      )}
      {notice && (
        <div className="p-3.5 rounded-lg border border-emerald-500/30 bg-emerald-950/20 text-xs text-emerald-200">
          {notice}
        </div>
      )}

      {/* Read-Only Server Runtime Summary (No API key input UI) */}
      {runtime && (
        <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-3">
          <h2 className="font-display text-sm font-semibold text-white">
            Server Runtime &amp; Service Connections
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
            <div>
              <div className="text-[#64748B]">Server-Side AI Credentials</div>
              <div className={`mt-1 font-medium ${runtime.hasGeminiApiKey ? 'text-emerald-400' : 'text-amber-400'}`}>
                {runtime.hasGeminiApiKey ? 'Configured via server environment' : 'Not detected in server environment'}
              </div>
            </div>
            <div>
              <div className="text-[#64748B]">Active Generation / Embedding Models</div>
              <div className="mt-1 font-mono text-[#E6EDF7]">
                {runtime.generationModel} · {runtime.embeddingModel}
              </div>
            </div>
            <div>
              <div className="text-[#64748B]">Python Cross-Encoder Service</div>
              <div className={`mt-1 font-medium ${crossEncoder?.healthy ? 'text-emerald-400' : 'text-amber-400'}`}>
                {crossEncoder?.healthy
                  ? `Online (${crossEncoder.modelName})`
                  : crossEncoder?.configured
                    ? 'Configured but unreachable'
                    : 'Not configured (CROSS_ENCODER_SERVICE_URL)'}
              </div>
            </div>
          </div>
        </section>
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* Retrieval & BM25 */}
        <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
          <h2 className="font-display text-sm font-semibold text-white">
            BM25 Lexical &amp; Candidate Retrieval
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Default Search Mode</span>
              <select
                value={settings.defaultMode}
                onChange={(e) => setSettings({ ...settings, defaultMode: e.target.value as SearchMode })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
              >
                <option value="hybrid">Hybrid</option>
                <option value="semantic">Semantic</option>
                <option value="keyword">Keyword (BM25)</option>
              </select>
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Final Results (Top-K)</span>
              <input
                type="number"
                min={1}
                max={20}
                value={settings.topK}
                onChange={(e) => setSettings({ ...settings, topK: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">BM25 Candidate Pool</span>
              <input
                type="number"
                min={5}
                max={100}
                value={settings.bm25Candidates}
                onChange={(e) => setSettings({ ...settings, bm25Candidates: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">BM25 k1 (TF Saturation)</span>
              <input
                type="number"
                step={0.05}
                min={0.5}
                max={3}
                value={settings.bm25K1}
                onChange={(e) => setSettings({ ...settings, bm25K1: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">BM25 b (Length Norm)</span>
              <input
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={settings.bm25B}
                onChange={(e) => setSettings({ ...settings, bm25B: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Semantic Candidate Pool</span>
              <input
                type="number"
                min={5}
                max={100}
                value={settings.semanticCandidates}
                onChange={(e) => setSettings({ ...settings, semanticCandidates: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>
          </div>
        </section>

        {/* Hybrid Fusion */}
        <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
          <h2 className="font-display text-sm font-semibold text-white">
            Hybrid Ranking &amp; Reciprocal Rank Fusion
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Fusion Algorithm</span>
              <select
                value={settings.hybridFusionMethod}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    hybridFusionMethod: e.target.value as HybridFusionMethod,
                  })
                }
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
              >
                <option value="rrf">Reciprocal Rank Fusion (RRF)</option>
                <option value="weighted">Weighted Linear Score Blending</option>
              </select>
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">RRF Constant (k)</span>
              <input
                type="number"
                min={1}
                max={200}
                value={settings.rrfK}
                onChange={(e) => setSettings({ ...settings, rrfK: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Hybrid Candidate Pool</span>
              <input
                type="number"
                min={3}
                max={50}
                value={settings.hybridCandidates}
                onChange={(e) => setSettings({ ...settings, hybridCandidates: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">BM25 Weight</span>
              <input
                type="number"
                step={0.1}
                min={0}
                max={5}
                value={settings.bm25Weight}
                onChange={(e) => setSettings({ ...settings, bm25Weight: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Semantic Weight</span>
              <input
                type="number"
                step={0.1}
                min={0}
                max={5}
                value={settings.semanticWeight}
                onChange={(e) => setSettings({ ...settings, semanticWeight: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>
          </div>
        </section>

        {/* Reranking */}
        <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
          <h2 className="font-display text-sm font-semibold text-white">
            Second-Stage Neural &amp; LLM Reranking
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Default Reranker</span>
              <select
                value={settings.defaultRerankerMode}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    defaultRerankerMode: e.target.value as RerankerMode,
                  })
                }
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
              >
                <option value="none">Disabled</option>
                <option value="cross_encoder">Cross-Encoder Service (Python MS MARCO)</option>
                <option value="llm">LLM-Based Reranker (Gemini)</option>
              </select>
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Candidates Reranked</span>
              <input
                type="number"
                min={2}
                max={30}
                value={settings.rerankCandidateCount}
                onChange={(e) =>
                  setSettings({ ...settings, rerankCandidateCount: Number(e.target.value) })
                }
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Inference Batch Size</span>
              <input
                type="number"
                min={1}
                max={32}
                value={settings.rerankBatchSize}
                onChange={(e) =>
                  setSettings({ ...settings, rerankBatchSize: Number(e.target.value) })
                }
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Max Sequence Length (tokens)</span>
              <input
                type="number"
                min={128}
                max={1024}
                step={64}
                value={settings.rerankMaxSeqLength}
                onChange={(e) =>
                  setSettings({ ...settings, rerankMaxSeqLength: Number(e.target.value) })
                }
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Reranking Score Threshold</span>
              <input
                type="number"
                step={0.05}
                min={-10}
                max={1}
                value={settings.rerankThreshold}
                onChange={(e) =>
                  setSettings({ ...settings, rerankThreshold: Number(e.target.value) })
                }
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>
          </div>
        </section>

        {/* Ingestion, Embeddings & RAG Defaults */}
        <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
          <h2 className="font-display text-sm font-semibold text-white">
            Ingestion Chunking, Embeddings &amp; RAG Defaults
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Chunk Size (words)</span>
              <input
                type="number"
                min={60}
                max={600}
                value={settings.chunkSize}
                onChange={(e) => setSettings({ ...settings, chunkSize: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Chunk Overlap (words)</span>
              <input
                type="number"
                min={0}
                max={200}
                value={settings.chunkOverlap}
                onChange={(e) => setSettings({ ...settings, chunkOverlap: Number(e.target.value) })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Embedding Model</span>
              <select
                value={settings.embeddingModel}
                onChange={(e) => setSettings({ ...settings, embeddingModel: e.target.value })}
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono text-[#E6EDF7]"
              >
                {(runtime?.supportedEmbeddingModels || ['gemini-embedding-2']).map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Embedding Batch Size</span>
              <input
                type="number"
                min={1}
                max={32}
                value={settings.embeddingBatchSize}
                onChange={(e) =>
                  setSettings({ ...settings, embeddingBatchSize: Number(e.target.value) })
                }
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] font-mono tabular-nums text-[#E6EDF7]"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-[#94A3B8]">Default RAG Answer Style</span>
              <select
                value={settings.defaultAnswerStyle}
                onChange={(e) =>
                  setSettings({ ...settings, defaultAnswerStyle: e.target.value as AnswerStyle })
                }
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
              >
                <option value="concise">Concise</option>
                <option value="detailed">Detailed Research</option>
                <option value="bullets">Bullet-Point Summary</option>
                <option value="comparative">Comparative Analysis</option>
              </select>
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-6 pt-2 text-xs text-[#CBD5E1]">
            <label className="inline-flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.autoEmbedOnUpload}
                onChange={(e) =>
                  setSettings({ ...settings, autoEmbedOnUpload: e.target.checked })
                }
              />
              <span>Automatically generate embeddings on document upload</span>
            </label>
            <label className="inline-flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.enableRagByDefault}
                onChange={(e) =>
                  setSettings({ ...settings, enableRagByDefault: e.target.checked })
                }
              />
              <span>Generate citation-grounded RAG answer by default</span>
            </label>
          </div>
        </section>

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={saving}
            className="h-10 px-6 rounded-md bg-[#3B82F6] hover:bg-[#2563EB] disabled:opacity-50 text-sm font-medium text-white flex items-center gap-2"
          >
            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            <span>{saving ? 'Saving...' : 'Save Configuration'}</span>
          </button>
        </div>
      </form>
    </div>
  );
}
