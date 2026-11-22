import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  Download,
  Edit3,
  FileUp,
  Loader2,
  Play,
  Plus,
  Trash2,
} from 'lucide-react';
import { lucentApi } from '../services/api';
import type {
  EvaluationMethodId,
  EvaluationQuery,
  EvaluationRunRecord,
  ResearchDocument,
} from '../types/lucent';

const ALL_METHODS: Array<{ id: EvaluationMethodId; label: string }> = [
  { id: 'bm25', label: 'BM25 Lexical' },
  { id: 'dense', label: 'Dense Retrieval' },
  { id: 'hybrid', label: 'Hybrid RRF' },
  { id: 'hybrid_cross_encoder', label: 'Hybrid + Cross-Encoder' },
  { id: 'hybrid_llm_rerank', label: 'Hybrid + LLM Reranker' },
];

const METHOD_COLORS: Record<EvaluationMethodId, string> = {
  bm25: '#60A5FA',
  dense: '#34D399',
  hybrid: '#3B82F6',
  hybrid_cross_encoder: '#A78BFA',
  hybrid_llm_rerank: '#F59E0B',
};

export function EvaluationView() {
  const [queries, setQueries] = useState<EvaluationQuery[]>([]);
  const [runs, setRuns] = useState<EvaluationRunRecord[]>([]);
  const [documents, setDocuments] = useState<Array<Omit<ResearchDocument, 'content'>>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Experiment runner config
  const [selectedMethods, setSelectedMethods] = useState<EvaluationMethodId[]>([
    'bm25',
    'dense',
    'hybrid',
  ]);
  const [kValue, setKValue] = useState(5);
  const [datasetScope, setDatasetScope] = useState<'all' | 'human_only' | 'demo_only'>('all');
  const [running, setRunning] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  // Add / Edit query form
  const [showQueryForm, setShowQueryForm] = useState(false);
  const [editingQueryId, setEditingQueryId] = useState<string | null>(null);
  const [formQuery, setFormQuery] = useState('');
  const [formCategory, setFormCategory] = useState('Custom');
  const [formNotes, setFormNotes] = useState('');
  const [formGrades, setFormGrades] = useState<Record<string, number>>({});

  // Import CSV/JSON panel
  const [showImport, setShowImport] = useState(false);
  const [importFormat, setImportFormat] = useState<'csv' | 'json'>('csv');
  const [importContent, setImportContent] = useState('');
  const [importing, setImporting] = useState(false);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [evalRes, docsRes] = await Promise.all([
        lucentApi.getEvaluation(),
        lucentApi.listDocuments(),
      ]);
      setQueries(evalRes.queries);
      setRuns(evalRes.runs);
      setDocuments(docsRes.documents);
      if (evalRes.runs.length > 0 && !selectedRunId) {
        setSelectedRunId(evalRes.runs[0].id);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load evaluation data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const docTitleMap = useMemo(
    () => new Map(documents.map((d) => [d.id, d.title])),
    [documents]
  );

  const activeRun = useMemo(() => {
    if (runs.length === 0) return null;
    return runs.find((r) => r.id === selectedRunId) || runs[0];
  }, [runs, selectedRunId]);

  const toggleMethod = (id: EvaluationMethodId) => {
    setSelectedMethods((prev) =>
      prev.includes(id)
        ? prev.length > 1
          ? prev.filter((m) => m !== id)
          : prev
        : [...prev, id]
    );
  };

  const handleRunExperiment = async () => {
    setRunning(true);
    setError(null);
    setNotice(null);
    try {
      const res = await lucentApi.runEvaluation({
        methods: selectedMethods,
        k: kValue,
        datasetScope,
      });
      setRuns((prev) => [res.run, ...prev]);
      setSelectedRunId(res.run.id);
      setNotice(`Completed experiment ${res.run.id} across ${res.run.queryCount} queries (K=${res.run.k}).`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Evaluation experiment failed.');
    } finally {
      setRunning(false);
    }
  };

  const openCreateForm = () => {
    setEditingQueryId(null);
    setFormQuery('');
    setFormCategory('Custom');
    setFormNotes('');
    setFormGrades({});
    setShowQueryForm(true);
  };

  const openEditForm = (q: EvaluationQuery) => {
    setEditingQueryId(q.id);
    setFormQuery(q.query);
    setFormCategory(q.category);
    setFormNotes(q.notes);
    setFormGrades({ ...q.relevanceGrades });
    setShowQueryForm(true);
  };

  const handleSaveQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setNotice(null);
    const nonZero = Object.fromEntries(
      Object.entries(formGrades).filter(([, g]) => g > 0)
    );
    if (!formQuery.trim() || Object.keys(nonZero).length === 0) {
      setError('Enter a query and assign a relevance grade (1–3) to at least one document.');
      return;
    }

    try {
      if (editingQueryId) {
        await lucentApi.updateEvaluationQuery(editingQueryId, {
          query: formQuery.trim(),
          category: formCategory.trim() || 'Custom',
          notes: formNotes.trim(),
          relevanceGrades: nonZero,
        });
        setNotice('Updated evaluation query relevance judgments.');
      } else {
        await lucentApi.createEvaluationQuery({
          query: formQuery.trim(),
          category: formCategory.trim() || 'Custom',
          notes: formNotes.trim(),
          relevanceGrades: nonZero,
        });
        setNotice('Added human-labelled evaluation query.');
      }
      setShowQueryForm(false);
      await loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to save query.');
    }
  };

  const handleDeleteQuery = async (id: string) => {
    setError(null);
    try {
      await lucentApi.deleteEvaluationQuery(id);
      await loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete query.');
    }
  };

  const handleImportDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    setImporting(true);
    setError(null);
    setNotice(null);
    try {
      const res = await lucentApi.importEvaluationDataset(importFormat, importContent);
      setNotice(`Imported ${res.importedCount} human-labelled evaluation queries from ${importFormat.toUpperCase()}.`);
      setImportContent('');
      setShowImport(false);
      await loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Dataset import failed.');
    } finally {
      setImporting(false);
    }
  };

  const completedReports = activeRun
    ? activeRun.reports.filter((r) => r.status === 'completed')
    : [];

  const maxLatency = Math.max(1, ...completedReports.map((r) => r.avgLatencyMs));

  const humanCount = queries.filter((q) => !q.isDemo).length;
  const demoCount = queries.filter((q) => q.isDemo).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-semibold text-white">
            Model Evaluation Laboratory
          </h1>
          <p className="text-sm text-[#94A3B8] mt-1">
            Quantitatively benchmark retrieval and reranking models using genuine Precision@K,
            Recall@K, MRR, MAP, and graded NDCG@K calculations.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setShowImport((v) => !v)}
            className="h-9 px-3.5 rounded-md border border-[#24324D] bg-[#0D1321] hover:bg-[#131B2E] text-xs font-medium text-[#E6EDF7] flex items-center gap-1.5 whitespace-nowrap"
          >
            <FileUp className="w-3.5 h-3.5 text-[#3B82F6]" />
            <span>Import CSV / JSON</span>
          </button>
          <button
            type="button"
            onClick={openCreateForm}
            className="h-9 px-3.5 rounded-md border border-[#24324D] bg-[#0D1321] hover:bg-[#131B2E] text-xs font-medium text-[#E6EDF7] flex items-center gap-1.5 whitespace-nowrap"
          >
            <Plus className="w-3.5 h-3.5 text-[#3B82F6]" />
            <span>Add Test Query</span>
          </button>
        </div>
      </div>

      {error && (
        <div
          role="alert"
          className="p-3.5 rounded-lg border border-red-500/40 bg-red-950/30 text-xs text-red-200 flex items-start gap-2.5"
        >
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
      {notice && (
        <div className="p-3.5 rounded-lg border border-emerald-500/30 bg-emerald-950/20 text-xs text-emerald-200">
          {notice}
        </div>
      )}

      {/* Experiment Runner Panel */}
      <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-display text-sm font-semibold text-white">Run Experiment</h2>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              Every selected method runs live over identical query judgments and corpus snapshots.
            </p>
          </div>

          <button
            type="button"
            disabled={running || queries.length === 0}
            onClick={handleRunExperiment}
            className="h-9 px-5 rounded-md bg-[#3B82F6] hover:bg-[#2563EB] disabled:opacity-50 text-xs font-medium text-white flex items-center gap-2 whitespace-nowrap"
          >
            {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            <span>{running ? 'Running Experiment...' : 'Run Evaluation'}</span>
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-4 pt-3 border-t border-[#1E293B] text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[#94A3B8]">Methods:</span>
            {ALL_METHODS.map((m) => {
              const checked = selectedMethods.includes(m.id);
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => toggleMethod(m.id)}
                  className={`px-3 py-1.5 rounded border transition-colors whitespace-nowrap ${
                    checked
                      ? 'border-[#3B82F6] bg-[#101A33] text-white'
                      : 'border-[#1E293B] bg-[#070B14] text-[#94A3B8] hover:text-white'
                  }`}
                >
                  {m.label}
                </button>
              );
            })}
          </div>

          <label className="inline-flex items-center gap-2 text-[#94A3B8]">
            <span>Cutoff K:</span>
            <input
              type="number"
              min={1}
              max={20}
              value={kValue}
              onChange={(e) => setKValue(Math.max(1, Math.min(20, Number(e.target.value) || 5)))}
              className="w-16 h-8 px-2 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono tabular-nums"
            />
          </label>

          <label className="inline-flex items-center gap-2 text-[#94A3B8]">
            <span>Dataset:</span>
            <select
              value={datasetScope}
              onChange={(e) =>
                setDatasetScope(e.target.value as 'all' | 'human_only' | 'demo_only')
              }
              className="h-8 px-2.5 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
            >
              <option value="all">All Queries ({queries.length})</option>
              <option value="human_only">Human-Labelled Only ({humanCount})</option>
              <option value="demo_only">Demonstration Only ({demoCount})</option>
            </select>
          </label>
        </div>
      </section>

      {/* CSV / JSON Dataset Import Form */}
      {showImport && (
        <section className="rounded-lg border border-[#24324D] bg-[#0D1321] p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-sm font-semibold text-white">
              Import Human-Labelled Evaluation Dataset
            </h2>
            <div className="inline-flex p-0.5 rounded bg-[#070B14] border border-[#1E293B] text-xs">
              <button
                type="button"
                onClick={() => setImportFormat('csv')}
                className={`px-3 py-1 rounded ${
                  importFormat === 'csv' ? 'bg-[#131B2E] text-white' : 'text-[#94A3B8]'
                }`}
              >
                CSV
              </button>
              <button
                type="button"
                onClick={() => setImportFormat('json')}
                className={`px-3 py-1 rounded ${
                  importFormat === 'json' ? 'bg-[#131B2E] text-white' : 'text-[#94A3B8]'
                }`}
              >
                JSON
              </button>
            </div>
          </div>
          <p className="text-xs text-[#94A3B8]">
            {importFormat === 'csv'
              ? 'CSV format: query,document_id,grade,category,notes (grade 1–3; document_id must match a Document Library ID).'
              : 'JSON format: [{"query": "...", "relevanceGrades": {"doc-id": 3}, "category": "...", "notes": "..."}]'}
          </p>
          <form onSubmit={handleImportDataset} className="space-y-3">
            <textarea
              rows={5}
              value={importContent}
              onChange={(e) => setImportContent(e.target.value)}
              placeholder={
                importFormat === 'csv'
                  ? 'query,document_id,grade,category,notes\n"How does BM25 handle term saturation?",demo-doc-bm25-saturation,3,Lexical,Exact technical test'
                  : '[{"query": "How does BM25 handle term saturation?", "relevanceGrades": {"demo-doc-bm25-saturation": 3}}]'
              }
              className="w-full p-3 rounded bg-[#070B14] border border-[#1E293B] font-mono text-xs text-[#E6EDF7]"
            />
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowImport(false)}
                className="h-8 px-3 rounded border border-[#1E293B] text-xs text-[#94A3B8]"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={importing}
                className="h-8 px-4 rounded bg-[#3B82F6] text-xs font-medium text-white"
              >
                {importing ? 'Importing...' : 'Import Judgments'}
              </button>
            </div>
          </form>
        </section>
      )}

      {/* Add / Edit Query Form with Graded Relevance Labels */}
      {showQueryForm && (
        <section className="rounded-lg border border-[#24324D] bg-[#0D1321] p-5 space-y-4">
          <h2 className="font-display text-sm font-semibold text-white">
            {editingQueryId ? 'Edit Relevance Judgments' : 'Add Human-Labelled Test Query'}
          </h2>
          <form onSubmit={handleSaveQuery} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <label className="sm:col-span-2 block space-y-1">
                <span className="text-[#94A3B8]">Test Query *</span>
                <input
                  type="text"
                  value={formQuery}
                  onChange={(e) => setFormQuery(e.target.value)}
                  placeholder="Enter evaluation question..."
                  className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                />
              </label>
              <label className="block space-y-1">
                <span className="text-[#94A3B8]">Category</span>
                <input
                  type="text"
                  value={formCategory}
                  onChange={(e) => setFormCategory(e.target.value)}
                  className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                />
              </label>
            </div>

            <label className="block space-y-1 text-xs">
              <span className="text-[#94A3B8]">Annotation Notes</span>
              <input
                type="text"
                value={formNotes}
                onChange={(e) => setFormNotes(e.target.value)}
                placeholder="Why these documents are relevant..."
                className="w-full h-9 px-3 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
              />
            </label>

            <div className="space-y-2">
              <div className="text-xs text-[#94A3B8]">
                Graded Relevance Labels (0 = Not Relevant, 1 = Marginal, 2 = Relevant, 3 = Highly Relevant)
              </div>
              <div className="max-h-56 overflow-y-auto divide-y divide-[#1E293B] rounded border border-[#1E293B] bg-[#070B14]">
                {documents.map((doc) => {
                  const grade = formGrades[doc.id] || 0;
                  return (
                    <div
                      key={doc.id}
                      className="px-3 py-2 flex items-center justify-between gap-3 text-xs"
                    >
                      <div className="min-w-0">
                        <div className="text-[#E6EDF7] truncate">{doc.title}</div>
                        <div className="font-mono text-[10px] text-[#64748B]">{doc.id}</div>
                      </div>
                      <div className="inline-flex items-center gap-1 shrink-0">
                        {[0, 1, 2, 3].map((g) => (
                          <button
                            key={g}
                            type="button"
                            onClick={() =>
                              setFormGrades((prev) => ({
                                ...prev,
                                [doc.id]: g,
                              }))
                            }
                            className={`w-7 h-7 rounded font-mono text-xs transition-colors ${
                              grade === g
                                ? g === 0
                                  ? 'bg-[#1E293B] text-white'
                                  : 'bg-[#3B82F6] text-white'
                                : 'bg-[#0D1321] text-[#64748B] hover:text-white border border-[#1E293B]'
                            }`}
                          >
                            {g}
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowQueryForm(false)}
                className="h-9 px-4 rounded border border-[#1E293B] text-xs text-[#94A3B8]"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="h-9 px-5 rounded bg-[#3B82F6] text-xs font-medium text-white"
              >
                Save Judgments
              </button>
            </div>
          </form>
        </section>
      )}

      {/* Latest / Selected Experiment Results + SVG Visualizations */}
      {activeRun ? (
        <div className="space-y-6">
          {/* Experiment Selector & Report Download */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#1E293B]">
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <label className="inline-flex items-center gap-2 text-[#94A3B8]">
                <span>Experiment:</span>
                <select
                  value={activeRun.id}
                  onChange={(e) => setSelectedRunId(e.target.value)}
                  className="h-8 px-2.5 rounded bg-[#0D1321] border border-[#1E293B] text-[#E6EDF7] font-mono"
                >
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.id} · {new Date(r.timestamp).toLocaleString()} (K={r.k}, n={r.queryCount})
                    </option>
                  ))}
                </select>
              </label>
              <span className="text-[#94A3B8]">
                {activeRun.datasetLabel} · <span className="font-mono">{activeRun.datasetVersion}</span>
              </span>
            </div>

            <a
              href={`/api/evaluation/runs/${encodeURIComponent(activeRun.id)}/report`}
              className="h-8 px-3.5 rounded-md border border-[#24324D] bg-[#0D1321] hover:bg-[#131B2E] text-xs font-medium text-[#E6EDF7] flex items-center gap-1.5 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5 text-[#3B82F6]" />
              <span>Download Report (.md)</span>
            </a>
          </div>

          {/* Metrics Comparison Table */}
          <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-[#1E293B] text-[#64748B]">
                  <th className="py-3 px-4">Method &amp; Model</th>
                  <th className="py-3 px-3 text-right font-mono">P@{activeRun.k}</th>
                  <th className="py-3 px-3 text-right font-mono">R@{activeRun.k}</th>
                  <th className="py-3 px-3 text-right font-mono">MRR</th>
                  <th className="py-3 px-3 text-right font-mono">MAP</th>
                  <th className="py-3 px-3 text-right font-mono">NDCG@{activeRun.k}</th>
                  <th className="py-3 px-4 text-right font-mono">Avg Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E293B]">
                {activeRun.reports.map((rep) => (
                  <tr key={rep.method} className="hover:bg-[#11192C]">
                    <td className="py-3 px-4">
                      <div className="font-medium text-white flex items-center gap-2">
                        <span
                          className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
                          style={{ backgroundColor: METHOD_COLORS[rep.method] }}
                        />
                        <span>{rep.label}</span>
                      </div>
                      <div className="text-[11px] font-mono text-[#64748B] pl-4.5 mt-0.5">
                        {rep.modelName} ({rep.modelVersion})
                      </div>
                      {rep.status === 'error' && (
                        <div className="text-[11px] text-red-300 pl-4.5 mt-1">
                          {rep.errorMessage}
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-[#E6EDF7]">
                      {rep.status === 'completed' ? `${(rep.avgPrecisionAtK * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-[#E6EDF7]">
                      {rep.status === 'completed' ? `${(rep.avgRecallAtK * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-[#E6EDF7]">
                      {rep.status === 'completed' ? rep.meanReciprocalRank.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-[#E6EDF7]">
                      {rep.status === 'completed' ? rep.meanAveragePrecision.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums font-medium text-white">
                      {rep.status === 'completed' ? rep.avgNdcgAtK.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums text-[#94A3B8]">
                      {rep.status === 'completed' ? `${rep.avgLatencyMs} ms` : 'Error'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {/* Visualisations Grid (100% derived from actual run results) */}
          {completedReports.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Chart 1: NDCG, MRR, MAP Comparison */}
              <div className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-4 space-y-3">
                <h3 className="text-xs font-semibold text-white">
                  NDCG@{activeRun.k}, MRR &amp; MAP Comparison
                </h3>
                <div className="space-y-3">
                  {completedReports.map((rep) => (
                    <div key={rep.method} className="space-y-1 text-[11px]">
                      <div className="flex justify-between text-[#CBD5E1]">
                        <span>{rep.label}</span>
                        <span className="font-mono tabular-nums">
                          NDCG {rep.avgNdcgAtK.toFixed(3)} · MRR {rep.meanReciprocalRank.toFixed(3)}
                        </span>
                      </div>
                      <div className="space-y-1">
                        <div className="h-2 rounded bg-[#070B14] overflow-hidden">
                          <div
                            className="h-full rounded"
                            style={{
                              width: `${Math.round(rep.avgNdcgAtK * 100)}%`,
                              backgroundColor: METHOD_COLORS[rep.method],
                            }}
                          />
                        </div>
                        <div className="h-1.5 rounded bg-[#070B14] overflow-hidden">
                          <div
                            className="h-full rounded opacity-65"
                            style={{
                              width: `${Math.round(rep.meanAveragePrecision * 100)}%`,
                              backgroundColor: METHOD_COLORS[rep.method],
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                  <p className="text-[10px] text-[#64748B]">
                    Top bar: NDCG@{activeRun.k} · Lower bar: MAP
                  </p>
                </div>
              </div>

              {/* Chart 2: Precision vs Recall SVG Scatter */}
              <div className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-4 space-y-3">
                <h3 className="text-xs font-semibold text-white">
                  Precision@{activeRun.k} vs Recall@{activeRun.k}
                </h3>
                <svg
                  viewBox="0 0 260 175"
                  className="w-full h-44"
                  role="img"
                  aria-label="Precision versus Recall chart"
                >
                  {/* Grid lines */}
                  {[0, 0.25, 0.5, 0.75, 1].map((t) => {
                    const y = 145 - t * 125;
                    const x = 35 + t * 205;
                    return (
                      <g key={t}>
                        <line x1="35" y1={y} x2="240" y2={y} stroke="#1E293B" strokeWidth="1" />
                        <line x1={x} y1="20" x2={x} y2="145" stroke="#1E293B" strokeWidth="1" />
                        <text x="28" y={y + 3} textAnchor="end" fill="#64748B" fontSize="9">
                          {(t * 100).toFixed(0)}
                        </text>
                        <text x={x} y="158" textAnchor="middle" fill="#64748B" fontSize="9">
                          {(t * 100).toFixed(0)}
                        </text>
                      </g>
                    );
                  })}
                  <text x="137" y="172" textAnchor="middle" fill="#94A3B8" fontSize="9">
                    Recall@{activeRun.k} (%)
                  </text>
                  <text
                    x="10"
                    y="85"
                    textAnchor="middle"
                    fill="#94A3B8"
                    fontSize="9"
                    transform="rotate(-90 10 85)"
                  >
                    Precision (%)
                  </text>
                  {completedReports.map((rep) => {
                    const cx = 35 + rep.avgRecallAtK * 205;
                    const cy = 145 - rep.avgPrecisionAtK * 125;
                    return (
                      <g key={rep.method}>
                        <circle
                          cx={cx}
                          cy={cy}
                          r="5"
                          fill={METHOD_COLORS[rep.method]}
                          stroke="#070B14"
                          strokeWidth="1.5"
                        />
                        <text
                          x={Math.min(230, cx + 7)}
                          y={Math.max(28, cy - 5)}
                          fill="#E6EDF7"
                          fontSize="9"
                        >
                          {rep.label.split(' ')[0]}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              {/* Chart 3: Latency Comparison */}
              <div className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-4 space-y-3">
                <h3 className="text-xs font-semibold text-white">Average Query Latency (ms)</h3>
                <div className="space-y-3">
                  {completedReports.map((rep) => (
                    <div key={rep.method} className="space-y-1 text-[11px]">
                      <div className="flex justify-between text-[#CBD5E1]">
                        <span>{rep.label}</span>
                        <span className="font-mono tabular-nums">{rep.avgLatencyMs} ms</span>
                      </div>
                      <div className="h-2 rounded bg-[#070B14] overflow-hidden">
                        <div
                          className="h-full rounded"
                          style={{
                            width: `${Math.max(4, Math.round((rep.avgLatencyMs / maxLatency) * 100))}%`,
                            backgroundColor: METHOD_COLORS[rep.method],
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Error Analysis Panel */}
          <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
            <h2 className="font-display text-sm font-semibold text-white">
              Automated Error Analysis
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
              <div className="p-3.5 rounded border border-[#1E293B] bg-[#070B14] space-y-1.5">
                <div className="text-[#94A3B8] font-medium">
                  Zero-Relevant Queries ({activeRun.errorAnalysis.zeroRelevantQueries.length})
                </div>
                {activeRun.errorAnalysis.zeroRelevantQueries.length === 0 ? (
                  <p className="text-[#64748B]">All evaluated queries surfaced at least one relevant document.</p>
                ) : (
                  <ul className="space-y-1 text-red-300">
                    {activeRun.errorAnalysis.zeroRelevantQueries.slice(0, 4).map((z, i) => (
                      <li key={i} className="truncate">
                        [{z.method}] {z.query}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="p-3.5 rounded border border-[#1E293B] bg-[#070B14] space-y-1.5">
                <div className="text-[#94A3B8] font-medium">
                  Poor Recall (&lt;50%) ({activeRun.errorAnalysis.poorRecallQueries.length})
                </div>
                {activeRun.errorAnalysis.poorRecallQueries.length === 0 ? (
                  <p className="text-[#64748B]">No queries fell below 50% Recall@{activeRun.k}.</p>
                ) : (
                  <ul className="space-y-1 text-amber-300">
                    {activeRun.errorAnalysis.poorRecallQueries.slice(0, 4).map((p, i) => (
                      <li key={i} className="truncate">
                        [{p.method}] {(p.recallAtK * 100).toFixed(0)}% · {p.query}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="p-3.5 rounded border border-[#1E293B] bg-[#070B14] space-y-1.5">
                <div className="text-[#94A3B8] font-medium">
                  Reranking Regressions ({activeRun.errorAnalysis.rerankingRegressions.length})
                </div>
                {activeRun.errorAnalysis.rerankingRegressions.length === 0 ? (
                  <p className="text-[#64748B]">No reranking regressions detected against Hybrid RRF.</p>
                ) : (
                  <ul className="space-y-1 text-amber-300">
                    {activeRun.errorAnalysis.rerankingRegressions.slice(0, 4).map((r, i) => (
                      <li key={i} className="truncate font-mono">
                        {r.delta.toFixed(3)} NDCG · {r.query}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {activeRun.errorAnalysis.keywordVsSemanticDivergence.length > 0 && (
              <div className="pt-3 border-t border-[#1E293B] space-y-2 text-xs">
                <div className="font-medium text-[#CBD5E1]">
                  Keyword (BM25) vs. Semantic (Dense) Per-Query Divergence
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="text-[#64748B] border-b border-[#1E293B]">
                        <th className="py-2 pr-4">Query</th>
                        <th className="py-2 px-3 text-right font-mono">BM25 NDCG</th>
                        <th className="py-2 px-3 text-right font-mono">Dense NDCG</th>
                        <th className="py-2 pl-3 text-right">Winner</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1E293B]">
                      {activeRun.errorAnalysis.keywordVsSemanticDivergence.map((row) => (
                        <tr key={row.queryId}>
                          <td className="py-2 pr-4 text-[#CBD5E1]">{row.query}</td>
                          <td className="py-2 px-3 text-right font-mono tabular-nums">
                            {row.bm25Ndcg.toFixed(4)}
                          </td>
                          <td className="py-2 px-3 text-right font-mono tabular-nums">
                            {row.denseNdcg.toFixed(4)}
                          </td>
                          <td className="py-2 pl-3 text-right font-mono uppercase text-[11px]">
                            <span
                              className={
                                row.winner === 'dense'
                                  ? 'text-emerald-400'
                                  : row.winner === 'bm25'
                                    ? 'text-[#60A5FA]'
                                    : 'text-[#64748B]'
                              }
                            >
                              {row.winner}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        </div>
      ) : (
        <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-8 text-center space-y-2">
          <p className="text-sm font-medium text-white">No evaluation experiments recorded yet</p>
          <p className="text-xs text-[#94A3B8] max-w-md mx-auto">
            Select retrieval methods above and click Run Evaluation to compute real Precision@K,
            Recall@K, MRR, MAP, and NDCG@K metrics.
          </p>
        </section>
      )}

      {/* Evaluation Dataset Queries Table */}
      <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] overflow-x-auto">
        <div className="px-5 py-3.5 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="font-display text-sm font-semibold text-white">
              Evaluation Dataset ({queries.length} queries)
            </h2>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              {humanCount} human-labelled · {demoCount} clearly labeled demonstration benchmark queries
            </p>
          </div>
        </div>

        {loading ? (
          <div className="p-6 text-xs text-[#94A3B8]">Loading queries...</div>
        ) : (
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#1E293B] text-[#64748B]">
                <th className="py-3 px-4">Query &amp; Origin</th>
                <th className="py-3 px-3">Category</th>
                <th className="py-3 px-3">Graded Relevance Judgments</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B]">
              {queries.map((q) => (
                <tr key={q.id} className="hover:bg-[#11192C] align-top">
                  <td className="py-3 px-4 max-w-md">
                    <div className="text-[#E6EDF7] font-medium">{q.query}</div>
                    <div className="text-[11px] text-[#94A3B8] mt-0.5">
                      <span className={q.isDemo ? 'text-amber-300/90' : 'text-emerald-400'}>
                        {q.isDemo ? 'Demonstration Dataset' : 'Human-Labelled'}
                      </span>
                      {q.notes ? ` · ${q.notes}` : ''}
                    </div>
                  </td>
                  <td className="py-3 px-3 text-[#94A3B8] whitespace-nowrap">{q.category}</td>
                  <td className="py-3 px-3 space-y-1">
                    {Object.entries(q.relevanceGrades || {}).map(([docId, grade]) => (
                      <div key={docId} className="text-[11px] text-[#CBD5E1]">
                        <span className="font-mono text-[#60A5FA] mr-1.5">G{grade}</span>
                        <span>{docTitleMap.get(docId) || docId}</span>
                      </div>
                    ))}
                  </td>
                  <td className="py-3 px-4 text-right whitespace-nowrap">
                    <div className="inline-flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => openEditForm(q)}
                        title="Edit relevance judgments"
                        className="p-1.5 rounded hover:bg-[#1B2640] text-[#94A3B8] hover:text-white"
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDeleteQuery(q.id)}
                        title="Delete test query"
                        className="p-1.5 rounded hover:bg-red-950/60 text-[#94A3B8] hover:text-red-400"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
