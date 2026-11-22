import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  ChevronDown,
  ChevronUp,
  Columns2,
  Download,
  ExternalLink,
  FileText,
  History,
  Loader2,
  Search,
  SlidersHorizontal,
} from 'lucide-react';
import { lucentApi } from '../services/api';
import type {
  AnswerStyle,
  EngineSettings,
  QueryLogEntry,
  RerankerMode,
  ResearchDocument,
  SearchComparisonResponse,
  SearchFilters,
  SearchMode,
  SearchResponsePayload,
  SearchResultItem,
} from '../types/lucent';

const EXAMPLE_QUESTIONS = [
  'How does Reciprocal Rank Fusion combine BM25 and dense semantic rankings?',
  'What causes hallucinations in RAG pipelines and how do chunking and citations mitigate them?',
  'Compare iron-air batteries and lithium-ion storage economics for multi-day renewable grids.',
  'How do product quantization and asymmetric rescoring reduce vector index memory?',
];

const DOCUMENT_TYPES = [
  'all',
  'Research Paper',
  'Technical Report',
  'Policy Brief',
  'Clinical Study',
  'Dataset Notes',
  'Uploaded Document',
];

interface SearchViewProps {
  settings: EngineSettings | null;
  onInspectDocument: (documentId: string, chunkId?: string) => void;
}

export function SearchView({ settings, onInspectDocument }: SearchViewProps) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<SearchMode>(settings?.defaultMode || 'hybrid');
  const [rerankerMode, setRerankerMode] = useState<RerankerMode>(
    settings?.defaultRerankerMode || 'none'
  );
  const [answerStyle, setAnswerStyle] = useState<AnswerStyle>(
    settings?.defaultAnswerStyle || 'detailed'
  );
  const [generateAnswer, setGenerateAnswer] = useState<boolean>(
    settings?.enableRagByDefault ?? true
  );

  // Filters & search parameter overrides
  const [showFilters, setShowFilters] = useState(false);
  const [docTypeFilter, setDocTypeFilter] = useState('all');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [includeDemo, setIncludeDemo] = useState(true);
  const [bm25Candidates, setBm25Candidates] = useState(settings?.bm25Candidates || 25);
  const [semanticCandidates, setSemanticCandidates] = useState(settings?.semanticCandidates || 25);
  const [hybridCandidates, setHybridCandidates] = useState(settings?.hybridCandidates || 10);
  const [bm25Weight, setBm25Weight] = useState(settings?.bm25Weight ?? 1.0);
  const [semanticWeight, setSemanticWeight] = useState(settings?.semanticWeight ?? 1.0);
  const [rrfK, setRrfK] = useState(settings?.rrfK || 60);

  // Corpus metadata & history
  const [documentsMeta, setDocumentsMeta] = useState<Array<Omit<ResearchDocument, 'content'>>>([]);
  const [recentHistory, setRecentHistory] = useState<QueryLogEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Search state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponsePayload | null>(null);
  const [expandedExplain, setExpandedExplain] = useState<Record<string, boolean>>({});
  const [highlightedRank, setHighlightedRank] = useState<number | null>(null);
  const [answerTab, setAnswerTab] = useState<'synthesis' | 'evidence'>('synthesis');

  // Mode comparison drawer
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<SearchComparisonResponse | null>(null);
  const [compareIncludeCE, setCompareIncludeCE] = useState(false);
  const [compareIncludeLlm, setCompareIncludeLlm] = useState(false);

  useEffect(() => {
    if (settings) {
      setMode(settings.defaultMode);
      setRerankerMode(settings.defaultRerankerMode);
      setAnswerStyle(settings.defaultAnswerStyle);
      setGenerateAnswer(settings.enableRagByDefault);
      setBm25Candidates(settings.bm25Candidates);
      setSemanticCandidates(settings.semanticCandidates);
      setHybridCandidates(settings.hybridCandidates);
      setBm25Weight(settings.bm25Weight);
      setSemanticWeight(settings.semanticWeight);
      setRrfK(settings.rrfK);
    }
  }, [settings]);

  const loadSidebarMetadata = () => {
    lucentApi
      .listDocuments()
      .then((res) => setDocumentsMeta(res.documents))
      .catch(() => {});
    lucentApi
      .getAnalytics()
      .then((res) => setRecentHistory(res.queries.recentQueries.slice(0, 8)))
      .catch(() => {});
  };

  useEffect(() => {
    loadSidebarMetadata();
  }, []);

  const uniqueSources = useMemo(() => {
    const set = new Set<string>();
    for (const d of documentsMeta) {
      if (d.source) set.add(d.source);
    }
    return Array.from(set).sort();
  }, [documentsMeta]);

  const buildFilters = (): SearchFilters => ({
    documentType: docTypeFilter !== 'all' ? docTypeFilter : undefined,
    source: sourceFilter !== 'all' ? sourceFilter : undefined,
    dateFrom: dateFrom || undefined,
    dateTo: dateTo || undefined,
    includeDemo,
  });

  const executeSearch = async (overrideQuery?: string, overrideMode?: SearchMode) => {
    const q = (overrideQuery ?? query).trim();
    if (!q) return;
    if (overrideQuery !== undefined) setQuery(overrideQuery);
    const targetMode = overrideMode ?? mode;

    setLoading(true);
    setError(null);
    setHighlightedRank(null);

    try {
      const res = await lucentApi.search({
        query: q,
        mode: targetMode,
        rerankerMode,
        answerStyle,
        filters: buildFilters(),
        bm25Candidates,
        semanticCandidates,
        hybridCandidates,
        bm25Weight,
        semanticWeight,
        rrfK,
        generateAnswer,
      });
      setResponse(res);
      setExpandedExplain({});
      loadSidebarMetadata();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Search request failed.');
    } finally {
      setLoading(false);
    }
  };

  const executeComparison = async () => {
    const q = query.trim();
    if (!q) return;
    setComparing(true);
    setCompareError(null);
    try {
      const res = await lucentApi.compareSearchModes({
        query: q,
        filters: buildFilters(),
        topK: 5,
        includeCrossEncoder: compareIncludeCE,
        includeLlmReranker: compareIncludeLlm,
      });
      setComparison(res);
    } catch (err: unknown) {
      setCompareError(err instanceof Error ? err.message : 'Model comparison failed.');
    } finally {
      setComparing(false);
    }
  };

  const handleCitationClick = (citationIndex: number) => {
    setHighlightedRank(citationIndex);
    const el = document.getElementById(`search-result-rank-${citationIndex}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const renderInlineCitations = (text: string) => {
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((part, idx) => {
      const match = /^\[(\d+)\]$/.exec(part);
      if (match) {
        const num = Number(match[1]);
        return (
          <button
            key={idx}
            type="button"
            onClick={() => handleCitationClick(num)}
            className="mx-0.5 px-1 py-0.5 text-[11px] font-mono font-medium text-[#60A5FA] hover:text-white bg-[#131B2E] hover:bg-[#3B82F6] rounded transition-colors align-baseline"
            title={`Jump to retrieved source [${num}]`}
          >
            [{num}]
          </button>
        );
      }
      return <React.Fragment key={idx}>{part}</React.Fragment>;
    });
  };

  const toggleExplain = (docId: string) => {
    setExpandedExplain((prev) => ({ ...prev, [docId]: !prev[docId] }));
  };

  const activeFilterCount =
    (docTypeFilter !== 'all' ? 1 : 0) +
    (sourceFilter !== 'all' ? 1 : 0) +
    (dateFrom ? 1 : 0) +
    (dateTo ? 1 : 0) +
    (!includeDemo ? 1 : 0);

  return (
    <div className="space-y-6">
      {/* Central Search Header & Input */}
      <section className="max-w-4xl mx-auto space-y-5">
        {!response && (
          <div className="pt-4 pb-1 text-center space-y-2">
            <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight text-white">
              Find what matters. Understand why.
            </h1>
            <p className="text-sm sm:text-base text-[#94A3B8] max-w-2xl mx-auto">
              Hybrid lexical and dense semantic search with transparent ranking signals, optional
              second-stage neural reranking, and citation-verified research synthesis.
            </p>
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            executeSearch();
          }}
          className="space-y-3"
        >
          <div className="flex flex-col sm:flex-row gap-2">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-[#64748B] absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a natural-language research question or search technical terms..."
                aria-label="Research search query"
                className="w-full h-12 pl-11 pr-4 rounded-lg bg-[#0D1321] border border-[#24324D] text-sm sm:text-base text-white placeholder-[#64748B] focus:border-[#3B82F6] focus:outline-none transition-colors"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="h-12 px-6 rounded-lg bg-[#3B82F6] hover:bg-[#2563EB] disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors whitespace-nowrap shrink-0 flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              <span>{loading ? 'Searching' : 'Search'}</span>
            </button>
          </div>

          {/* Control Strip: Retrieval Mode, Reranker, Answer Format, Filters, Compare, History */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <div className="flex flex-wrap items-center gap-2">
              {/* Mode Selector */}
              <div
                role="group"
                aria-label="Search retrieval mode"
                className="inline-flex items-center p-1 rounded-md bg-[#0D1321] border border-[#1E293B]"
              >
                {(['keyword', 'semantic', 'hybrid'] as SearchMode[]).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => {
                      setMode(m);
                      if (response && query.trim()) {
                        executeSearch(query, m);
                      }
                    }}
                    className={`px-3 py-1.5 text-xs font-medium rounded transition-colors capitalize whitespace-nowrap ${
                      mode === m
                        ? 'bg-[#3B82F6] text-white'
                        : 'text-[#94A3B8] hover:text-white'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>

              {/* Second-Stage Reranker Selector */}
              <select
                aria-label="Second-stage reranker"
                value={rerankerMode}
                onChange={(e) => setRerankerMode(e.target.value as RerankerMode)}
                className="h-9 px-3 rounded-md bg-[#0D1321] border border-[#1E293B] text-xs text-[#E6EDF7] focus:border-[#3B82F6] focus:outline-none"
              >
                <option value="none">Reranker: Off</option>
                <option value="cross_encoder">Reranker: Cross-Encoder (Python)</option>
                <option value="llm">Reranker: LLM-Based (Gemini)</option>
              </select>

              {/* Answer Style Selector */}
              <select
                aria-label="AI answer format"
                value={generateAnswer ? answerStyle : 'off'}
                onChange={(e) => {
                  if (e.target.value === 'off') {
                    setGenerateAnswer(false);
                  } else {
                    setGenerateAnswer(true);
                    setAnswerStyle(e.target.value as AnswerStyle);
                  }
                }}
                className="h-9 px-3 rounded-md bg-[#0D1321] border border-[#1E293B] text-xs text-[#E6EDF7] focus:border-[#3B82F6] focus:outline-none"
              >
                <option value="concise">Answer: Concise</option>
                <option value="detailed">Answer: Detailed Research</option>
                <option value="bullets">Answer: Bullet Summary</option>
                <option value="comparative">Answer: Comparative Analysis</option>
                <option value="off">Answer: Disabled</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowFilters((v) => !v)}
                className={`h-9 px-3 rounded-md border text-xs font-medium transition-colors flex items-center gap-1.5 whitespace-nowrap ${
                  showFilters || activeFilterCount > 0
                    ? 'border-[#3B82F6]/60 bg-[#101A33] text-white'
                    : 'border-[#1E293B] bg-[#0D1321] text-[#94A3B8] hover:text-white'
                }`}
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
                <span>Filters &amp; Weights{activeFilterCount > 0 ? ` (${activeFilterCount})` : ''}</span>
              </button>

              <button
                type="button"
                disabled={!query.trim() || comparing}
                onClick={executeComparison}
                className="h-9 px-3 rounded-md border border-[#1E293B] bg-[#0D1321] hover:bg-[#131B2E] disabled:opacity-40 text-xs font-medium text-[#CBD5E1] transition-colors flex items-center gap-1.5 whitespace-nowrap"
              >
                {comparing ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Columns2 className="w-3.5 h-3.5 text-[#8B5CF6]" />
                )}
                <span>Compare Models</span>
              </button>

              {recentHistory.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowHistory((v) => !v)}
                  className="h-9 px-3 rounded-md border border-[#1E293B] bg-[#0D1321] hover:bg-[#131B2E] text-xs font-medium text-[#94A3B8] hover:text-white transition-colors flex items-center gap-1.5 whitespace-nowrap"
                >
                  <History className="w-3.5 h-3.5" />
                  <span>History</span>
                </button>
              )}
            </div>
          </div>

          {/* Collapsible Filters & Retrieval Parameters */}
          {showFilters && (
            <div className="p-4 rounded-lg border border-[#1E293B] bg-[#0D1321] space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">Document Type</span>
                  <select
                    value={docTypeFilter}
                    onChange={(e) => setDocTypeFilter(e.target.value)}
                    className="w-full h-9 px-2.5 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                  >
                    {DOCUMENT_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t === 'all' ? 'All Document Types' : t}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">Source</span>
                  <select
                    value={sourceFilter}
                    onChange={(e) => setSourceFilter(e.target.value)}
                    className="w-full h-9 px-2.5 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7]"
                  >
                    <option value="all">All Sources</option>
                    {uniqueSources.map((src) => (
                      <option key={src} value={src}>
                        {src}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">Published From</span>
                  <input
                    type="date"
                    value={dateFrom}
                    onChange={(e) => setDateFrom(e.target.value)}
                    className="w-full h-9 px-2.5 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono"
                  />
                </label>

                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">Published To</span>
                  <input
                    type="date"
                    value={dateTo}
                    onChange={(e) => setDateTo(e.target.value)}
                    className="w-full h-9 px-2.5 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono"
                  />
                </label>
              </div>

              <div className="pt-3 border-t border-[#1E293B] grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">BM25 Candidates</span>
                  <input
                    type="number"
                    min={5}
                    max={100}
                    value={bm25Candidates}
                    onChange={(e) => setBm25Candidates(Number(e.target.value) || 25)}
                    className="w-full h-8 px-2 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono tabular-nums"
                  />
                </label>
                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">Semantic Candidates</span>
                  <input
                    type="number"
                    min={5}
                    max={100}
                    value={semanticCandidates}
                    onChange={(e) => setSemanticCandidates(Number(e.target.value) || 25)}
                    className="w-full h-8 px-2 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono tabular-nums"
                  />
                </label>
                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">Hybrid Candidates</span>
                  <input
                    type="number"
                    min={3}
                    max={50}
                    value={hybridCandidates}
                    onChange={(e) => setHybridCandidates(Number(e.target.value) || 10)}
                    className="w-full h-8 px-2 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono tabular-nums"
                  />
                </label>
                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">BM25 Weight</span>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={5}
                    value={bm25Weight}
                    onChange={(e) => setBm25Weight(Number(e.target.value))}
                    className="w-full h-8 px-2 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono tabular-nums"
                  />
                </label>
                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">Semantic Weight</span>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={5}
                    value={semanticWeight}
                    onChange={(e) => setSemanticWeight(Number(e.target.value))}
                    className="w-full h-8 px-2 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono tabular-nums"
                  />
                </label>
                <label className="block space-y-1 text-xs">
                  <span className="text-[#94A3B8]">RRF Constant k</span>
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={rrfK}
                    onChange={(e) => setRrfK(Number(e.target.value) || 60)}
                    className="w-full h-8 px-2 rounded bg-[#070B14] border border-[#1E293B] text-[#E6EDF7] font-mono tabular-nums"
                  />
                </label>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-[#1E293B] text-xs">
                <label className="inline-flex items-center gap-2 text-[#CBD5E1] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeDemo}
                    onChange={(e) => setIncludeDemo(e.target.checked)}
                    className="rounded border-[#334155] bg-[#070B14]"
                  />
                  <span>Include clearly labeled demonstration documents</span>
                </label>

                <div className="flex items-center gap-4 text-[#94A3B8]">
                  <label className="inline-flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={compareIncludeCE}
                      onChange={(e) => setCompareIncludeCE(e.target.checked)}
                    />
                    <span>Compare Cross-Encoder</span>
                  </label>
                  <label className="inline-flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={compareIncludeLlm}
                      onChange={(e) => setCompareIncludeLlm(e.target.checked)}
                    />
                    <span>Compare LLM Reranker</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Recent Search History Popover Strip */}
          {showHistory && recentHistory.length > 0 && (
            <div className="p-3 rounded-lg border border-[#1E293B] bg-[#0D1321] space-y-1.5">
              <div className="text-xs font-medium text-[#94A3B8] px-1">Recent Searches</div>
              <div className="divide-y divide-[#1E293B]">
                {recentHistory.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setShowHistory(false);
                      executeSearch(item.query, item.mode);
                    }}
                    className="w-full text-left px-2 py-2 hover:bg-[#131B2E] rounded flex items-center justify-between gap-4 text-xs"
                  >
                    <span className="text-[#E6EDF7] truncate">{item.query}</span>
                    <span className="text-[#64748B] font-mono tabular-nums whitespace-nowrap shrink-0">
                      {item.mode} · {item.resultCount} docs · {item.totalLatencyMs}ms
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </form>

        {/* Empty Library Guidance or Example Questions (shown prior to first query) */}
        {!response && !loading && (
          documentsMeta.length === 0 ? (
            <div className="pt-2">
              <div className="p-5 rounded-lg border border-[#1E293B] bg-[#0D1321] text-center space-y-1.5">
                <p className="text-sm font-medium text-white">No documents indexed yet</p>
                <p className="text-xs text-[#94A3B8]">
                  Upload PDF, DOCX, TXT, or Markdown files in the Document Library to start searching
                  and generating citation-backed answers.
                </p>
              </div>
            </div>
          ) : documentsMeta.some((d) => d.isDemo) ? (
            <div className="pt-2 space-y-2.5">
              <p className="text-xs text-[#64748B] text-center">
                Example research questions
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => executeSearch(q)}
                    className="text-left p-3.5 rounded-lg border border-[#1E293B] bg-[#0D1321] hover:border-[#3B82F6]/60 hover:bg-[#11192C] transition-colors text-sm text-[#CBD5E1] leading-snug"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : null
        )}
      </section>

      {/* Error Banner */}
      {error && (
        <div
          role="alert"
          className="max-w-4xl mx-auto p-4 rounded-lg border border-red-500/40 bg-red-950/30 flex items-start gap-3 text-sm text-red-200"
        >
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-medium text-red-100">Search could not complete</p>
            <p className="text-xs text-red-200/90 leading-relaxed">{error}</p>
          </div>
        </div>
      )}

      {/* Warnings Banner */}
      {response && response.warnings.length > 0 && (
        <div className="p-3.5 rounded-lg border border-amber-500/30 bg-amber-950/20 text-xs text-amber-200 space-y-1">
          {response.warnings.map((w, i) => (
            <p key={i}>{w}</p>
          ))}
        </div>
      )}

      {/* Side-by-Side Model Comparison Panel */}
      {compareError && (
        <div className="p-3.5 rounded-lg border border-red-500/40 bg-red-950/30 text-xs text-red-200">
          Comparison error: {compareError}
        </div>
      )}

      {comparison && (
        <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#1E293B]">
            <div>
              <h2 className="font-display text-sm font-semibold text-white">
                Model Ranking Comparison
              </h2>
              <p className="text-xs text-[#94A3B8] mt-0.5">
                Same query and {comparison.corpusDocumentsScanned} filtered corpus documents evaluated across methods.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setComparison(null)}
              className="text-xs text-[#94A3B8] hover:text-white"
            >
              Close Comparison
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-3">
            {comparison.methods.map((m) => (
              <div
                key={m.method}
                className="rounded-md border border-[#1E293B] bg-[#070B14] p-3.5 space-y-3"
              >
                <div className="flex items-baseline justify-between gap-2 pb-2 border-b border-[#1E293B]">
                  <span className="text-xs font-semibold text-white truncate">{m.label}</span>
                  <span className="text-[11px] font-mono tabular-nums text-[#64748B] shrink-0">
                    {m.latencyMs}ms
                  </span>
                </div>

                {m.status === 'error' ? (
                  <p className="text-xs text-red-300 leading-relaxed">{m.errorMessage}</p>
                ) : m.results.length === 0 ? (
                  <p className="text-xs text-[#64748B]">No documents matched.</p>
                ) : (
                  <ol className="space-y-2.5">
                    {m.results.map((r) => (
                      <li key={r.documentId} className="text-xs space-y-0.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-[#64748B] tabular-nums">#{r.rank}</span>
                          <div className="flex items-center gap-1.5 font-mono tabular-nums text-[11px]">
                            {typeof r.rankMovement === 'number' && r.rankMovement !== 0 && (
                              <span
                                className={
                                  r.rankMovement > 0 ? 'text-emerald-400' : 'text-amber-400'
                                }
                              >
                                {r.rankMovement > 0 ? `+${r.rankMovement}` : r.rankMovement}
                              </span>
                            )}
                            <span className="text-[#94A3B8]">{r.score.toFixed(4)}</span>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => onInspectDocument(r.documentId)}
                          className="text-left text-[#CBD5E1] hover:text-[#60A5FA] line-clamp-2 leading-snug"
                        >
                          {r.title}
                        </button>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 space-y-4">
            {[1, 2, 3].map((n) => (
              <div
                key={n}
                className="p-5 rounded-lg border border-[#1E293B] bg-[#0D1321] space-y-3"
              >
                <div className="h-4 w-3/4 bg-[#131B2E] rounded animate-pulse" />
                <div className="h-3 w-1/2 bg-[#131B2E] rounded animate-pulse" />
                <div className="h-14 w-full bg-[#131B2E] rounded animate-pulse" />
              </div>
            ))}
          </div>
          <div className="lg:col-span-5">
            <div className="p-5 rounded-lg border border-[#1E293B] bg-[#0D1321] space-y-3">
              <div className="h-4 w-1/2 bg-[#131B2E] rounded animate-pulse" />
              <div className="h-24 w-full bg-[#131B2E] rounded animate-pulse" />
              <div className="h-16 w-full bg-[#131B2E] rounded animate-pulse" />
            </div>
          </div>
        </div>
      )}

      {/* Search Results + Separate AI Answer Panel */}
      {!loading && response && (
        <div className="space-y-4">
          {/* Execution Summary Line (Unboxed clean metadata) */}
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#1E293B] text-xs text-[#94A3B8]">
            <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
              <span className="text-[#E6EDF7] font-medium tabular-nums">
                {response.results.length} of {response.totalMatches} matching documents
              </span>
              <span aria-hidden="true">·</span>
              <span className="capitalize">{response.mode} retrieval</span>
              {response.reranking.status === 'executed' && (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="text-[#A78BFA]">
                    Reranked ({response.reranking.modelName}, {response.reranking.latencyMs}ms)
                  </span>
                </>
              )}
              {response.classification && (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="capitalize">
                    Intent: {response.classification.category}
                  </span>
                </>
              )}
            </div>
            <div className="font-mono tabular-nums text-[#64748B]">
              Total {response.trace.totalLatencyMs}ms · {response.trace.corpusChunksScanned} chunks scanned
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Retrieved Evidence Column */}
            <div className="lg:col-span-7 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-sm font-semibold text-white">
                  Retrieved Evidence
                </h2>
                <span className="text-xs text-[#64748B]">
                  Ranked passages directly from indexed corpus
                </span>
              </div>

              {response.results.length === 0 ? (
                <div className="p-8 rounded-lg border border-[#1E293B] bg-[#0D1321] text-center space-y-2">
                  <p className="text-sm font-medium text-white">No matching documents found</p>
                  <p className="text-xs text-[#94A3B8] max-w-md mx-auto">
                    Try switching to Semantic or Hybrid search, broadening your query phrasing, or
                    relaxing active document type/date filters.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {response.results.map((item: SearchResultItem) => {
                    const isExpanded = Boolean(expandedExplain[item.documentId]);
                    const isHighlighted = highlightedRank === item.rank;
                    const exp = item.explanation;

                    return (
                      <article
                        key={item.documentId}
                        id={`search-result-rank-${item.rank}`}
                        className={`p-5 rounded-lg border transition-colors ${
                          isHighlighted
                            ? 'border-[#3B82F6] bg-[#0F1930]'
                            : 'border-[#1E293B] bg-[#0D1321] hover:border-[#2B3B59]'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0 space-y-1">
                            <div className="flex items-center gap-2.5 text-xs text-[#94A3B8]">
                              <span className="font-mono font-medium text-[#60A5FA] tabular-nums">
                                [{item.rank}]
                              </span>
                              <span>{item.source || 'Source not specified'}</span>
                              {item.publishedDate && (
                                <>
                                  <span aria-hidden="true">·</span>
                                  <span className="tabular-nums">{item.publishedDate}</span>
                                </>
                              )}
                              {item.pageNumber !== null && (
                                <>
                                  <span aria-hidden="true">·</span>
                                  <span className="tabular-nums">p. {item.pageNumber}</span>
                                </>
                              )}
                              <span aria-hidden="true">·</span>
                              <span>{item.documentType}</span>
                              {item.isDemo && (
                                <>
                                  <span aria-hidden="true">·</span>
                                  <span className="text-amber-300/90">Demonstration Data</span>
                                </>
                              )}
                            </div>

                            <h3 className="font-display text-base font-semibold text-white leading-snug">
                              <button
                                type="button"
                                onClick={() => onInspectDocument(item.documentId, item.chunkId)}
                                className="text-left hover:text-[#60A5FA] transition-colors"
                              >
                                {item.title}
                              </button>
                            </h3>
                          </div>

                          <div className="text-right shrink-0">
                            <div className="font-mono text-sm font-medium text-white tabular-nums">
                              {item.relevanceScore.toFixed(4)}
                            </div>
                            <div className="text-[11px] text-[#64748B] flex items-center justify-end gap-1 mt-0.5">
                              <span>Score</span>
                              {exp.rerankProvider && exp.rankMovement !== 0 && (
                                <span
                                  className={`inline-flex items-center font-mono tabular-nums ${
                                    exp.rankMovement > 0 ? 'text-emerald-400' : 'text-amber-400'
                                  }`}
                                  title={`Moved from position #${exp.originalPosition} to #${exp.newPosition}`}
                                >
                                  {exp.rankMovement > 0 ? (
                                    <ArrowUp className="w-3 h-3" />
                                  ) : (
                                    <ArrowDown className="w-3 h-3" />
                                  )}
                                  {Math.abs(exp.rankMovement)}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>

                        <p className="mt-3 text-sm text-[#CBD5E1] leading-relaxed">{item.excerpt}</p>

                        <div className="mt-4 pt-3 border-t border-[#1E293B] flex flex-wrap items-center justify-between gap-3 text-xs">
                          <div className="flex flex-wrap items-center gap-4">
                            <button
                              type="button"
                              onClick={() => onInspectDocument(item.documentId, item.chunkId)}
                              className="text-[#60A5FA] hover:text-white flex items-center gap-1.5 transition-colors"
                            >
                              <FileText className="w-3.5 h-3.5" />
                              <span>Inspect Document</span>
                            </button>

                            <a
                              href={`/api/documents/${encodeURIComponent(item.documentId)}/download`}
                              className="text-[#94A3B8] hover:text-white flex items-center gap-1.5 transition-colors"
                            >
                              <Download className="w-3.5 h-3.5" />
                              <span>{item.hasUploadedFile ? 'Uploaded File' : 'Download Text'}</span>
                            </a>

                            {item.sourceUrl && (
                              <a
                                href={item.sourceUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[#94A3B8] hover:text-white flex items-center gap-1.5 transition-colors"
                              >
                                <ExternalLink className="w-3.5 h-3.5" />
                                <span>Original Link</span>
                              </a>
                            )}
                          </div>

                          <button
                            type="button"
                            onClick={() => toggleExplain(item.documentId)}
                            className="text-[#94A3B8] hover:text-white flex items-center gap-1 font-medium transition-colors"
                          >
                            <span>Why this was retrieved</span>
                            {isExpanded ? (
                              <ChevronUp className="w-3.5 h-3.5" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </div>

                        {/* Expandable Retrieval & Ranking Explanation */}
                        {isExpanded && (
                          <div className="mt-3 pt-3 border-t border-[#1E293B] space-y-3 text-xs">
                            <p className="text-[#CBD5E1] leading-relaxed">{exp.summary}</p>
                            {exp.rerankExplanation && (
                              <p className="text-[#C4B5FD] leading-relaxed">
                                {exp.rerankProvider === 'cross_encoder'
                                  ? 'Cross-Encoder Assessment: '
                                  : 'LLM Reranker Assessment: '}
                                {exp.rerankExplanation}
                              </p>
                            )}

                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-[#070B14] p-3 rounded border border-[#1E293B] font-mono tabular-nums">
                              <div>
                                <div className="text-[11px] text-[#64748B] font-sans">BM25 Score / Rank</div>
                                <div className="text-[#E6EDF7] mt-0.5">
                                  {exp.bm25Score !== null
                                    ? `${exp.bm25Score.toFixed(4)} (#${exp.bm25Rank})`
                                    : '—'}
                                </div>
                              </div>
                              <div>
                                <div className="text-[11px] text-[#64748B] font-sans">Semantic / Rank</div>
                                <div className="text-[#E6EDF7] mt-0.5">
                                  {exp.semanticScore !== null
                                    ? `${exp.semanticScore.toFixed(4)} (#${exp.semanticRank})`
                                    : '—'}
                                </div>
                              </div>
                              <div>
                                <div className="text-[11px] text-[#64748B] font-sans">RRF Score</div>
                                <div className="text-[#E6EDF7] mt-0.5">
                                  {exp.rrfScore !== null ? exp.rrfScore.toFixed(5) : '—'}
                                </div>
                              </div>
                              <div>
                                <div className="text-[11px] text-[#64748B] font-sans">Position / Chunk</div>
                                <div className="text-[#E6EDF7] mt-0.5">
                                  #{exp.originalPosition} → #{exp.newPosition} · c{exp.chunkIndex + 1}/{exp.totalChunksInDocument}
                                </div>
                              </div>
                            </div>

                            {exp.matchedTerms.length > 0 && (
                              <div className="text-[#94A3B8]">
                                Matched lexical terms:{' '}
                                <span className="font-mono text-[#E6EDF7]">
                                  {exp.matchedTerms.join(', ')}
                                </span>
                              </div>
                            )}
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Separate AI Answer Panel */}
            <aside className="lg:col-span-5 lg:sticky lg:top-20 rounded-lg border border-[#24324D] bg-[#0D1321] overflow-hidden">
              <div className="px-5 py-4 border-b border-[#1E293B] flex items-center justify-between gap-3">
                <div>
                  <h2 className="font-display text-sm font-semibold text-white">
                    AI-Generated Answer
                  </h2>
                  <p className="text-[11px] text-[#94A3B8] mt-0.5">
                    Generated synthesis strictly separated from retrieved evidence
                  </p>
                </div>

                <div className="inline-flex p-0.5 rounded bg-[#070B14] border border-[#1E293B] text-[11px]">
                  <button
                    type="button"
                    onClick={() => setAnswerTab('synthesis')}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      answerTab === 'synthesis'
                        ? 'bg-[#131B2E] text-white'
                        : 'text-[#94A3B8] hover:text-white'
                    }`}
                  >
                    Synthesis
                  </button>
                  <button
                    type="button"
                    onClick={() => setAnswerTab('evidence')}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      answerTab === 'evidence'
                        ? 'bg-[#131B2E] text-white'
                        : 'text-[#94A3B8] hover:text-white'
                    }`}
                  >
                    Evidence ({response.results.length})
                  </button>
                </div>
              </div>

              <div className="p-5 space-y-5">
                {answerTab === 'evidence' ? (
                  <div className="space-y-3">
                    <p className="text-xs text-[#94A3B8]">
                      Exact passages passed to the synthesis model (citations are validated strictly
                      against these chunk IDs):
                    </p>
                    {response.results.slice(0, 6).map((res) => (
                      <div
                        key={res.chunkId}
                        className="p-3 rounded border border-[#1E293B] bg-[#070B14] space-y-1.5 text-xs"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <button
                            type="button"
                            onClick={() => onInspectDocument(res.documentId, res.chunkId)}
                            className="font-medium text-[#60A5FA] hover:text-white text-left truncate"
                          >
                            [{res.rank}] {res.title}
                          </button>
                          {res.pageNumber !== null && (
                            <span className="font-mono text-[#64748B] shrink-0">
                              p. {res.pageNumber}
                            </span>
                          )}
                        </div>
                        <p className="text-[#CBD5E1] leading-relaxed line-clamp-4">{res.excerpt}</p>
                        <div className="font-mono text-[11px] text-[#64748B]">{res.chunkId}</div>
                      </div>
                    ))}
                  </div>
                ) : response.answer.status === 'skipped' ? (
                  <p className="text-sm text-[#94A3B8]">
                    AI answer generation is disabled for this query. Select an answer format above
                    to generate a citation-grounded response.
                  </p>
                ) : response.answer.status === 'error' ? (
                  <div className="p-3.5 rounded border border-red-500/30 bg-red-950/30 text-xs text-red-200 space-y-1">
                    <p className="font-medium text-red-100">Answer generation unavailable</p>
                    <p className="leading-relaxed">{response.answer.errorMessage}</p>
                  </div>
                ) : response.answer.status === 'no_evidence' ? (
                  <div className="space-y-2 text-sm text-[#94A3B8]">
                    <p>{response.answer.summary}</p>
                    {response.answer.evidenceGaps && (
                      <p className="text-xs text-[#64748B]">{response.answer.evidenceGaps}</p>
                    )}
                  </div>
                ) : (
                  <>
                    {/* Metadata line */}
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-[#94A3B8]">
                      <span className="text-[#A78BFA] font-medium">Generated Explanation</span>
                      <span aria-hidden="true">·</span>
                      <span className="capitalize">{response.answer.answerStyle}</span>
                      {response.answer.queryCategory && (
                        <>
                          <span aria-hidden="true">·</span>
                          <span className="capitalize">{response.answer.queryCategory} query</span>
                        </>
                      )}
                      {response.answer.model && (
                        <>
                          <span aria-hidden="true">·</span>
                          <span className="font-mono">{response.answer.model}</span>
                        </>
                      )}
                      <span aria-hidden="true">·</span>
                      <span className="font-mono tabular-nums">{response.answer.latencyMs}ms</span>
                    </div>

                    {/* Executive Summary */}
                    <p className="text-sm text-[#E6EDF7] leading-relaxed">
                      {renderInlineCitations(response.answer.summary)}
                    </p>

                    {/* Structured Findings (Fact vs Interpretation) */}
                    {response.answer.keyFindings.length > 0 && (
                      <div className="space-y-2.5 pt-3 border-t border-[#1E293B]">
                        <h3 className="text-xs font-semibold text-[#CBD5E1]">
                          Grounded Findings
                        </h3>
                        <ul className="space-y-2.5">
                          {response.answer.keyFindings.map((finding, idx) => (
                            <li key={idx} className="text-xs text-[#CBD5E1] leading-relaxed space-y-1">
                              <div className="flex items-baseline justify-between gap-2">
                                <span
                                  className={
                                    finding.evidenceType === 'fact'
                                      ? 'text-[#60A5FA] font-medium'
                                      : 'text-[#A78BFA] font-medium'
                                  }
                                >
                                  {finding.evidenceType === 'fact' ? 'Fact' : 'Interpretation'}
                                </span>
                                <span className="shrink-0">
                                  {finding.citationIndices.map((cIdx) => (
                                    <button
                                      key={cIdx}
                                      type="button"
                                      onClick={() => handleCitationClick(cIdx)}
                                      className="ml-1 px-1 py-0.5 text-[11px] font-mono text-[#60A5FA] hover:text-white bg-[#131B2E] hover:bg-[#3B82F6] rounded"
                                    >
                                      [{cIdx}]
                                    </button>
                                  ))}
                                </span>
                              </div>
                              <p>{finding.claim}</p>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Conflicting Evidence */}
                    {response.answer.conflictingEvidence && (
                      <div className="p-3 rounded border border-amber-500/30 bg-amber-950/20 space-y-1 text-xs">
                        <div className="font-medium text-amber-200">Conflicting Evidence Noted</div>
                        <p className="text-amber-100/90 leading-relaxed">
                          {response.answer.conflictingEvidence}
                        </p>
                      </div>
                    )}

                    {/* Evidence Gaps */}
                    {response.answer.evidenceGaps && (
                      <div className="pt-3 border-t border-[#1E293B] space-y-1 text-xs">
                        <div className="font-medium text-[#94A3B8]">Evidence Limitations &amp; Gaps</div>
                        <p className="text-[#94A3B8] leading-relaxed">
                          {response.answer.evidenceGaps}
                        </p>
                      </div>
                    )}

                    {/* Validated Sources Used */}
                    {response.answer.sourcesUsed.length > 0 && (
                      <div className="pt-3 border-t border-[#1E293B] space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <h3 className="font-semibold text-[#CBD5E1]">Sources Used</h3>
                          <span className="text-[11px] text-[#64748B]">
                            Validated against retrieved chunks
                          </span>
                        </div>
                        <ul className="space-y-2">
                          {response.answer.sourcesUsed.map((src) => (
                            <li
                              key={`${src.citationIndex}-${src.chunkId}`}
                              className="text-xs flex items-start justify-between gap-2"
                            >
                              <button
                                type="button"
                                onClick={() => onInspectDocument(src.documentId, src.chunkId)}
                                className="text-left group min-w-0"
                              >
                                <span className="font-mono text-[#60A5FA] mr-1.5">
                                  [{src.citationIndex}]
                                </span>
                                <span className="text-[#E6EDF7] group-hover:text-[#60A5FA] transition-colors">
                                  {src.title}
                                </span>
                                <span className="block text-[11px] text-[#64748B] pl-6 mt-0.5">
                                  {src.source || 'Source not specified'}
                                  {src.publishedDate ? ` · ${src.publishedDate}` : ''}
                                  {src.pageNumber !== null ? ` · p. ${src.pageNumber}` : ''}
                                  {src.isDemo ? ' · Demo' : ''}
                                </span>
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                )}
              </div>
            </aside>
          </div>
        </div>
      )}
    </div>
  );
}
