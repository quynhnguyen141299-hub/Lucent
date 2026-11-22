import React, { useEffect, useState } from 'react';
import { Loader2, RefreshCw, Trash2 } from 'lucide-react';
import { lucentApi } from '../services/api';
import type { AnalyticsSnapshot } from '../types/lucent';

export function AnalyticsView() {
  const [data, setData] = useState<AnalyticsSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await lucentApi.getAnalytics();
      setData(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load analytics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  const handleClearLogs = async () => {
    try {
      await lucentApi.clearAnalyticsLogs();
      await loadAnalytics();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to clear query logs.');
    }
  };

  if (loading && !data) {
    return (
      <div className="p-8 text-center text-sm text-[#94A3B8] flex items-center justify-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Loading real telemetry and corpus analytics...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-4 rounded-lg border border-red-500/40 bg-red-950/30 text-sm text-red-200">
        {error || 'Analytics unavailable.'}
      </div>
    );
  }

  const stageRows = [
    { label: 'Query Processing', ms: data.queries.avgStageLatencies.queryProcessingMs },
    { label: 'Query Classification', ms: data.queries.avgStageLatencies.queryClassificationMs },
    { label: 'Embedding Generation', ms: data.queries.avgStageLatencies.embeddingGenerationMs },
    { label: 'BM25 Lexical Retrieval', ms: data.queries.avgStageLatencies.bm25RetrievalMs },
    { label: 'Vector Similarity Search', ms: data.queries.avgStageLatencies.vectorRetrievalMs },
    { label: 'Hybrid RRF Fusion', ms: data.queries.avgStageLatencies.hybridFusionMs },
    { label: 'Second-Stage Reranking', ms: data.queries.avgStageLatencies.rerankingMs },
    { label: 'RAG Answer Synthesis', ms: data.queries.avgStageLatencies.answerGenerationMs },
  ];
  const maxStage = Math.max(1, ...stageRows.map((r) => r.ms));
  const series = data.queries.latencySeries;
  const maxSeriesMs = Math.max(50, ...series.map((s) => s.totalLatencyMs));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-semibold text-white">
            Search &amp; Corpus Analytics
          </h1>
          <p className="text-sm text-[#94A3B8] mt-1">
            Observed telemetry recorded strictly from live searches and indexed documents — zero
            fabricated metrics.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {data.queries.totalQueries > 0 && (
            <button
              type="button"
              onClick={handleClearLogs}
              className="h-9 px-3 rounded-md border border-[#1E293B] bg-[#0D1321] hover:bg-red-950/40 text-xs text-[#94A3B8] hover:text-red-300 flex items-center gap-1.5"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear Query History</span>
            </button>
          )}
          <button
            type="button"
            onClick={loadAnalytics}
            className="h-9 px-3.5 rounded-md border border-[#24324D] bg-[#0D1321] hover:bg-[#131B2E] text-xs font-medium text-[#E6EDF7] flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Primary KPI Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 border-y border-[#1E293B] py-4">
        <div>
          <div className="text-xs text-[#64748B]">Total Searches Logged</div>
          <div className="mt-1 font-mono text-2xl font-semibold text-white tabular-nums">
            {data.queries.totalQueries}
          </div>
          <div className="text-[11px] text-[#94A3B8] mt-0.5 tabular-nums">
            {data.queries.failedQueries} failed · {data.queries.zeroResultCount} empty results
          </div>
        </div>
        <div>
          <div className="text-xs text-[#64748B]">Avg End-to-End Latency</div>
          <div className="mt-1 font-mono text-2xl font-semibold text-white tabular-nums">
            {data.queries.totalQueries > 0 ? `${data.queries.avgLatencyMs} ms` : '—'}
          </div>
          <div className="text-[11px] text-[#94A3B8] mt-0.5">Across executed queries</div>
        </div>
        <div>
          <div className="text-xs text-[#64748B]">Corpus Volume</div>
          <div className="mt-1 font-mono text-2xl font-semibold text-white tabular-nums">
            {data.corpus.totalDocuments} docs
          </div>
          <div className="text-[11px] text-[#94A3B8] mt-0.5 tabular-nums">
            {data.corpus.totalWords.toLocaleString()} words · {data.corpus.totalChunks} chunks
          </div>
        </div>
        <div>
          <div className="text-xs text-[#64748B]">Vector Coverage</div>
          <div className="mt-1 font-mono text-2xl font-semibold text-white tabular-nums">
            {data.corpus.totalChunks > 0
              ? `${Math.round((data.corpus.embeddedChunks / data.corpus.totalChunks) * 100)}%`
              : '0%'}
          </div>
          <div className="text-[11px] text-[#94A3B8] mt-0.5 tabular-nums">
            {data.corpus.embeddedChunks} / {data.corpus.totalChunks} embedded chunks
          </div>
        </div>
      </div>

      {data.queries.totalQueries === 0 ? (
        <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-10 text-center space-y-2">
          <p className="text-sm font-medium text-white">No search telemetry recorded yet</p>
          <p className="text-xs text-[#94A3B8] max-w-md mx-auto">
            Execute queries in the Search workspace to populate latency breakdowns, query category
            distributions, mode usage, and performance trends over time.
          </p>
        </section>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Stage Latency Breakdown */}
            <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
              <h2 className="font-display text-sm font-semibold text-white">
                Average Stage Latency Breakdown (ms)
              </h2>
              <div className="space-y-3">
                {stageRows.map((row) => (
                  <div key={row.label} className="space-y-1 text-xs">
                    <div className="flex justify-between text-[#CBD5E1]">
                      <span>{row.label}</span>
                      <span className="font-mono tabular-nums">{row.ms} ms</span>
                    </div>
                    <div className="h-1.5 rounded bg-[#070B14] overflow-hidden">
                      <div
                        className="h-full rounded bg-[#3B82F6]"
                        style={{
                          width: `${row.ms > 0 ? Math.max(3, Math.round((row.ms / maxStage) * 100)) : 0}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Latency Over Time SVG Trend */}
            <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-4">
              <h2 className="font-display text-sm font-semibold text-white">
                End-to-End Search Latency Over Recent Queries
              </h2>
              {series.length < 2 ? (
                <p className="text-xs text-[#94A3B8] py-8">
                  Run at least 2 searches to plot latency trends over time.
                </p>
              ) : (
                <svg
                  viewBox="0 0 420 180"
                  className="w-full h-48"
                  role="img"
                  aria-label="Search latency over time chart"
                >
                  {[0, 0.5, 1].map((t) => {
                    const y = 150 - t * 130;
                    return (
                      <g key={t}>
                        <line x1="42" y1={y} x2="405" y2={y} stroke="#1E293B" strokeWidth="1" />
                        <text x="36" y={y + 3} textAnchor="end" fill="#64748B" fontSize="9">
                          {Math.round(t * maxSeriesMs)}
                        </text>
                      </g>
                    );
                  })}
                  <polyline
                    fill="none"
                    stroke="#3B82F6"
                    strokeWidth="2"
                    points={series
                      .map((pt, idx) => {
                        const x = 42 + (idx / Math.max(1, series.length - 1)) * 360;
                        const y = 150 - (pt.totalLatencyMs / maxSeriesMs) * 130;
                        return `${x},${y}`;
                      })
                      .join(' ')}
                  />
                  {series.map((pt, idx) => {
                    const x = 42 + (idx / Math.max(1, series.length - 1)) * 360;
                    const y = 150 - (pt.totalLatencyMs / maxSeriesMs) * 130;
                    return (
                      <circle key={pt.index} cx={x} cy={y} r="3.5" fill="#8B5CF6">
                        <title>{`#${pt.index} (${pt.mode}): ${pt.totalLatencyMs}ms`}</title>
                      </circle>
                    );
                  })}
                </svg>
              )}

              {/* Mode & Category Summary */}
              <div className="grid grid-cols-2 gap-4 pt-3 border-t border-[#1E293B] text-xs">
                <div className="space-y-1.5">
                  <div className="text-[#64748B] font-medium">Search Mode Usage</div>
                  {data.queries.modeBreakdown.map((m) => (
                    <div key={m.mode} className="flex justify-between">
                      <span className="capitalize text-[#CBD5E1]">{m.mode}</span>
                      <span className="font-mono tabular-nums text-[#94A3B8]">
                        {m.count} ({m.avgLatencyMs}ms)
                      </span>
                    </div>
                  ))}
                </div>
                <div className="space-y-1.5">
                  <div className="text-[#64748B] font-medium">Query Categories</div>
                  {data.queries.categoryBreakdown.map((c) => (
                    <div key={c.category} className="flex justify-between">
                      <span className="capitalize text-[#CBD5E1]">{c.category}</span>
                      <span className="font-mono tabular-nums text-[#94A3B8]">{c.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </div>

          {/* Recent Query Log Table */}
          <section className="rounded-lg border border-[#1E293B] bg-[#0D1321] overflow-x-auto">
            <div className="px-5 py-3.5 border-b border-[#1E293B]">
              <h2 className="font-display text-sm font-semibold text-white">Recent Search Log</h2>
            </div>
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-[#1E293B] text-[#64748B]">
                  <th className="py-3 px-4">Query</th>
                  <th className="py-3 px-3">Mode / Reranker</th>
                  <th className="py-3 px-3">Category</th>
                  <th className="py-3 px-3 text-right font-mono">Results</th>
                  <th className="py-3 px-3 text-right font-mono">Latency</th>
                  <th className="py-3 px-4 text-right font-mono">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E293B]">
                {data.queries.recentQueries.map((log) => (
                  <tr key={log.id} className="hover:bg-[#11192C]">
                    <td className="py-2.5 px-4 max-w-md">
                      <div className="text-[#E6EDF7] truncate">{log.query}</div>
                      {log.failed && log.errorMessage && (
                        <div className="text-[11px] text-red-300 truncate">{log.errorMessage}</div>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-[#94A3B8] whitespace-nowrap">
                      <span className="capitalize">{log.mode}</span>
                      {log.rerankerMode !== 'none' ? ` + ${log.rerankerMode}` : ''}
                    </td>
                    <td className="py-2.5 px-3 capitalize text-[#94A3B8]">
                      {log.queryCategory || '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                      {log.failed ? (
                        <span className="text-red-400">Failed</span>
                      ) : (
                        <span className={log.resultCount === 0 ? 'text-amber-400' : 'text-[#E6EDF7]'}>
                          {log.resultCount}
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-[#94A3B8]">
                      {log.totalLatencyMs} ms
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#64748B] whitespace-nowrap">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}

      {/* Corpus Composition Breakdown */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-3 text-xs">
          <h2 className="font-display text-sm font-semibold text-white">Documents by Type</h2>
          {data.corpus.byType.map((row) => (
            <div key={row.type} className="flex justify-between py-1 border-b border-[#1E293B]/60">
              <span className="text-[#CBD5E1]">{row.type}</span>
              <span className="font-mono tabular-nums text-[#94A3B8]">{row.count}</span>
            </div>
          ))}
        </div>

        <div className="rounded-lg border border-[#1E293B] bg-[#0D1321] p-5 space-y-3 text-xs">
          <h2 className="font-display text-sm font-semibold text-white">Documents by Source</h2>
          {data.corpus.bySource.map((row) => (
            <div key={row.source} className="flex justify-between py-1 border-b border-[#1E293B]/60 gap-4">
              <span className="text-[#CBD5E1] truncate">{row.source}</span>
              <span className="font-mono tabular-nums text-[#94A3B8] shrink-0">{row.count}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
