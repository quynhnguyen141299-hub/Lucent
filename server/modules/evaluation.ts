import crypto from 'crypto';
import { serverConfig } from '../config.ts';
import { normalizeGeminiError } from '../geminiClient.ts';
import { lucentStore } from '../store.ts';
import { rankCandidates } from './ranking.ts';
import { RerankerServiceError, runRerankingStage } from './reranking.ts';
import { runBm25Retrieval, runSemanticRetrieval } from './retrieval.ts';
import type {
  EvaluationErrorAnalysis,
  EvaluationMethodId,
  EvaluationModeReport,
  EvaluationQuery,
  EvaluationQueryResult,
  EvaluationRunRecord,
} from '../../src/types/lucent.ts';

/**
 * Pure IR evaluation metric functions exported for direct unit testing.
 */
export function computePrecisionAtK(
  retrieved: string[],
  relevanceGrades: Record<string, number>,
  k: number
): number {
  const safeK = Math.max(1, k);
  const top = retrieved.slice(0, safeK);
  if (top.length === 0) return 0;
  const hits = top.filter((id) => (relevanceGrades[id] || 0) > 0).length;
  return hits / safeK;
}

export function computeRecallAtK(
  retrieved: string[],
  relevanceGrades: Record<string, number>,
  k: number
): number {
  const relevantIds = Object.entries(relevanceGrades)
    .filter(([, grade]) => grade > 0)
    .map(([id]) => id);
  if (relevantIds.length === 0) return 0;
  const relevantSet = new Set(relevantIds);
  const top = retrieved.slice(0, Math.max(1, k));
  const hits = top.filter((id) => relevantSet.has(id)).length;
  return hits / relevantSet.size;
}

export function computeMrr(
  retrieved: string[],
  relevanceGrades: Record<string, number>
): number {
  for (let i = 0; i < retrieved.length; i++) {
    if ((relevanceGrades[retrieved[i]] || 0) > 0) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

export function computeAveragePrecision(
  retrieved: string[],
  relevanceGrades: Record<string, number>,
  k: number
): number {
  const totalRelevant = Object.values(relevanceGrades).filter((g) => g > 0).length;
  if (totalRelevant === 0) return 0;

  const top = retrieved.slice(0, Math.max(1, k));
  let hits = 0;
  let sumPrecision = 0;

  for (let i = 0; i < top.length; i++) {
    if ((relevanceGrades[top[i]] || 0) > 0) {
      hits++;
      sumPrecision += hits / (i + 1);
    }
  }

  return sumPrecision / totalRelevant;
}

/**
 * Graded NDCG@K using standard exponential gain: (2^rel - 1) / log2(rank + 1).
 */
export function computeGradedNdcgAtK(
  retrieved: string[],
  relevanceGrades: Record<string, number>,
  k: number
): number {
  const safeK = Math.max(1, k);
  const top = retrieved.slice(0, safeK);

  let dcg = 0;
  for (let i = 0; i < top.length; i++) {
    const rel = Math.max(0, relevanceGrades[top[i]] || 0);
    if (rel > 0) {
      dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
    }
  }

  const idealGrades = Object.values(relevanceGrades)
    .filter((g) => g > 0)
    .sort((a, b) => b - a)
    .slice(0, safeK);

  let idcg = 0;
  for (let i = 0; i < idealGrades.length; i++) {
    idcg += (Math.pow(2, idealGrades[i]) - 1) / Math.log2(i + 2);
  }

  return idcg === 0 ? 0 : dcg / idcg;
}

function roundMetric(n: number): number {
  return Math.round(n * 10000) / 10000;
}

const METHOD_META: Record<
  EvaluationMethodId,
  { label: string; modelName: (emb: string, gen: string) => string; version: string }
> = {
  bm25: {
    label: 'BM25 Lexical',
    modelName: () => 'Okapi BM25',
    version: 'sparse-v1',
  },
  dense: {
    label: 'Dense Retrieval',
    modelName: (emb) => emb,
    version: 'flatip-f32',
  },
  hybrid: {
    label: 'Hybrid RRF',
    modelName: (emb) => `BM25 + ${emb}`,
    version: 'rrf-v1',
  },
  hybrid_cross_encoder: {
    label: 'Hybrid + Cross-Encoder',
    modelName: () => 'MS MARCO Cross-Encoder Service',
    version: 'external-http',
  },
  hybrid_llm_rerank: {
    label: 'Hybrid + LLM Reranker',
    modelName: (_emb, gen) => `${gen} (LLM Reranker)`,
    version: 'structured-v1',
  },
};

function getGradesForQuery(q: EvaluationQuery): Record<string, number> {
  const grades: Record<string, number> = { ...(q.relevanceGrades || {}) };
  for (const id of q.relevantDocumentIds || []) {
    if (typeof grades[id] !== 'number') {
      grades[id] = 2;
    }
  }
  return grades;
}

async function evaluateSingleMethod(
  method: EvaluationMethodId,
  queries: EvaluationQuery[],
  k: number
): Promise<EvaluationModeReport> {
  const settings = lucentStore.getSettings();
  const docs = lucentStore.getDocuments();
  const docMap = new Map(docs.map((d) => [d.id, d]));
  const chunks = lucentStore.getChunks();
  const meta = METHOD_META[method];
  const embModel = settings.embeddingModel || serverConfig.embeddingModel;
  const genModel = serverConfig.generationModel;

  const perQuery: EvaluationQueryResult[] = [];
  let resolvedModelName = meta.modelName(embModel, genModel);

  try {
    for (const q of queries) {
      const qStart = performance.now();
      const needsLexical = method !== 'dense';
      const needsDense = method !== 'bm25';

      const lexical = needsLexical
        ? runBm25Retrieval(q.query, chunks, docMap, settings).candidates
        : [];
      const semantic = needsDense
        ? (await runSemanticRetrieval(q.query, chunks, docMap)).candidates
        : [];

      const baseMode =
        method === 'bm25' ? 'keyword' : method === 'dense' ? 'semantic' : 'hybrid';

      const candidateTopK =
        method === 'hybrid_cross_encoder' || method === 'hybrid_llm_rerank'
          ? Math.max(k, settings.rerankCandidateCount || 10)
          : k;

      const { results: baseResults } = rankCandidates({
        mode: baseMode,
        lexicalCandidates: lexical,
        semanticCandidates: semantic,
        settings,
        topK: candidateTopK,
      });

      let finalResults = baseResults.slice(0, k);
      let rankMovements: Array<{ documentId: string; fromRank: number; toRank: number; delta: number }> | undefined;

      if (method === 'hybrid_cross_encoder' || method === 'hybrid_llm_rerank') {
        const rerankerMode = method === 'hybrid_cross_encoder' ? 'cross_encoder' : 'llm';
        const reranked = await runRerankingStage({
          query: q.query,
          mode: rerankerMode,
          candidates: baseResults,
          settings,
          finalTopK: k,
        });
        finalResults = reranked.results;
        if (reranked.metrics.modelName) {
          resolvedModelName = reranked.metrics.modelName;
        }
        rankMovements = finalResults.map((r) => ({
          documentId: r.documentId,
          fromRank: r.explanation.originalPosition,
          toRank: r.explanation.newPosition,
          delta: r.explanation.rankMovement,
        }));
      }

      const retrievedIds = finalResults.map((r) => r.documentId);
      const grades = getGradesForQuery(q);

      perQuery.push({
        queryId: q.id,
        query: q.query,
        relevantDocumentIds: Object.entries(grades)
          .filter(([, g]) => g > 0)
          .map(([id]) => id),
        relevanceGrades: grades,
        retrievedDocumentIds: retrievedIds,
        precisionAtK: roundMetric(computePrecisionAtK(retrievedIds, grades, k)),
        recallAtK: roundMetric(computeRecallAtK(retrievedIds, grades, k)),
        mrr: roundMetric(computeMrr(retrievedIds, grades)),
        averagePrecision: roundMetric(computeAveragePrecision(retrievedIds, grades, k)),
        ndcgAtK: roundMetric(computeGradedNdcgAtK(retrievedIds, grades, k)),
        latencyMs: Math.max(1, Math.round(performance.now() - qStart)),
        rankMovements,
      });
    }

    const n = Math.max(1, perQuery.length);
    return {
      method,
      label: meta.label,
      modelName: resolvedModelName,
      modelVersion: meta.version,
      status: 'completed',
      k,
      avgPrecisionAtK: roundMetric(perQuery.reduce((s, r) => s + r.precisionAtK, 0) / n),
      avgRecallAtK: roundMetric(perQuery.reduce((s, r) => s + r.recallAtK, 0) / n),
      meanReciprocalRank: roundMetric(perQuery.reduce((s, r) => s + r.mrr, 0) / n),
      meanAveragePrecision: roundMetric(perQuery.reduce((s, r) => s + r.averagePrecision, 0) / n),
      avgNdcgAtK: roundMetric(perQuery.reduce((s, r) => s + r.ndcgAtK, 0) / n),
      avgLatencyMs: Math.round(perQuery.reduce((s, r) => s + r.latencyMs, 0) / n),
      perQuery,
    };
  } catch (err) {
    const message =
      err instanceof RerankerServiceError
        ? err.message
        : normalizeGeminiError(err).message;
    return {
      method,
      label: meta.label,
      modelName: resolvedModelName,
      modelVersion: meta.version,
      status: 'error',
      errorMessage: message,
      k,
      avgPrecisionAtK: 0,
      avgRecallAtK: 0,
      meanReciprocalRank: 0,
      meanAveragePrecision: 0,
      avgNdcgAtK: 0,
      avgLatencyMs: 0,
      perQuery: [],
    };
  }
}

function buildErrorAnalysis(reports: EvaluationModeReport[]): EvaluationErrorAnalysis {
  const zeroRelevantQueries: EvaluationErrorAnalysis['zeroRelevantQueries'] = [];
  const rankedTooLow: EvaluationErrorAnalysis['rankedTooLow'] = [];
  const poorRecallQueries: EvaluationErrorAnalysis['poorRecallQueries'] = [];
  const rerankingRegressions: EvaluationErrorAnalysis['rerankingRegressions'] = [];
  const keywordVsSemanticDivergence: EvaluationErrorAnalysis['keywordVsSemanticDivergence'] = [];

  for (const report of reports) {
    if (report.status !== 'completed') continue;
    for (const q of report.perQuery) {
      const relevantSet = new Set(q.relevantDocumentIds);
      const hitIndices = q.retrievedDocumentIds
        .map((id, idx) => (relevantSet.has(id) ? { id, rank: idx + 1 } : null))
        .filter((x): x is { id: string; rank: number } => x !== null);

      if (hitIndices.length === 0) {
        zeroRelevantQueries.push({ method: report.method, queryId: q.queryId, query: q.query });
      } else {
        for (const hit of hitIndices) {
          if (hit.rank > 3) {
            rankedTooLow.push({
              method: report.method,
              queryId: q.queryId,
              query: q.query,
              documentId: hit.id,
              rank: hit.rank,
            });
          }
        }
      }

      if (q.recallAtK < 0.5) {
        poorRecallQueries.push({
          method: report.method,
          queryId: q.queryId,
          query: q.query,
          recallAtK: q.recallAtK,
        });
      }
    }
  }

  const hybridReport = reports.find((r) => r.method === 'hybrid' && r.status === 'completed');
  if (hybridReport) {
    const hybridMap = new Map(hybridReport.perQuery.map((q) => [q.queryId, q]));
    for (const rerankMethod of ['hybrid_cross_encoder', 'hybrid_llm_rerank'] as EvaluationMethodId[]) {
      const rReport = reports.find((r) => r.method === rerankMethod && r.status === 'completed');
      if (!rReport) continue;
      for (const q of rReport.perQuery) {
        const base = hybridMap.get(q.queryId);
        if (base && q.ndcgAtK < base.ndcgAtK - 0.001) {
          rerankingRegressions.push({
            rerankerMethod: rerankMethod,
            queryId: q.queryId,
            query: q.query,
            hybridNdcg: base.ndcgAtK,
            rerankedNdcg: q.ndcgAtK,
            delta: roundMetric(q.ndcgAtK - base.ndcgAtK),
          });
        }
      }
    }
  }

  const bm25Report = reports.find((r) => r.method === 'bm25' && r.status === 'completed');
  const denseReport = reports.find((r) => r.method === 'dense' && r.status === 'completed');
  if (bm25Report && denseReport) {
    const denseMap = new Map(denseReport.perQuery.map((q) => [q.queryId, q]));
    for (const bq of bm25Report.perQuery) {
      const dq = denseMap.get(bq.queryId);
      if (!dq) continue;
      const diff = bq.ndcgAtK - dq.ndcgAtK;
      const winner = Math.abs(diff) < 0.01 ? 'tie' : diff > 0 ? 'bm25' : 'dense';
      keywordVsSemanticDivergence.push({
        queryId: bq.queryId,
        query: bq.query,
        bm25Ndcg: bq.ndcgAtK,
        denseNdcg: dq.ndcgAtK,
        winner,
      });
    }
  }

  return {
    zeroRelevantQueries,
    rankedTooLow,
    poorRecallQueries,
    rerankingRegressions,
    keywordVsSemanticDivergence,
  };
}

export async function runEvaluationSuite(params: {
  methods: EvaluationMethodId[];
  k: number;
  datasetScope?: 'all' | 'human_only' | 'demo_only';
}): Promise<EvaluationRunRecord> {
  const allQueries = lucentStore.getEvaluationQueries();
  const scope = params.datasetScope || 'all';
  const queries = allQueries.filter((q) => {
    if (scope === 'human_only') return !q.isDemo;
    if (scope === 'demo_only') return q.isDemo;
    return true;
  });

  if (queries.length === 0) {
    throw new Error(
      scope === 'human_only'
        ? 'No human-labelled evaluation queries exist yet. Add or import custom queries first.'
        : 'Add at least one benchmark query with relevance judgments before running an experiment.'
    );
  }

  const safeK = Math.max(1, Math.min(20, Math.floor(params.k || 5)));
  const uniqueMethods = Array.from(new Set(params.methods));
  const reports: EvaluationModeReport[] = [];

  for (const method of uniqueMethods) {
    const report = await evaluateSingleMethod(method, queries, safeK);
    reports.push(report);
  }

  const demoQueryCount = queries.filter((q) => q.isDemo).length;
  const humanQueryCount = queries.length - demoQueryCount;
  const datasetLabel =
    humanQueryCount > 0 && demoQueryCount === 0
      ? 'Human-Labelled Evaluation Set'
      : demoQueryCount > 0 && humanQueryCount === 0
        ? 'Demonstration Benchmark Dataset'
        : 'Mixed (Human-Labelled + Demonstration)';

  const datasetVersion = `v${queries.length}-${crypto
    .createHash('sha256')
    .update(JSON.stringify(queries.map((q) => [q.id, q.query, q.relevanceGrades])))
    .digest('hex')
    .slice(0, 8)}`;

  const record: EvaluationRunRecord = {
    id: `exp-${crypto.randomUUID().slice(0, 12)}`,
    timestamp: new Date().toISOString(),
    datasetVersion,
    datasetLabel,
    k: safeK,
    queryCount: queries.length,
    demoQueryCount,
    humanQueryCount,
    configSnapshot: lucentStore.getSettings(),
    reports,
    errorAnalysis: buildErrorAnalysis(reports),
  };

  lucentStore.addEvaluationRun(record);
  return record;
}

/**
 * Parses CSV or JSON evaluation datasets uploaded by researchers.
 * Supported JSON: array of { query, relevantDocumentIds?, relevanceGrades?, category?, notes? }
 * Supported CSV columns: query,document_id,grade,category,notes (multiple rows per query aggregated)
 */
export function parseEvaluationDatasetImport(
  format: 'json' | 'csv',
  content: string
): EvaluationQuery[] {
  const validDocIds = new Set(lucentStore.getDocuments().map((d) => d.id));
  const created: EvaluationQuery[] = [];

  if (format === 'json') {
    const parsed = JSON.parse(content) as unknown;
    const rows = Array.isArray(parsed)
      ? parsed
      : typeof parsed === 'object' && parsed !== null && Array.isArray((parsed as { queries?: unknown }).queries)
        ? (parsed as { queries: unknown[] }).queries
        : null;

    if (!rows) {
      throw new Error('JSON import must be an array of query objects or { "queries": [...] }.');
    }

    for (const rawRow of rows) {
      const row = rawRow as Record<string, unknown>;
      const queryText = typeof row.query === 'string' ? row.query.trim() : '';
      if (!queryText) continue;

      const grades: Record<string, number> = {};
      if (row.relevanceGrades && typeof row.relevanceGrades === 'object') {
        for (const [docId, grade] of Object.entries(row.relevanceGrades as Record<string, unknown>)) {
          if (validDocIds.has(docId) && typeof grade === 'number' && grade >= 0 && grade <= 3) {
            grades[docId] = Math.round(grade);
          }
        }
      }
      if (Array.isArray(row.relevantDocumentIds)) {
        for (const id of row.relevantDocumentIds) {
          if (typeof id === 'string' && validDocIds.has(id) && !(id in grades)) {
            grades[id] = 2;
          }
        }
      }

      const relevantDocumentIds = Object.entries(grades)
        .filter(([, g]) => g > 0)
        .map(([id]) => id);
      if (relevantDocumentIds.length === 0) continue;

      created.push({
        id: `eval-q-${crypto.randomUUID().slice(0, 10)}`,
        query: queryText.slice(0, 400),
        relevantDocumentIds,
        relevanceGrades: grades,
        category: typeof row.category === 'string' && row.category.trim() ? row.category.trim().slice(0, 80) : 'Imported',
        notes: typeof row.notes === 'string' ? row.notes.trim().slice(0, 300) : 'Imported from JSON evaluation dataset.',
        isDemo: false,
      });
    }
  } else {
    const lines = content
      .replace(/\r\n/g, '\n')
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length < 2) {
      throw new Error('CSV import must include a header row and at least one data row.');
    }

    const byQuery = new Map<
      string,
      { grades: Record<string, number>; category: string; notes: string }
    >();

    // Simple CSV parser supporting quoted fields
    const parseLine = (line: string): string[] => {
      const cols: string[] = [];
      let cur = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
          inQuotes = !inQuotes;
        } else if (ch === ',' && !inQuotes) {
          cols.push(cur.trim());
          cur = '';
        } else {
          cur += ch;
        }
      }
      cols.push(cur.trim());
      return cols;
    };

    const header = parseLine(lines[0]).map((h) => h.toLowerCase());
    const qIdx = header.indexOf('query');
    const docIdx = header.findIndex((h) => h === 'document_id' || h === 'documentid' || h === 'doc_id');
    const gradeIdx = header.findIndex((h) => h === 'grade' || h === 'relevance' || h === 'label');
    const catIdx = header.indexOf('category');
    const notesIdx = header.indexOf('notes');

    if (qIdx === -1 || docIdx === -1) {
      throw new Error('CSV header must contain at least "query" and "document_id" columns.');
    }

    for (let i = 1; i < lines.length; i++) {
      const cols = parseLine(lines[i]);
      const queryText = (cols[qIdx] || '').trim();
      const docId = (cols[docIdx] || '').trim();
      if (!queryText || !validDocIds.has(docId)) continue;

      const rawGrade = gradeIdx >= 0 ? Number(cols[gradeIdx]) : 2;
      const grade = Number.isFinite(rawGrade) ? Math.max(0, Math.min(3, Math.round(rawGrade))) : 2;
      const category = catIdx >= 0 && cols[catIdx] ? cols[catIdx] : 'Imported';
      const notes = notesIdx >= 0 && cols[notesIdx] ? cols[notesIdx] : 'Imported from CSV.';

      const entry = byQuery.get(queryText) || { grades: {}, category, notes };
      entry.grades[docId] = grade;
      byQuery.set(queryText, entry);
    }

    for (const [queryText, data] of byQuery.entries()) {
      const relevantDocumentIds = Object.entries(data.grades)
        .filter(([, g]) => g > 0)
        .map(([id]) => id);
      if (relevantDocumentIds.length === 0) continue;

      created.push({
        id: `eval-q-${crypto.randomUUID().slice(0, 10)}`,
        query: queryText.slice(0, 400),
        relevantDocumentIds,
        relevanceGrades: data.grades,
        category: data.category.slice(0, 80),
        notes: data.notes.slice(0, 300),
        isDemo: false,
      });
    }
  }

  if (created.length === 0) {
    throw new Error(
      'No valid evaluation queries could be imported. Ensure document IDs match documents currently in the Document Library.'
    );
  }

  for (const item of created) {
    lucentStore.addEvaluationQuery(item);
  }
  return created;
}

/**
 * Generates a structured Markdown experiment report derived strictly from actual experiment metrics.
 */
export function buildExperimentReportMarkdown(run: EvaluationRunRecord): string {
  const completed = run.reports.filter((r) => r.status === 'completed');
  const bestByNdcg = [...completed].sort((a, b) => b.avgNdcgAtK - a.avgNdcgAtK)[0];
  const fastest = [...completed].sort((a, b) => a.avgLatencyMs - b.avgLatencyMs)[0];

  const rows = run.reports
    .map((r) => {
      if (r.status === 'error') {
        return `| ${r.label} | ${r.modelName} | Error: ${r.errorMessage || 'Failed'} | — | — | — | — | — |`;
      }
      return `| ${r.label} | ${r.modelName} | ${(r.avgPrecisionAtK * 100).toFixed(1)}% | ${(r.avgRecallAtK * 100).toFixed(1)}% | ${r.meanReciprocalRank.toFixed(4)} | ${r.meanAveragePrecision.toFixed(4)} | ${r.avgNdcgAtK.toFixed(4)} | ${r.avgLatencyMs} ms |`;
    })
    .join('\n');

  const lines = [
    `# Lucent Evaluation Experiment Report`,
    ``,
    `- **Experiment ID:** \`${run.id}\``,
    `- **Run Timestamp:** ${run.timestamp}`,
    `- **Dataset:** ${run.datasetLabel} (\`${run.datasetVersion}\`)`,
    `- **Queries Evaluated:** ${run.queryCount} (${run.humanQueryCount} human-labelled, ${run.demoQueryCount} demonstration)`,
    `- **Cutoff K:** ${run.k}`,
    ``,
    `## 1. Research Objective`,
    `Quantitatively benchmark lexical retrieval (BM25), dense semantic retrieval, Reciprocal Rank Fusion (RRF), and optional second-stage neural rerankers on identical query judgments and corpus snapshots to evaluate ranking precision, recall coverage, graded utility (NDCG@${run.k}), and serving latency.`,
    ``,
    `## 2. Experimental Methodology`,
    `- **BM25 Hyperparameters:** k1 = ${run.configSnapshot.bm25K1}, b = ${run.configSnapshot.bm25B}, candidate cutoff = ${run.configSnapshot.bm25Candidates}`,
    `- **Dense Retrieval:** Model = \`${run.configSnapshot.embeddingModel}\`, candidate cutoff = ${run.configSnapshot.semanticCandidates}, metric = L2-normalized FlatIP cosine similarity`,
    `- **Hybrid Fusion:** Method = ${run.configSnapshot.hybridFusionMethod}, RRF k = ${run.configSnapshot.rrfK}, weights (BM25=${run.configSnapshot.bm25Weight}, Semantic=${run.configSnapshot.semanticWeight})`,
    `- **Chunking Configuration:** ${run.configSnapshot.chunkSize} words/chunk, ${run.configSnapshot.chunkOverlap} words overlap`,
    `- **Metrics Calculated:** Precision@${run.k}, Recall@${run.k}, MRR, MAP, and graded NDCG@${run.k} using exponential gain (2^rel - 1) / log2(rank + 1).`,
    ``,
    `## 3. Models Compared & Results`,
    ``,
    `| Method | Model | P@${run.k} | R@${run.k} | MRR | MAP | NDCG@${run.k} | Avg Latency |`,
    `| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |`,
    rows,
    ``,
    `## 4. Interpretation`,
    bestByNdcg
      ? `- **Highest Ranking Quality:** **${bestByNdcg.label}** achieved the top NDCG@${run.k} of **${bestByNdcg.avgNdcgAtK.toFixed(4)}** (MRR ${bestByNdcg.meanReciprocalRank.toFixed(4)}, Recall@${run.k} ${(bestByNdcg.avgRecallAtK * 100).toFixed(1)}%).`
      : `- No methods completed without error in this run.`,
    fastest
      ? `- **Lowest Latency:** **${fastest.label}** averaged **${fastest.avgLatencyMs} ms** per query.`
      : ``,
    `- **Error Analysis Summary:** ${run.errorAnalysis.zeroRelevantQueries.length} zero-relevant query instances, ${run.errorAnalysis.poorRecallQueries.length} low-recall (<50%) instances, ${run.errorAnalysis.rankedTooLow.length} relevant documents ranked below rank #3, and ${run.errorAnalysis.rerankingRegressions.length} reranking regression cases detected.`,
    ``,
    `## 5. Limitations`,
    run.demoQueryCount > 0
      ? `- This run includes ${run.demoQueryCount} demonstration queries over the sample corpus; treat those scores as functional baselines rather than production domain benchmarks.`
      : `- All queries in this run were human-labelled.`,
    `- Corpus size (${lucentStore.getDocuments().length} documents) constrains statistical significance; expand labeled query sets to at least 50+ queries before making production weight commitments.`,
    ``,
    `## 6. Recommendations for Further Experiments`,
    `1. Inspect low-recall and zero-relevant queries in the Error Analysis panel to determine whether misses stem from chunking splits or vocabulary mismatch.`,
    `2. Sweep RRF constant k (e.g., 20, 60, 100) and BM25/Semantic weights to observe sensitivity on domain-specific terminology.`,
    `3. Compare a dedicated MS MARCO Cross-Encoder microservice against Hybrid RRF to quantify accuracy gains relative to added reranking latency.`,
  ];

  return lines.filter((line) => line !== undefined).join('\n');
}
