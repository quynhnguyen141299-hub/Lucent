import { extractExcerpt } from './retrieval.ts';
import type { LexicalCandidate, SemanticCandidate } from './retrieval.ts';
import type {
  DocumentChunk,
  EngineSettings,
  ResearchDocument,
  RetrievalExplanation,
  SearchMode,
  SearchResultItem,
} from '../../src/types/lucent.ts';

interface MergedChunkCandidate {
  chunk: DocumentChunk;
  document: ResearchDocument;
  bm25Score: number | null;
  bm25Rank: number | null;
  matchedTerms: string[];
  semanticScore: number | null;
  semanticRank: number | null;
  rrfScore: number | null;
  finalScore: number;
  fusionMethod: string | null;
}

export function roundScore(value: number): number {
  return Math.round(value * 100000) / 100000;
}

/**
 * Pure Reciprocal Rank Fusion calculator exported for unit testing and inspection.
 * Formula: RRF(d) = sum_i [ weight_i / (k + rank_i(d)) ]
 */
export function computeRrfScore(params: {
  bm25Rank: number | null;
  semanticRank: number | null;
  k: number;
  bm25Weight?: number;
  semanticWeight?: number;
}): number {
  const safeK = Math.max(1, params.k);
  const wBm25 = params.bm25Weight ?? 1.0;
  const wSem = params.semanticWeight ?? 1.0;
  const lexTerm = params.bm25Rank !== null ? wBm25 / (safeK + params.bm25Rank) : 0;
  const semTerm = params.semanticRank !== null ? wSem / (safeK + params.semanticRank) : 0;
  return lexTerm + semTerm;
}

function buildExplanationSummary(
  mode: SearchMode,
  candidate: MergedChunkCandidate,
  matchingChunksCount: number
): string {
  const parts: string[] = [];

  if (mode === 'keyword') {
    parts.push(
      `Ranked via Okapi BM25 lexical matching (score ${candidate.bm25Score?.toFixed(4) ?? '0.0000'})`
    );
    if (candidate.matchedTerms.length > 0) {
      parts.push(`matching terms: ${candidate.matchedTerms.slice(0, 6).join(', ')}`);
    }
  } else if (mode === 'semantic') {
    const pct = candidate.semanticScore !== null ? (candidate.semanticScore * 100).toFixed(1) : '0.0';
    parts.push(
      `Ranked via dense embedding cosine similarity (${pct}% vector alignment) on chunk #${candidate.chunk.chunkIndex + 1}`
    );
  } else {
    const methodLabel =
      candidate.fusionMethod === 'rrf'
        ? `Reciprocal Rank Fusion (RRF ${candidate.rrfScore?.toFixed(5) ?? '0.00000'})`
        : 'weighted score fusion';
    parts.push(`Combined lexical and semantic signals using ${methodLabel}`);
    if (candidate.bm25Rank !== null && candidate.semanticRank !== null) {
      parts.push(`at BM25 rank #${candidate.bm25Rank} and semantic rank #${candidate.semanticRank}`);
    } else if (candidate.semanticRank !== null) {
      parts.push(`surfaced via semantic similarity (rank #${candidate.semanticRank}) without exact keyword overlap`);
    } else if (candidate.bm25Rank !== null) {
      parts.push(`surfaced via BM25 lexical match (rank #${candidate.bm25Rank})`);
    }
  }

  if (candidate.chunk.pageNumber !== null) {
    parts.push(`from page ${candidate.chunk.pageNumber}`);
  }
  if (matchingChunksCount > 1) {
    parts.push(`with ${matchingChunksCount} matching passages in this document`);
  }

  return `${parts.join('; ')}.`;
}

export interface RankCandidatesParams {
  mode: SearchMode;
  lexicalCandidates: LexicalCandidate[];
  semanticCandidates: SemanticCandidate[];
  settings: Pick<
    EngineSettings,
    'hybridFusionMethod' | 'hybridAlpha' | 'rrfK' | 'bm25Weight' | 'semanticWeight'
  >;
  topK: number;
  rrfKOverride?: number;
  bm25WeightOverride?: number;
  semanticWeightOverride?: number;
}

/**
 * Fuses BM25 and dense semantic candidates using genuine RRF or weighted linear fusion,
 * eliminates duplicate documents (keeping the strongest passage per document),
 * and produces deterministic ranked results with full score provenance.
 */
export function rankCandidates(params: RankCandidatesParams): {
  results: SearchResultItem[];
  totalDocumentMatches: number;
  latencyMs: number;
} {
  const start = performance.now();
  const { mode, lexicalCandidates, semanticCandidates, settings, topK } = params;

  const rrfK = Math.max(1, params.rrfKOverride ?? settings.rrfK ?? 60);
  const bm25Weight = Math.max(0, params.bm25WeightOverride ?? settings.bm25Weight ?? 1.0);
  const semanticWeight = Math.max(0, params.semanticWeightOverride ?? settings.semanticWeight ?? 1.0);

  const chunkMap = new Map<string, MergedChunkCandidate>();

  const maxBm25 =
    lexicalCandidates.length > 0
      ? Math.max(...lexicalCandidates.map((c) => c.bm25Score), 0.0001)
      : 1;
  const maxSem =
    semanticCandidates.length > 0
      ? Math.max(...semanticCandidates.map((c) => c.cosineSimilarity), 0.0001)
      : 1;
  const minSem =
    semanticCandidates.length > 0
      ? Math.min(...semanticCandidates.map((c) => c.cosineSimilarity), 0)
      : 0;
  const semRange = Math.max(0.0001, maxSem - minSem);

  lexicalCandidates.forEach((item, idx) => {
    chunkMap.set(item.chunk.id, {
      chunk: item.chunk,
      document: item.document,
      bm25Score: item.bm25Score,
      bm25Rank: idx + 1,
      matchedTerms: item.matchedTerms,
      semanticScore: null,
      semanticRank: null,
      rrfScore: null,
      finalScore: 0,
      fusionMethod: null,
    });
  });

  semanticCandidates.forEach((item, idx) => {
    const existing = chunkMap.get(item.chunk.id);
    if (existing) {
      existing.semanticScore = item.cosineSimilarity;
      existing.semanticRank = idx + 1;
    } else {
      chunkMap.set(item.chunk.id, {
        chunk: item.chunk,
        document: item.document,
        bm25Score: null,
        bm25Rank: null,
        matchedTerms: [],
        semanticScore: item.cosineSimilarity,
        semanticRank: idx + 1,
        rrfScore: null,
        finalScore: 0,
        fusionMethod: null,
      });
    }
  });

  const mergedChunks = Array.from(chunkMap.values());

  for (const cand of mergedChunks) {
    if (mode === 'keyword') {
      cand.finalScore = cand.bm25Score !== null ? cand.bm25Score : 0;
      cand.fusionMethod = 'bm25';
    } else if (mode === 'semantic') {
      cand.finalScore = cand.semanticScore !== null ? Math.max(0, cand.semanticScore) : 0;
      cand.fusionMethod = 'cosine_similarity';
    } else {
      const hasSemantic = semanticCandidates.length > 0;
      const rawRrf = computeRrfScore({
        bm25Rank: cand.bm25Rank,
        semanticRank: cand.semanticRank,
        k: rrfK,
        bm25Weight,
        semanticWeight,
      });
      cand.rrfScore = rawRrf;

      if (!hasSemantic) {
        cand.finalScore = cand.bm25Score !== null ? cand.bm25Score : 0;
        cand.fusionMethod = 'bm25_fallback';
      } else if (settings.hybridFusionMethod === 'rrf') {
        cand.finalScore = rawRrf;
        cand.fusionMethod = 'rrf';
      } else {
        const totalWeight = Math.max(0.0001, bm25Weight + semanticWeight);
        const wLex = bm25Weight / totalWeight;
        const wSem = semanticWeight / totalWeight;
        const normLex = cand.bm25Score !== null ? cand.bm25Score / maxBm25 : 0;
        const normSem = cand.semanticScore !== null ? (cand.semanticScore - minSem) / semRange : 0;
        cand.finalScore = wSem * normSem + wLex * normLex;
        cand.fusionMethod = 'weighted_linear';
      }
    }
  }

  // Deduplicate by document, retaining the highest-scoring passage chunk per document
  const byDoc = new Map<string, { best: MergedChunkCandidate; matchingChunks: number }>();

  for (const cand of mergedChunks) {
    if (cand.finalScore <= 0) continue;
    const existing = byDoc.get(cand.document.id);
    if (!existing) {
      byDoc.set(cand.document.id, { best: cand, matchingChunks: 1 });
    } else {
      existing.matchingChunks++;
      if (
        cand.finalScore > existing.best.finalScore ||
        (cand.finalScore === existing.best.finalScore &&
          cand.chunk.chunkIndex < existing.best.chunk.chunkIndex)
      ) {
        existing.best = cand;
      }
    }
  }

  // Deterministic ranking: descending score, tie-break by document ID
  const rankedDocs = Array.from(byDoc.values()).sort((a, b) => {
    if (b.best.finalScore !== a.best.finalScore) {
      return b.best.finalScore - a.best.finalScore;
    }
    return a.best.document.id.localeCompare(b.best.document.id);
  });

  const limited = rankedDocs.slice(0, Math.max(1, topK));

  const results: SearchResultItem[] = limited.map((entry, idx) => {
    const c = entry.best;
    const position = idx + 1;
    const explanation: RetrievalExplanation = {
      summary: buildExplanationSummary(mode, c, entry.matchingChunks),
      matchedTerms: c.matchedTerms,
      bm25Score: c.bm25Score !== null ? roundScore(c.bm25Score) : null,
      bm25Rank: c.bm25Rank,
      semanticScore: c.semanticScore !== null ? roundScore(c.semanticScore) : null,
      semanticRank: c.semanticRank,
      rrfScore: c.rrfScore !== null ? roundScore(c.rrfScore) : null,
      fusionMethod: c.fusionMethod,
      hybridRank: mode === 'hybrid' ? position : null,
      originalPosition: position,
      newPosition: position,
      rankMovement: 0,
      rerankScore: null,
      rerankProvider: null,
      rerankModel: null,
      rerankExplanation: null,
      finalScore: roundScore(c.finalScore),
      chunkId: c.chunk.id,
      chunkIndex: c.chunk.chunkIndex,
      pageNumber: c.chunk.pageNumber,
      totalChunksInDocument: c.document.chunkCount,
      matchingChunksCount: entry.matchingChunks,
    };

    return {
      rank: position,
      documentId: c.document.id,
      chunkId: c.chunk.id,
      title: c.document.title,
      source: c.document.source,
      documentType: c.document.documentType,
      publishedDate: c.document.publishedDate,
      pageNumber: c.chunk.pageNumber,
      authors: c.document.authors,
      sourceUrl: c.document.sourceUrl,
      originalFilename: c.document.originalFilename,
      hasUploadedFile: Boolean(c.document.storedFilePath),
      isDemo: c.document.isDemo,
      excerpt: extractExcerpt(c.chunk.text, c.matchedTerms),
      chunkText: c.chunk.text,
      relevanceScore: roundScore(c.finalScore),
      explanation,
    };
  });

  return {
    results,
    totalDocumentMatches: rankedDocs.length,
    latencyMs: Math.max(1, Math.round(performance.now() - start)),
  };
}
