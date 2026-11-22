import { serverConfig } from '../config.ts';
import { lucentStore } from '../store.ts';
import { embedTexts, ensureChunksEmbedded } from './ingestion.ts';
import { vectorIndex } from './vectorIndex.ts';
import type {
  DocumentChunk,
  EngineSettings,
  ResearchDocument,
  SearchFilters,
} from '../../src/types/lucent.ts';

const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'have',
  'how', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'their',
  'this', 'to', 'was', 'were', 'what', 'when', 'where', 'which', 'who', 'why',
  'with', 'can', 'do', 'does', 'into', 'than', 'then', 'about', 'between',
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, ' ')
    .split(/\s+/)
    .map((t) => t.replace(/^-+|-+$/g, ''))
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

export function computeCosineSimilarity(vecA: number[], vecB: number[]): number {
  const len = Math.min(vecA.length, vecB.length);
  if (len === 0) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function filterDocuments(
  documents: ResearchDocument[],
  filters?: SearchFilters
): ResearchDocument[] {
  if (!filters) return documents;

  return documents.filter((doc) => {
    if (filters.includeDemo === false && doc.isDemo) {
      return false;
    }
    if (filters.documentType && filters.documentType !== 'all' && doc.documentType !== filters.documentType) {
      return false;
    }
    if (filters.source && filters.source !== 'all') {
      const normalizedSource = doc.source || '__unspecified__';
      if (normalizedSource !== filters.source) {
        return false;
      }
    }
    if (filters.dateFrom && (!doc.publishedDate || doc.publishedDate < filters.dateFrom)) {
      return false;
    }
    if (filters.dateTo && (!doc.publishedDate || doc.publishedDate > filters.dateTo)) {
      return false;
    }
    return true;
  });
}

export interface LexicalCandidate {
  chunk: DocumentChunk;
  document: ResearchDocument;
  bm25Score: number;
  matchedTerms: string[];
}

export interface SemanticCandidate {
  chunk: DocumentChunk;
  document: ResearchDocument;
  cosineSimilarity: number;
}

/**
 * Deterministic Okapi BM25 lexical retrieval over document chunks.
 * Uses configurable k1 (TF saturation), b (length normalization), and candidate cutoff.
 */
export function runBm25Retrieval(
  query: string,
  chunks: DocumentChunk[],
  docMap: Map<string, ResearchDocument>,
  settings: Pick<EngineSettings, 'bm25K1' | 'bm25B' | 'bm25Candidates'>,
  candidateLimitOverride?: number
): { candidates: LexicalCandidate[]; queryTokens: string[]; latencyMs: number } {
  const start = performance.now();
  const queryTokens = Array.from(new Set(tokenize(query)));

  if (queryTokens.length === 0 || chunks.length === 0) {
    return { candidates: [], queryTokens, latencyMs: Math.round(performance.now() - start) };
  }

  const k1 = settings.bm25K1;
  const b = settings.bm25B;
  const maxCandidates = Math.max(1, candidateLimitOverride ?? settings.bm25Candidates ?? 25);

  const tokenizedDocs = chunks.map((chunk) => {
    const doc = docMap.get(chunk.documentId);
    const combinedText = doc ? `${doc.title} ${chunk.text}` : chunk.text;
    const tokens = tokenize(combinedText);
    const freq = new Map<string, number>();
    for (const tok of tokens) {
      freq.set(tok, (freq.get(tok) || 0) + 1);
    }
    return { chunk, doc, length: Math.max(1, tokens.length), freq };
  });

  const N = tokenizedDocs.length;
  const avgDocLen =
    tokenizedDocs.reduce((sum, item) => sum + item.length, 0) / Math.max(1, N);

  const df = new Map<string, number>();
  for (const term of queryTokens) {
    let count = 0;
    for (const item of tokenizedDocs) {
      if (item.freq.has(term)) count++;
    }
    df.set(term, count);
  }

  const candidates: LexicalCandidate[] = [];

  for (const item of tokenizedDocs) {
    if (!item.doc) continue;
    let score = 0;
    const matchedTerms: string[] = [];

    for (const term of queryTokens) {
      const tf = item.freq.get(term) || 0;
      if (tf === 0) continue;

      matchedTerms.push(term);
      const termDf = df.get(term) || 0;
      const idf = Math.log(1 + (N - termDf + 0.5) / (termDf + 0.5));
      const numerator = tf * (k1 + 1);
      const denominator = tf + k1 * (1 - b + b * (item.length / avgDocLen));
      score += idf * (numerator / denominator);
    }

    if (score > 0) {
      candidates.push({
        chunk: item.chunk,
        document: item.doc,
        bm25Score: score,
        matchedTerms,
      });
    }
  }

  // Deterministic sort: descending BM25 score, tie-break by chunk ID
  candidates.sort((a, bItem) => {
    if (bItem.bm25Score !== a.bm25Score) {
      return bItem.bm25Score - a.bm25Score;
    }
    return a.chunk.id.localeCompare(bItem.chunk.id);
  });

  return {
    candidates: candidates.slice(0, maxCandidates),
    queryTokens,
    latencyMs: Math.max(1, Math.round(performance.now() - start)),
  };
}

/**
 * Dense vector semantic retrieval backed by the local FAISS-compatible Float32 Vector Index
 * and server-side Google embedding model. Separates embedding latency from vector search latency.
 */
export async function runSemanticRetrieval(
  query: string,
  chunks: DocumentChunk[],
  docMap: Map<string, ResearchDocument>,
  candidateLimitOverride?: number
): Promise<{
  candidates: SemanticCandidate[];
  embeddedNewChunks: number;
  missingEmbeddingsCount: number;
  model: string;
  embeddingLatencyMs: number;
  vectorSearchLatencyMs: number;
  latencyMs: number;
}> {
  const totalStart = performance.now();
  const settings = lucentStore.getSettings();
  const activeModel = (settings.embeddingModel || serverConfig.embeddingModel).trim();
  const maxCandidates = Math.max(1, candidateLimitOverride ?? settings.semanticCandidates ?? 25);

  if (!query.trim() || chunks.length === 0) {
    return {
      candidates: [],
      embeddedNewChunks: 0,
      missingEmbeddingsCount: 0,
      model: activeModel,
      embeddingLatencyMs: 0,
      vectorSearchLatencyMs: 0,
      latencyMs: 0,
    };
  }

  const embedStart = performance.now();
  const { embeddedNow } = await ensureChunksEmbedded(chunks, { model: activeModel });
  const { vectors } = await embedTexts([query.trim()], activeModel, 1);
  const queryVector = vectors[0];
  const embeddingLatencyMs = Math.max(1, Math.round(performance.now() - embedStart));

  const searchStart = performance.now();
  const chunkMap = new Map(chunks.map((c) => [c.id, c]));
  const allowedChunkIds = new Set(chunks.map((c) => c.id));

  const hits = vectorIndex.search(queryVector, {
    model: activeModel,
    allowedChunkIds,
    topK: maxCandidates,
  });

  const candidates: SemanticCandidate[] = [];
  for (const hit of hits) {
    const chunk = chunkMap.get(hit.chunkId);
    const doc = docMap.get(hit.documentId);
    if (!chunk || !doc) continue;
    candidates.push({
      chunk,
      document: doc,
      cosineSimilarity: hit.score,
    });
  }

  // Deterministic tie-breaking
  candidates.sort((a, b) => {
    if (b.cosineSimilarity !== a.cosineSimilarity) {
      return b.cosineSimilarity - a.cosineSimilarity;
    }
    return a.chunk.id.localeCompare(b.chunk.id);
  });

  const vectorSearchLatencyMs = Math.max(1, Math.round(performance.now() - searchStart));
  const indexedInScope = chunks.filter((c) =>
    vectorIndex.hasMatchingVector(c.id, c.contentHash, activeModel)
  ).length;

  return {
    candidates,
    embeddedNewChunks: embeddedNow,
    missingEmbeddingsCount: Math.max(0, chunks.length - indexedInScope),
    model: activeModel,
    embeddingLatencyMs,
    vectorSearchLatencyMs,
    latencyMs: Math.max(1, Math.round(performance.now() - totalStart)),
  };
}

export function extractExcerpt(chunkText: string, matchedTerms: string[], maxLength = 340): string {
  const clean = chunkText.replace(/\s+/g, ' ').trim();
  if (clean.length <= maxLength) return clean;

  if (matchedTerms.length > 0) {
    const lower = clean.toLowerCase();
    let bestPos = -1;
    for (const term of matchedTerms) {
      const idx = lower.indexOf(term.toLowerCase());
      if (idx !== -1 && (bestPos === -1 || idx < bestPos)) {
        bestPos = idx;
      }
    }
    if (bestPos > 60) {
      const start = Math.max(0, bestPos - 70);
      const slice = clean.slice(start, start + maxLength);
      const prefix = start > 0 ? '...' : '';
      const suffix = start + maxLength < clean.length ? '...' : '';
      return `${prefix}${slice.trim()}${suffix}`;
    }
  }

  return `${clean.slice(0, maxLength).trim()}...`;
}
