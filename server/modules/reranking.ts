import { ThinkingLevel, Type } from '@google/genai';
import { getMaskedCrossEncoderUrl, serverConfig } from '../config.ts';
import { getGeminiClient, normalizeGeminiError, withRetry } from '../geminiClient.ts';
import { roundScore } from './ranking.ts';
import type {
  CrossEncoderServiceStatus,
  EngineSettings,
  RerankerMode,
  RerankingMetrics,
  SearchResultItem,
} from '../../src/types/lucent.ts';

export interface RerankExecutionConfig {
  candidateCount: number;
  batchSize: number;
  maxSeqLength: number;
  threshold: number;
}

export interface RerankScoreItem {
  documentId: string;
  chunkId: string;
  score: number;
  explanation: string;
}

export interface RerankerProviderOutput {
  providerType: 'cross_encoder' | 'llm';
  modelName: string;
  scores: RerankScoreItem[];
  processingTimeMs: number;
}

/**
 * Provider interface enabling independent hosting of pretrained cross-encoder models
 * (e.g., Python + FastAPI + PyTorch + Sentence Transformers) or LLM-based rerankers.
 */
export interface RerankerProvider {
  readonly type: 'cross_encoder' | 'llm';
  readonly name: string;
  rerank(
    query: string,
    candidates: SearchResultItem[],
    config: RerankExecutionConfig
  ): Promise<RerankerProviderOutput>;
}

export class RerankerServiceError extends Error {
  statusCode: number;
  code: string;

  constructor(message: string, statusCode = 503, code = 'RERANKER_UNAVAILABLE') {
    super(message);
    this.name = 'RerankerServiceError';
    this.statusCode = statusCode;
    this.code = code;
  }
}

interface PythonCrossEncoderResponse {
  model_name: string;
  model_version?: string;
  device?: string;
  processing_time_ms: number;
  results: Array<{
    document_id: string;
    chunk_id?: string;
    score: number;
    rank: number;
  }>;
}

/**
 * Adapter connecting Lucent to the separate Python FastAPI + PyTorch + Sentence Transformers
 * MS MARCO Cross-Encoder microservice via CROSS_ENCODER_SERVICE_URL.
 * Never silently falls back to keyword heuristics or LLM scoring if unavailable.
 */
export class CrossEncoderHttpProvider implements RerankerProvider {
  public readonly type = 'cross_encoder' as const;
  public readonly name = 'Cross-Encoder Neural Reranker (Python Sentence Transformers)';

  public async rerank(
    query: string,
    candidates: SearchResultItem[],
    config: RerankExecutionConfig
  ): Promise<RerankerProviderOutput> {
    const baseUrl = serverConfig.crossEncoderServiceUrl;
    if (!baseUrl) {
      throw new RerankerServiceError(
        'Cross-encoder reranking service is not configured. Deploy services/cross_encoder and set CROSS_ENCODER_SERVICE_URL on the server.',
        503,
        'CROSS_ENCODER_NOT_CONFIGURED'
      );
    }

    const start = performance.now();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), serverConfig.crossEncoderTimeoutMs);

    try {
      const response = await fetch(`${baseUrl}/rerank`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        signal: controller.signal,
        body: JSON.stringify({
          query,
          batch_size: config.batchSize,
          max_length: config.maxSeqLength,
          documents: candidates.map((c) => ({
            document_id: c.documentId,
            chunk_id: c.chunkId,
            text: c.chunkText || c.excerpt,
          })),
        }),
      });

      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const errJson = (await response.json()) as { detail?: string; error?: string };
          detail = errJson.detail || errJson.error || detail;
        } catch {
          // keep status detail
        }
        throw new RerankerServiceError(
          `Cross-encoder service returned an error: ${detail}`,
          response.status >= 400 && response.status < 600 ? response.status : 502,
          'CROSS_ENCODER_HTTP_ERROR'
        );
      }

      const payload = (await response.json()) as PythonCrossEncoderResponse;
      if (!payload || !Array.isArray(payload.results)) {
        throw new RerankerServiceError(
          'Cross-encoder service returned an invalid response payload.',
          502,
          'CROSS_ENCODER_INVALID_RESPONSE'
        );
      }

      const validIds = new Set(candidates.map((c) => c.documentId));
      const chunkByDoc = new Map(candidates.map((c) => [c.documentId, c.chunkId]));
      const scores: RerankScoreItem[] = [];

      for (const item of payload.results) {
        if (!validIds.has(item.document_id) || typeof item.score !== 'number' || !Number.isFinite(item.score)) {
          continue;
        }
        scores.push({
          documentId: item.document_id,
          chunkId: item.chunk_id || chunkByDoc.get(item.document_id) || '',
          score: item.score,
          explanation: `Cross-encoder (${payload.model_name}) joint query-passage relevance logit/probability score: ${item.score.toFixed(4)}.`,
        });
      }

      if (scores.length === 0 && candidates.length > 0) {
        throw new RerankerServiceError(
          'Cross-encoder service returned zero validated candidate scores.',
          502,
          'CROSS_ENCODER_EMPTY_SCORES'
        );
      }

      return {
        providerType: 'cross_encoder',
        modelName: payload.model_name || 'ms-marco-cross-encoder',
        scores,
        processingTimeMs: Math.max(
          1,
          Math.round(payload.processing_time_ms || performance.now() - start)
        ),
      };
    } catch (err) {
      if (err instanceof RerankerServiceError) {
        throw err;
      }
      if (err instanceof Error && err.name === 'AbortError') {
        throw new RerankerServiceError(
          `Cross-encoder service timed out after ${serverConfig.crossEncoderTimeoutMs}ms.`,
          504,
          'CROSS_ENCODER_TIMEOUT'
        );
      }
      const msg = err instanceof Error ? err.message : 'Network failure';
      throw new RerankerServiceError(
        `Unable to reach the configured cross-encoder service (${getMaskedCrossEncoderUrl()}): ${msg}`,
        503,
        'CROSS_ENCODER_UNREACHABLE'
      );
    } finally {
      clearTimeout(timeout);
    }
  }
}

interface LlmRerankStructuredResponse {
  assessments: Array<{
    documentId: string;
    relevanceScore: number;
    explanation: string;
  }>;
}

/**
 * Optional Gemini LLM-based reranking service.
 * Explicitly identified as an LLM-based reranker (never described as a trained cross-encoder).
 */
export class GeminiLlmRerankerProvider implements RerankerProvider {
  public readonly type = 'llm' as const;
  public readonly name = 'LLM-Based Reranker (Gemini)';

  public async rerank(
    query: string,
    candidates: SearchResultItem[],
    config: RerankExecutionConfig
  ): Promise<RerankerProviderOutput> {
    const start = performance.now();
    const ai = getGeminiClient();

    const maxCharsPerPassage = Math.max(300, Math.min(2400, config.maxSeqLength * 4));
    const candidatePromptBlock = candidates
      .map((c, idx) => {
        const snippet = (c.chunkText || c.excerpt).slice(0, maxCharsPerPassage);
        return `Candidate ${idx + 1}\nDocument ID: ${c.documentId}\nTitle: ${c.title}\nPassage: ${snippet}`;
      })
      .join('\n\n---\n\n');

    const systemInstruction = [
      'You are an information retrieval relevance judge (LLM-based reranker).',
      'Assess how directly each candidate passage answers the user query.',
      'Return a validated JSON object with one assessment per provided Document ID.',
      'Score each candidate from 0.0 (completely irrelevant) to 1.0 (directly and comprehensively answers the query).',
      'Keep each explanation concise (1 sentence, strictly grounded in the passage text).',
      'Do not invent Document IDs that were not supplied.',
    ].join('\n');

    try {
      const response = await withRetry(() =>
        ai.models.generateContent({
          model: serverConfig.generationModel,
          contents: `Query: ${query}\n\nCandidates:\n${candidatePromptBlock}`,
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            systemInstruction,
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                assessments: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      documentId: { type: Type.STRING },
                      relevanceScore: { type: Type.NUMBER },
                      explanation: { type: Type.STRING },
                    },
                    required: ['documentId', 'relevanceScore', 'explanation'],
                  },
                },
              },
              required: ['assessments'],
            },
          },
        })
      );

      const raw = (response.text || '').trim();
      if (!raw) {
        throw new RerankerServiceError('LLM reranker returned an empty response.', 502, 'LLM_RERANK_EMPTY');
      }

      const parsed = JSON.parse(raw) as LlmRerankStructuredResponse;
      if (!parsed || !Array.isArray(parsed.assessments)) {
        throw new RerankerServiceError('LLM reranker output failed schema validation.', 502, 'LLM_RERANK_INVALID');
      }

      const validCandidates = new Map(candidates.map((c) => [c.documentId, c]));
      const seen = new Set<string>();
      const scores: RerankScoreItem[] = [];

      for (const item of parsed.assessments) {
        if (!item || typeof item.documentId !== 'string') continue;
        const cand = validCandidates.get(item.documentId);
        if (!cand || seen.has(item.documentId)) continue;
        if (typeof item.relevanceScore !== 'number' || !Number.isFinite(item.relevanceScore)) continue;

        seen.add(item.documentId);
        const clampedScore = Math.max(0, Math.min(1, item.relevanceScore));
        const explanation =
          typeof item.explanation === 'string' && item.explanation.trim().length > 0
            ? item.explanation.trim().slice(0, 260)
            : 'Assessed by LLM-based relevance scoring.';

        scores.push({
          documentId: item.documentId,
          chunkId: cand.chunkId,
          score: clampedScore,
          explanation,
        });
      }

      if (scores.length === 0) {
        throw new RerankerServiceError(
          'LLM reranker produced no valid document assessments matching the candidate pool.',
          502,
          'LLM_RERANK_UNMATCHED'
        );
      }

      return {
        providerType: 'llm',
        modelName: serverConfig.generationModel,
        scores,
        processingTimeMs: Math.max(1, Math.round(performance.now() - start)),
      };
    } catch (err) {
      if (err instanceof RerankerServiceError) throw err;
      const normalized = normalizeGeminiError(err);
      throw new RerankerServiceError(normalized.message, normalized.statusCode, normalized.code);
    }
  }
}

export const crossEncoderProvider = new CrossEncoderHttpProvider();
export const llmRerankerProvider = new GeminiLlmRerankerProvider();

/**
 * Pure helper applying validated reranking scores to candidate items, computing
 * position movement (originalPosition -> newPosition) and enforcing score thresholds.
 */
export function applyValidatedRerankScores(
  candidates: SearchResultItem[],
  providerOutput: RerankerProviderOutput,
  threshold: number,
  finalLimit: number
): SearchResultItem[] {
  const scoreMap = new Map(providerOutput.scores.map((s) => [s.documentId, s]));

  const rescored = candidates
    .map((item) => {
      const hit = scoreMap.get(item.documentId);
      if (!hit) return null;
      return {
        item,
        rerankScore: roundScore(hit.score),
        rerankExplanation: hit.explanation,
      };
    })
    .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
    .filter((entry) => entry.rerankScore >= threshold);

  // Deterministic sort: descending rerank score, tie-break by original rank
  rescored.sort((a, b) => {
    if (b.rerankScore !== a.rerankScore) {
      return b.rerankScore - a.rerankScore;
    }
    return a.item.rank - b.item.rank;
  });

  return rescored.slice(0, Math.max(1, finalLimit)).map((entry, idx) => {
    const newPosition = idx + 1;
    const originalPosition = entry.item.explanation.originalPosition || entry.item.rank;
    const rankMovement = originalPosition - newPosition;

    return {
      ...entry.item,
      rank: newPosition,
      relevanceScore: entry.rerankScore,
      explanation: {
        ...entry.item.explanation,
        originalPosition,
        newPosition,
        rankMovement,
        rerankScore: entry.rerankScore,
        rerankProvider: providerOutput.providerType,
        rerankModel: providerOutput.modelName,
        rerankExplanation: entry.rerankExplanation,
        finalScore: entry.rerankScore,
      },
    };
  });
}

export async function runRerankingStage(params: {
  query: string;
  mode: RerankerMode;
  candidates: SearchResultItem[];
  settings: EngineSettings;
  finalTopK: number;
}): Promise<{ results: SearchResultItem[]; metrics: RerankingMetrics }> {
  const { query, mode, candidates, settings, finalTopK } = params;

  if (mode === 'none' || candidates.length === 0) {
    return {
      results: candidates.slice(0, finalTopK),
      metrics: {
        provider: 'none',
        modelName: null,
        documentsReranked: 0,
        latencyMs: 0,
        avgTimePerDocMs: 0,
        status: 'skipped',
      },
    };
  }

  const candidateSlice = candidates.slice(0, Math.max(1, settings.rerankCandidateCount || 10));
  const config: RerankExecutionConfig = {
    candidateCount: candidateSlice.length,
    batchSize: settings.rerankBatchSize || 8,
    maxSeqLength: settings.rerankMaxSeqLength || 512,
    threshold: settings.rerankThreshold ?? 0,
  };

  const provider: RerankerProvider =
    mode === 'cross_encoder' ? crossEncoderProvider : llmRerankerProvider;

  const start = performance.now();
  const output = await provider.rerank(query, candidateSlice, config);
  const latencyMs = Math.max(1, Math.round(performance.now() - start));
  const rerankedResults = applyValidatedRerankScores(
    candidateSlice,
    output,
    config.threshold,
    finalTopK
  );

  return {
    results: rerankedResults,
    metrics: {
      provider: mode,
      modelName: output.modelName,
      documentsReranked: candidateSlice.length,
      latencyMs,
      avgTimePerDocMs: Math.round((latencyMs / Math.max(1, candidateSlice.length)) * 10) / 10,
      status: 'executed',
    },
  };
}

/**
 * Checks connectivity and model metadata for the separately hosted Python Cross-Encoder service.
 */
export async function checkCrossEncoderServiceStatus(): Promise<CrossEncoderServiceStatus> {
  const baseUrl = serverConfig.crossEncoderServiceUrl;
  if (!baseUrl) {
    return {
      configured: false,
      serviceUrlMasked: null,
      healthy: false,
      modelName: null,
      modelVersion: null,
      device: null,
      maxSequenceLength: null,
      errorMessage:
        'CROSS_ENCODER_SERVICE_URL is not configured. Deploy services/cross_encoder and set this environment variable to enable neural cross-encoder reranking.',
    };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), Math.min(5000, serverConfig.crossEncoderTimeoutMs));

  try {
    const [healthRes, infoRes] = await Promise.all([
      fetch(`${baseUrl}/health`, { signal: controller.signal }),
      fetch(`${baseUrl}/model-info`, { signal: controller.signal }).catch(() => null),
    ]);

    if (!healthRes.ok) {
      return {
        configured: true,
        serviceUrlMasked: getMaskedCrossEncoderUrl(),
        healthy: false,
        modelName: null,
        modelVersion: null,
        device: null,
        maxSequenceLength: null,
        errorMessage: `Cross-encoder /health returned HTTP ${healthRes.status}.`,
      };
    }

    const infoJson = infoRes && infoRes.ok
      ? ((await infoRes.json()) as {
          model_name?: string;
          model_version?: string;
          device?: string;
          max_length?: number;
        })
      : {};

    return {
      configured: true,
      serviceUrlMasked: getMaskedCrossEncoderUrl(),
      healthy: true,
      modelName: infoJson.model_name || 'cross-encoder/ms-marco-MiniLM-L-6-v2',
      modelVersion: infoJson.model_version || 'unknown',
      device: infoJson.device || 'cpu',
      maxSequenceLength: infoJson.max_length || 512,
      errorMessage: null,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Connection failed';
    return {
      configured: true,
      serviceUrlMasked: getMaskedCrossEncoderUrl(),
      healthy: false,
      modelName: null,
      modelVersion: null,
      device: null,
      maxSequenceLength: null,
      errorMessage: `Could not reach cross-encoder service: ${message}`,
    };
  } finally {
    clearTimeout(timeout);
  }
}
