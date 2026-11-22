import 'dotenv/config';
import crypto from 'crypto';
import express from 'express';
import fs from 'fs';
import multer from 'multer';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { getPublicRuntimeConfig, serverConfig, SUPPORTED_EMBEDDING_MODELS } from './server/config.ts';
import { normalizeGeminiError } from './server/geminiClient.ts';
import { apiRateLimiter, apiRequestLogger, sanitizeShortString } from './server/middleware/reliability.ts';
import {
  buildExperimentReportMarkdown,
  parseEvaluationDatasetImport,
  runEvaluationSuite,
} from './server/modules/evaluation.ts';
import { classifyQuery, generateGroundedAnswer, normalizeAnswerStyle } from './server/modules/generation.ts';
import {
  embedSingleDocument,
  ensureChunksEmbedded,
  extractDocumentFromBuffer,
  ingestDocumentRecord,
  rechunkCorpus,
  reprocessDocument,
  restoreDemoCorpus,
} from './server/modules/ingestion.ts';
import { rankCandidates } from './server/modules/ranking.ts';
import {
  checkCrossEncoderServiceStatus,
  RerankerServiceError,
  runRerankingStage,
} from './server/modules/reranking.ts';
import { filterDocuments, runBm25Retrieval, runSemanticRetrieval } from './server/modules/retrieval.ts';
import { vectorIndex } from './server/modules/vectorIndex.ts';
import { lucentStore } from './server/store.ts';
import type {
  AnalyticsSnapshot,
  EngineSettings,
  EvaluationMethodId,
  ModeComparisonEntry,
  PipelineStageTrace,
  QueryClassificationResult,
  RerankerMode,
  RerankingMetrics,
  SearchComparisonResponse,
  SearchMode,
  SearchPipelineTrace,
  SearchRequestPayload,
  SearchResponsePayload,
  StageLatencyBreakdown,
} from './src/types/lucent.ts';

const app = express();
app.use(express.json({ limit: '6mb' }));
app.use(apiRequestLogger);
app.use('/api', apiRateLimiter);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: serverConfig.maxUploadBytes,
    files: 1,
  },
});

const VALID_SEARCH_MODES: Set<SearchMode> = new Set(['keyword', 'semantic', 'hybrid']);
const VALID_RERANKER_MODES: Set<RerankerMode> = new Set(['none', 'cross_encoder', 'llm']);
const VALID_EVAL_METHODS: Set<EvaluationMethodId> = new Set([
  'bm25',
  'dense',
  'hybrid',
  'hybrid_cross_encoder',
  'hybrid_llm_rerank',
]);

function emptyLatencies(totalLatencyMs = 0): StageLatencyBreakdown {
  return {
    queryProcessingMs: 0,
    queryClassificationMs: 0,
    embeddingGenerationMs: 0,
    bm25RetrievalMs: 0,
    vectorRetrievalMs: 0,
    hybridFusionMs: 0,
    rerankingMs: 0,
    answerGenerationMs: 0,
    totalLatencyMs,
  };
}

// Health, Config & Vector Index Status
app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    timestamp: new Date().toISOString(),
    documentsIndexed: lucentStore.getDocuments().length,
    chunksIndexed: lucentStore.getChunks().length,
    vectorIndex: vectorIndex.getStats(),
  });
});

app.get('/api/config', (_req, res) => {
  res.json({
    runtime: getPublicRuntimeConfig(),
    settings: lucentStore.getSettings(),
  });
});

app.get('/api/vector-index/stats', (_req, res) => {
  res.json({ stats: vectorIndex.getStats() });
});

app.get('/api/reranker/status', async (_req, res) => {
  const crossEncoder = await checkCrossEncoderServiceStatus();
  res.json({
    crossEncoder,
    llmReranker: {
      configured: getPublicRuntimeConfig().hasGeminiApiKey,
      modelName: serverConfig.generationModel,
      label: 'LLM-Based Reranker (Gemini)',
    },
  });
});

// Document Library & Ingestion Routes
app.get('/api/documents', (_req, res) => {
  const settings = lucentStore.getSettings();
  const activeModel = settings.embeddingModel || serverConfig.embeddingModel;
  const docs = lucentStore.getDocuments().map((doc) => {
    const docChunks = lucentStore.getChunksByDocumentId(doc.id);
    const embeddedCount = docChunks.filter((c) =>
      vectorIndex.hasMatchingVector(c.id, c.contentHash, activeModel)
    ).length;
    return {
      ...doc,
      content: undefined,
      chunkCount: docChunks.length,
      embeddedChunkCount: embeddedCount,
    };
  });
  res.json({ documents: docs, vectorIndex: vectorIndex.getStats() });
});

app.get('/api/documents/:id', (req, res) => {
  const doc = lucentStore.getDocumentById(req.params.id);
  if (!doc) {
    res.status(404).json({ error: 'Document not found.' });
    return;
  }
  const activeModel = lucentStore.getSettings().embeddingModel || serverConfig.embeddingModel;
  const chunks = lucentStore.getChunksByDocumentId(doc.id).map((c) => {
    const meta = vectorIndex.getVectorMeta(c.id);
    const hasEmbedding = vectorIndex.hasMatchingVector(c.id, c.contentHash, activeModel);
    return {
      ...c,
      hasEmbedding,
      embeddingModel: hasEmbedding ? meta?.model || activeModel : null,
    };
  });
  res.json({ document: doc, chunks });
});

app.get('/api/documents/:id/download', (req, res) => {
  const doc = lucentStore.getDocumentById(req.params.id);
  if (!doc) {
    res.status(404).json({ error: 'Document not found.' });
    return;
  }

  res.setHeader('X-Content-Type-Options', 'nosniff');

  if (doc.storedFilePath) {
    const safePath = lucentStore.resolveSafeUploadPath(doc.storedFilePath);
    if (safePath && fs.existsSync(safePath)) {
      const downloadName = (doc.originalFilename || `${doc.id}.txt`).replace(/["\r\n]/g, '_');
      res.setHeader('Content-Disposition', `attachment; filename="${downloadName}"`);
      res.setHeader('Content-Type', doc.mimeType || 'application/octet-stream');
      res.sendFile(safePath);
      return;
    }
  }

  const safeSlug = doc.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 60) || doc.id;
  res.setHeader('Content-Disposition', `attachment; filename="${safeSlug}.txt"`);
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.send(
    `${doc.title}\nSource: ${doc.source || 'Not specified'}\nPublished: ${doc.publishedDate || 'Undated'}\n\n${doc.content}`
  );
});

app.post('/api/documents', async (req, res) => {
  try {
    const body = req.body as Record<string, unknown>;
    const title = sanitizeShortString(body.title, 240);
    const source = sanitizeShortString(body.source, 160) || null;
    const documentType = sanitizeShortString(body.documentType, 60) || 'Uploaded Document';
    const publishedDate = sanitizeShortString(body.publishedDate, 20) || null;
    const sourceUrl = typeof body.sourceUrl === 'string' ? body.sourceUrl.trim() : null;
    const content = typeof body.content === 'string' ? body.content : '';
    const authors = Array.isArray(body.authors)
      ? body.authors.filter((a): a is string => typeof a === 'string')
      : typeof body.authors === 'string'
        ? body.authors.split(',').map((a) => a.trim()).filter(Boolean)
        : [];

    const { document, chunks } = ingestDocumentRecord({
      title,
      source,
      documentType,
      publishedDate,
      authors,
      sourceUrl,
      content,
      isDemo: false,
    });

    const settings = lucentStore.getSettings();
    let embeddingWarning: string | null = null;

    if (settings.autoEmbedOnUpload && chunks.length > 0) {
      try {
        await ensureChunksEmbedded(chunks, { model: settings.embeddingModel });
      } catch (err) {
        embeddingWarning = normalizeGeminiError(err).message;
      }
    }

    const updated = lucentStore.getDocumentById(document.id) || document;
    res.status(201).json({
      document: updated,
      embeddingWarning,
      vectorIndex: vectorIndex.getStats(),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to ingest document.';
    res.status(400).json({ error: message });
  }
});

app.post('/api/documents/upload', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      res.status(400).json({ error: 'No file was provided in the upload request.' });
      return;
    }

    const extracted = await extractDocumentFromBuffer(file.originalname, file.buffer);
    const body = req.body as Record<string, unknown>;
    const fallbackTitle = path
      .basename(file.originalname, path.extname(file.originalname))
      .replace(/[_-]+/g, ' ')
      .trim();

    const userTitle = sanitizeShortString(body.title, 240);
    const title = userTitle || extracted.detectedTitle || fallbackTitle;
    // Never invent a source or publication date if none was provided
    const source = sanitizeShortString(body.source, 160) || null;
    const documentType = sanitizeShortString(body.documentType, 60) || 'Uploaded Document';
    const publishedDate = sanitizeShortString(body.publishedDate, 20) || null;
    const sourceUrl = typeof body.sourceUrl === 'string' ? body.sourceUrl.trim() : null;
    const userAuthors =
      typeof body.authors === 'string'
        ? body.authors.split(',').map((a) => a.trim()).filter(Boolean)
        : [];
    const authors = userAuthors.length > 0 ? userAuthors : extracted.detectedAuthors;

    const { document, chunks } = ingestDocumentRecord({
      title,
      source,
      documentType,
      publishedDate,
      authors,
      sourceUrl,
      content: extracted.fullText,
      pages: extracted.pages,
      pageCount: extracted.pageCount,
      isDemo: false,
      uploadedFile: {
        originalName: file.originalname,
        mimeType: file.mimetype,
        buffer: file.buffer,
      },
    });

    const settings = lucentStore.getSettings();
    let embeddingWarning: string | null = null;

    if (settings.autoEmbedOnUpload && chunks.length > 0) {
      try {
        await ensureChunksEmbedded(chunks, { model: settings.embeddingModel });
      } catch (err) {
        embeddingWarning = normalizeGeminiError(err).message;
      }
    }

    const updated = lucentStore.getDocumentById(document.id) || document;
    res.status(201).json({
      document: updated,
      embeddingWarning,
      vectorIndex: vectorIndex.getStats(),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to process uploaded document.';
    res.status(400).json({ error: message });
  }
});

app.post('/api/documents/:id/reprocess', async (req, res) => {
  try {
    const forceReembed = Boolean((req.body as { forceReembed?: boolean })?.forceReembed);
    const result = await reprocessDocument(req.params.id, { forceReembed });
    res.json({
      ...result,
      vectorIndex: vectorIndex.getStats(),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to reprocess document.';
    res.status(400).json({ error: message });
  }
});

app.post('/api/documents/:id/embed', async (req, res) => {
  try {
    const force = Boolean((req.body as { force?: boolean })?.force);
    const result = await embedSingleDocument(req.params.id, { force });
    res.json({
      ...result,
      vectorIndex: vectorIndex.getStats(),
    });
  } catch (err) {
    const normalized = normalizeGeminiError(err);
    res.status(normalized.statusCode).json({ error: normalized.message, code: normalized.code });
  }
});

app.delete('/api/documents/demo', (_req, res) => {
  const result = lucentStore.deleteDemoDocuments();
  res.json({ ...result, vectorIndex: vectorIndex.getStats() });
});

app.delete('/api/documents/:id', (req, res) => {
  const deleted = lucentStore.deleteDocument(req.params.id);
  if (!deleted) {
    res.status(404).json({ error: 'Document not found.' });
    return;
  }
  res.json({ deleted: true, id: req.params.id, vectorIndex: vectorIndex.getStats() });
});

app.post('/api/documents/demo/restore', (_req, res) => {
  const result = restoreDemoCorpus();
  res.json(result);
});

app.post('/api/documents/reindex-embeddings', async (req, res) => {
  try {
    const force = Boolean((req.body as { force?: boolean })?.force);
    const result = await ensureChunksEmbedded(undefined, { force });
    res.json({
      ok: true,
      ...result,
      vectorIndex: vectorIndex.getStats(),
    });
  } catch (err) {
    const normalized = normalizeGeminiError(err);
    res.status(normalized.statusCode).json({ error: normalized.message, code: normalized.code });
  }
});

// Core Search, Reranking & RAG Pipeline
app.post('/api/search', async (req, res) => {
  const body = req.body as Partial<SearchRequestPayload>;
  const query = sanitizeShortString(body.query, 600);

  if (!query) {
    res.status(400).json({ error: 'Please enter a search query.' });
    return;
  }

  const settings = lucentStore.getSettings();
  const mode: SearchMode =
    body.mode && VALID_SEARCH_MODES.has(body.mode) ? body.mode : settings.defaultMode;
  const rerankerMode: RerankerMode =
    body.rerankerMode && VALID_RERANKER_MODES.has(body.rerankerMode)
      ? body.rerankerMode
      : settings.defaultRerankerMode;
  const answerStyle = normalizeAnswerStyle(body.answerStyle || settings.defaultAnswerStyle);
  const topK = Math.max(1, Math.min(20, Math.floor(Number(body.topK) || settings.topK)));
  const bm25CandidateCap = Math.max(1, Math.min(100, Math.floor(Number(body.bm25Candidates) || settings.bm25Candidates)));
  const semanticCandidateCap = Math.max(1, Math.min(100, Math.floor(Number(body.semanticCandidates) || settings.semanticCandidates)));
  const hybridCandidateCap = Math.max(topK, Math.min(50, Math.floor(Number(body.hybridCandidates) || settings.hybridCandidates)));
  const bm25Weight = typeof body.bm25Weight === 'number' && Number.isFinite(body.bm25Weight) ? body.bm25Weight : settings.bm25Weight;
  const semanticWeight = typeof body.semanticWeight === 'number' && Number.isFinite(body.semanticWeight) ? body.semanticWeight : settings.semanticWeight;
  const rrfK = typeof body.rrfK === 'number' && Number.isFinite(body.rrfK) ? body.rrfK : settings.rrfK;
  const generateAnswer =
    typeof body.generateAnswer === 'boolean' ? body.generateAnswer : settings.enableRagByDefault;
  const filters = body.filters || {};

  const queryId = `q-${crypto.randomUUID()}`;
  const totalStart = performance.now();
  const stages: PipelineStageTrace[] = [];
  const warnings: string[] = [];
  const latencies = emptyLatencies(0);

  // Stage 1: Query Normalization & Metadata Filtering
  const filterStart = performance.now();
  const allDocs = lucentStore.getDocuments();
  const eligibleDocs = filterDocuments(allDocs, filters);
  const eligibleDocMap = new Map(eligibleDocs.map((d) => [d.id, d]));
  const eligibleChunks = lucentStore.getChunks().filter((c) => eligibleDocMap.has(c.documentId));
  latencies.queryProcessingMs = Math.max(1, Math.round(performance.now() - filterStart));

  stages.push({
    stage: 'query_processing',
    label: 'Query Normalization & Corpus Filtering',
    status: 'executed',
    latencyMs: latencies.queryProcessingMs,
    candidateCount: eligibleChunks.length,
    details: `Filtered ${eligibleDocs.length} of ${allDocs.length} indexed documents (${eligibleChunks.length} chunks).`,
  });

  // Stage 2: Query Classification (when RAG or analytical tagging is active)
  let classification: QueryClassificationResult | null = null;
  if (generateAnswer) {
    classification = await classifyQuery(query);
    latencies.queryClassificationMs = classification.latencyMs;
    stages.push({
      stage: 'query_classification',
      label: 'Query Intent Classification',
      status: classification.status === 'classified' ? 'executed' : 'unavailable',
      latencyMs: classification.latencyMs,
      candidateCount: null,
      details: `Classified as "${classification.category}"${classification.status === 'classified' ? ` (confidence ${(classification.confidence * 100).toFixed(0)}%)` : ' (fallback heuristic)'}.`,
    });
  } else {
    stages.push({
      stage: 'query_classification',
      label: 'Query Intent Classification',
      status: 'skipped',
      latencyMs: 0,
      candidateCount: null,
      details: 'Skipped when AI answer synthesis is disabled.',
    });
  }

  // Stage 3: Lexical BM25 Retrieval
  let lexicalCandidates: ReturnType<typeof runBm25Retrieval>['candidates'] = [];
  if (mode === 'keyword' || mode === 'hybrid') {
    const bm25 = runBm25Retrieval(query, eligibleChunks, eligibleDocMap, settings, bm25CandidateCap);
    lexicalCandidates = bm25.candidates;
    latencies.bm25RetrievalMs = bm25.latencyMs;
    stages.push({
      stage: 'lexical_retrieval',
      label: 'Lexical Retrieval (Okapi BM25)',
      status: 'executed',
      latencyMs: bm25.latencyMs,
      candidateCount: bm25.candidates.length,
      details: `Evaluated ${bm25.queryTokens.length} tokens (k1=${settings.bm25K1}, b=${settings.bm25B}, cap=${bm25CandidateCap}).`,
    });
  } else {
    stages.push({
      stage: 'lexical_retrieval',
      label: 'Lexical Retrieval (Okapi BM25)',
      status: 'skipped',
      latencyMs: 0,
      candidateCount: null,
      details: 'Skipped in Semantic-only mode.',
    });
  }

  // Stage 4: Embedding Generation & Dense Vector Retrieval
  let semanticCandidates: Awaited<ReturnType<typeof runSemanticRetrieval>>['candidates'] = [];
  if (mode === 'semantic' || mode === 'hybrid') {
    try {
      const sem = await runSemanticRetrieval(query, eligibleChunks, eligibleDocMap, semanticCandidateCap);
      semanticCandidates = sem.candidates;
      latencies.embeddingGenerationMs = sem.embeddingLatencyMs;
      latencies.vectorRetrievalMs = sem.vectorSearchLatencyMs;

      stages.push({
        stage: 'embedding_generation',
        label: `Query & Index Embeddings (${sem.model})`,
        status: 'executed',
        latencyMs: sem.embeddingLatencyMs,
        candidateCount: eligibleChunks.length,
        details:
          sem.embeddedNewChunks > 0
            ? `Embedded query and ${sem.embeddedNewChunks} newly indexed document chunks.`
            : `Embedded query vector; reused cached chunk vectors from persistent FlatIP index.`,
      });

      stages.push({
        stage: 'semantic_retrieval',
        label: 'Dense Vector Similarity Search (FlatIP)',
        status: 'executed',
        latencyMs: sem.vectorSearchLatencyMs,
        candidateCount: sem.candidates.length,
        details: `Retrieved top ${sem.candidates.length} chunks by L2-normalized cosine similarity.`,
      });

      if (sem.missingEmbeddingsCount > 0) {
        warnings.push(`${sem.missingEmbeddingsCount} chunks in the filtered corpus lacked embeddings and were skipped during vector search.`);
      }
    } catch (err) {
      const normalized = normalizeGeminiError(err);
      stages.push({
        stage: 'embedding_generation',
        label: `Query & Index Embeddings (${settings.embeddingModel})`,
        status: 'error',
        latencyMs: 0,
        candidateCount: 0,
        details: normalized.message,
      });
      stages.push({
        stage: 'semantic_retrieval',
        label: 'Dense Vector Similarity Search (FlatIP)',
        status: 'error',
        latencyMs: 0,
        candidateCount: 0,
        details: 'Skipped because query/chunk embeddings could not be generated.',
      });

      if (mode === 'semantic') {
        const failTotal = Math.max(1, Math.round(performance.now() - totalStart));
        latencies.totalLatencyMs = failTotal;
        lucentStore.addQueryLog(
          {
            id: queryId,
            query,
            mode,
            rerankerMode,
            queryCategory: classification?.category || null,
            resultCount: 0,
            topScore: null,
            totalLatencyMs: failTotal,
            latencies,
            ragStatus: 'error',
            failed: true,
            errorMessage: normalized.message,
            timestamp: new Date().toISOString(),
            hasFilters: false,
          },
          null
        );
        res.status(normalized.statusCode).json({ error: normalized.message, code: normalized.code });
        return;
      }

      warnings.push(
        `Dense semantic retrieval failed (${normalized.message}). Hybrid search returned BM25 lexical results only.`
      );
    }
  } else {
    stages.push({
      stage: 'embedding_generation',
      label: `Query & Index Embeddings (${settings.embeddingModel})`,
      status: 'skipped',
      latencyMs: 0,
      candidateCount: null,
      details: 'Skipped in Keyword-only mode.',
    });
    stages.push({
      stage: 'semantic_retrieval',
      label: 'Dense Vector Similarity Search (FlatIP)',
      status: 'skipped',
      latencyMs: 0,
      candidateCount: null,
      details: 'Skipped in Keyword-only mode.',
    });
  }

  // Stage 5: Hybrid Rank Fusion & Deduplication
  const rankCutoff = rerankerMode !== 'none' ? Math.max(topK, hybridCandidateCap, settings.rerankCandidateCount) : topK;
  const ranked = rankCandidates({
    mode,
    lexicalCandidates,
    semanticCandidates,
    settings,
    topK: rankCutoff,
    rrfKOverride: rrfK,
    bm25WeightOverride: bm25Weight,
    semanticWeightOverride: semanticWeight,
  });
  latencies.hybridFusionMs = ranked.latencyMs;

  stages.push({
    stage: 'hybrid_ranking',
    label: mode === 'hybrid' ? 'Hybrid Fusion (RRF / Weighted)' : 'Score Ranking & Document Deduplication',
    status: 'executed',
    latencyMs: ranked.latencyMs,
    candidateCount: ranked.results.length,
    details:
      mode === 'hybrid'
        ? `Fused BM25 + Dense candidates via ${settings.hybridFusionMethod === 'rrf' ? `RRF(k=${rrfK}, w_bm25=${bm25Weight}, w_sem=${semanticWeight})` : `Weighted Linear`} into ${ranked.results.length} deduplicated documents.`
        : `Ranked ${ranked.results.length} unique documents from ${ranked.totalDocumentMatches} matched documents.`,
  });

  // Stage 6: Second-Stage Neural / LLM Reranking
  let finalResults = ranked.results.slice(0, topK);
  let rerankingMetrics: RerankingMetrics = {
    provider: rerankerMode,
    modelName: null,
    documentsReranked: 0,
    latencyMs: 0,
    avgTimePerDocMs: 0,
    status: 'skipped',
  };

  if (rerankerMode !== 'none' && ranked.results.length > 0) {
    try {
      const reranked = await runRerankingStage({
        query,
        mode: rerankerMode,
        candidates: ranked.results,
        settings,
        finalTopK: topK,
      });
      finalResults = reranked.results;
      rerankingMetrics = reranked.metrics;
      latencies.rerankingMs = reranked.metrics.latencyMs;

      stages.push({
        stage: 'neural_reranking',
        label:
          rerankerMode === 'cross_encoder'
            ? `Cross-Encoder Reranking (${reranked.metrics.modelName})`
            : `LLM-Based Reranking (${reranked.metrics.modelName})`,
        status: 'executed',
        latencyMs: reranked.metrics.latencyMs,
        candidateCount: reranked.metrics.documentsReranked,
        details: `Reranked ${reranked.metrics.documentsReranked} candidates (${reranked.metrics.avgTimePerDocMs} ms/doc avg).`,
      });
    } catch (err) {
      const message =
        err instanceof RerankerServiceError ? err.message : normalizeGeminiError(err).message;
      const statusCode = err instanceof RerankerServiceError ? err.statusCode : 503;

      rerankingMetrics = {
        provider: rerankerMode,
        modelName: null,
        documentsReranked: 0,
        latencyMs: 0,
        avgTimePerDocMs: 0,
        status: 'error',
        errorMessage: message,
      };
      stages.push({
        stage: 'neural_reranking',
        label: rerankerMode === 'cross_encoder' ? 'Cross-Encoder Reranking' : 'LLM-Based Reranking',
        status: 'error',
        latencyMs: 0,
        candidateCount: 0,
        details: message,
      });

      // Strictly do NOT silently substitute another model if cross-encoder was explicitly requested!
      if (rerankerMode === 'cross_encoder') {
        const failTotal = Math.max(1, Math.round(performance.now() - totalStart));
        latencies.totalLatencyMs = failTotal;
        lucentStore.addQueryLog(
          {
            id: queryId,
            query,
            mode,
            rerankerMode,
            queryCategory: classification?.category || null,
            resultCount: 0,
            topScore: null,
            totalLatencyMs: failTotal,
            latencies,
            ragStatus: 'skipped',
            failed: true,
            errorMessage: message,
            timestamp: new Date().toISOString(),
            hasFilters: false,
          },
          null
        );
        res.status(statusCode).json({
          error: message,
          code: err instanceof RerankerServiceError ? err.code : 'RERANKER_ERROR',
        });
        return;
      }

      warnings.push(`LLM-based reranking failed (${message}). Displaying first-stage ranking results.`);
    }
  } else {
    stages.push({
      stage: 'neural_reranking',
      label: 'Second-Stage Reranking',
      status: 'skipped',
      latencyMs: 0,
      candidateCount: null,
      details: 'Disabled for this search query. Enable Cross-Encoder or LLM Reranker in the search controls.',
    });
  }

  // Stage 7: Evidence-Grounded RAG Answer Generation
  const answer = await generateGroundedAnswer({
    query,
    results: finalResults,
    enabled: generateAnswer,
    answerStyle,
    queryCategory: classification?.category || null,
  });
  latencies.answerGenerationMs = answer.latencyMs;

  stages.push({
    stage: 'rag_generation',
    label: `Grounded RAG Synthesis (${serverConfig.generationModel})`,
    status:
      answer.status === 'generated'
        ? 'executed'
        : answer.status === 'error'
          ? 'error'
          : 'skipped',
    latencyMs: answer.latencyMs,
    candidateCount: answer.sourcesUsed.length,
    details:
      answer.status === 'generated'
        ? `Synthesized "${answerStyle}" answer grounded in ${answer.sourcesUsed.length} validated source citations.${answer.rejectedCitationCount > 0 ? ` Filtered ${answer.rejectedCitationCount} invalid citation(s).` : ''}`
        : answer.status === 'error'
          ? answer.errorMessage || 'Answer synthesis failed.'
          : answer.status === 'no_evidence'
            ? 'No evidence passages retrieved.'
            : 'Disabled by search configuration.',
  });

  const totalLatencyMs = Math.max(1, Math.round(performance.now() - totalStart));
  latencies.totalLatencyMs = totalLatencyMs;

  const trace: SearchPipelineTrace = {
    queryId,
    query,
    mode,
    rerankerMode,
    timestamp: new Date().toISOString(),
    totalLatencyMs,
    latencies,
    corpusDocumentsScanned: eligibleDocs.length,
    corpusChunksScanned: eligibleChunks.length,
    stages,
    warnings,
  };

  const hasFilters = Boolean(
    (filters.documentType && filters.documentType !== 'all') ||
      (filters.source && filters.source !== 'all') ||
      filters.dateFrom ||
      filters.dateTo ||
      filters.includeDemo === false
  );

  lucentStore.addQueryLog(
    {
      id: queryId,
      query,
      mode,
      rerankerMode,
      queryCategory: classification?.category || null,
      resultCount: finalResults.length,
      topScore: finalResults.length > 0 ? finalResults[0].relevanceScore : null,
      totalLatencyMs,
      latencies,
      ragStatus: answer.status,
      failed: false,
      timestamp: trace.timestamp,
      hasFilters,
    },
    trace
  );

  const payload: SearchResponsePayload = {
    queryId,
    query,
    mode,
    rerankerMode,
    classification,
    appliedFilters: filters,
    totalMatches: ranked.totalDocumentMatches,
    results: finalResults,
    reranking: rerankingMetrics,
    answer,
    trace,
    warnings,
  };

  res.json(payload);
});

function semanticCandidatesLimit(n: number): number {
  return Math.max(1, Math.min(100, Math.floor(n)));
}

// Side-by-Side Search Mode & Reranker Comparison Route
app.post('/api/search/compare', async (req, res) => {
  const body = req.body as {
    query?: string;
    filters?: SearchRequestPayload['filters'];
    topK?: number;
    includeCrossEncoder?: boolean;
    includeLlmReranker?: boolean;
  };
  const query = sanitizeShortString(body.query, 600);
  if (!query) {
    res.status(400).json({ error: 'Query is required for model comparison.' });
    return;
  }

  const settings = lucentStore.getSettings();
  const topK = Math.max(1, Math.min(10, Math.floor(Number(body.topK) || 5)));
  const eligibleDocs = filterDocuments(lucentStore.getDocuments(), body.filters);
  const eligibleDocMap = new Map(eligibleDocs.map((d) => [d.id, d]));
  const eligibleChunks = lucentStore.getChunks().filter((c) => eligibleDocMap.has(c.documentId));

  const methods: ModeComparisonEntry[] = [];

  // 1. BM25
  const bm25Start = performance.now();
  const bm25 = runBm25Retrieval(query, eligibleChunks, eligibleDocMap, settings);
  const bm25Ranked = rankCandidates({
    mode: 'keyword',
    lexicalCandidates: bm25.candidates,
    semanticCandidates: [],
    settings,
    topK,
  });
  methods.push({
    method: 'bm25',
    label: 'BM25 Lexical',
    status: 'completed',
    latencyMs: Math.max(1, Math.round(performance.now() - bm25Start)),
    results: bm25Ranked.results.map((r) => ({
      rank: r.rank,
      documentId: r.documentId,
      title: r.title,
      score: r.relevanceScore,
    })),
  });

  // 2. Dense Retrieval
  let semanticCandidates: Awaited<ReturnType<typeof runSemanticRetrieval>>['candidates'] = [];
  const denseStart = performance.now();
  try {
    const sem = await runSemanticRetrieval(query, eligibleChunks, eligibleDocMap);
    semanticCandidates = sem.candidates;
    const denseRanked = rankCandidates({
      mode: 'semantic',
      lexicalCandidates: [],
      semanticCandidates,
      settings,
      topK,
    });
    methods.push({
      method: 'dense',
      label: `Dense (${sem.model})`,
      status: 'completed',
      latencyMs: Math.max(1, Math.round(performance.now() - denseStart)),
      results: denseRanked.results.map((r) => ({
        rank: r.rank,
        documentId: r.documentId,
        title: r.title,
        score: r.relevanceScore,
      })),
    });
  } catch (err) {
    methods.push({
      method: 'dense',
      label: `Dense (${settings.embeddingModel})`,
      status: 'error',
      errorMessage: normalizeGeminiError(err).message,
      latencyMs: Math.max(1, Math.round(performance.now() - denseStart)),
      results: [],
    });
  }

  // 3. Hybrid RRF
  const hybridStart = performance.now();
  const hybridRanked = rankCandidates({
    mode: 'hybrid',
    lexicalCandidates: bm25.candidates,
    semanticCandidates,
    settings,
    topK: Math.max(topK, settings.rerankCandidateCount || 10),
  });
  methods.push({
    method: 'hybrid',
    label: `Hybrid RRF (k=${settings.rrfK})`,
    status: 'completed',
    latencyMs: Math.max(1, Math.round(performance.now() - hybridStart)),
    results: hybridRanked.results.slice(0, topK).map((r) => ({
      rank: r.rank,
      documentId: r.documentId,
      title: r.title,
      score: r.relevanceScore,
    })),
  });

  // 4. Optional Cross-Encoder Reranking
  if (body.includeCrossEncoder) {
    const ceStart = performance.now();
    try {
      const ce = await runRerankingStage({
        query,
        mode: 'cross_encoder',
        candidates: hybridRanked.results,
        settings,
        finalTopK: topK,
      });
      methods.push({
        method: 'hybrid_cross_encoder',
        label: `Cross-Encoder (${ce.metrics.modelName || 'MS MARCO'})`,
        status: 'completed',
        latencyMs: ce.metrics.latencyMs,
        results: ce.results.map((r) => ({
          rank: r.rank,
          documentId: r.documentId,
          title: r.title,
          score: r.relevanceScore,
          originalRank: r.explanation.originalPosition,
          rankMovement: r.explanation.rankMovement,
        })),
      });
    } catch (err) {
      methods.push({
        method: 'hybrid_cross_encoder',
        label: 'Cross-Encoder Reranker',
        status: 'error',
        errorMessage: err instanceof RerankerServiceError ? err.message : normalizeGeminiError(err).message,
        latencyMs: Math.max(1, Math.round(performance.now() - ceStart)),
        results: [],
      });
    }
  }

  // 5. Optional LLM-Based Reranking
  if (body.includeLlmReranker) {
    const llmStart = performance.now();
    try {
      const llm = await runRerankingStage({
        query,
        mode: 'llm',
        candidates: hybridRanked.results,
        settings,
        finalTopK: topK,
      });
      methods.push({
        method: 'hybrid_llm_rerank',
        label: `LLM Reranker (${llm.metrics.modelName || serverConfig.generationModel})`,
        status: 'completed',
        latencyMs: llm.metrics.latencyMs,
        results: llm.results.map((r) => ({
          rank: r.rank,
          documentId: r.documentId,
          title: r.title,
          score: r.relevanceScore,
          originalRank: r.explanation.originalPosition,
          rankMovement: r.explanation.rankMovement,
        })),
      });
    } catch (err) {
      methods.push({
        method: 'hybrid_llm_rerank',
        label: 'LLM-Based Reranker',
        status: 'error',
        errorMessage: err instanceof RerankerServiceError ? err.message : normalizeGeminiError(err).message,
        latencyMs: Math.max(1, Math.round(performance.now() - llmStart)),
        results: [],
      });
    }
  }

  const responsePayload: SearchComparisonResponse = {
    query,
    timestamp: new Date().toISOString(),
    corpusDocumentsScanned: eligibleDocs.length,
    methods,
  };
  res.json(responsePayload);
});

// Retrieval Pipeline Inspector Route
app.get('/api/pipeline/status', async (_req, res) => {
  const runtime = getPublicRuntimeConfig();
  const settings = lucentStore.getSettings();
  const docs = lucentStore.getDocuments();
  const chunks = lucentStore.getChunks();
  const activeModel = settings.embeddingModel || serverConfig.embeddingModel;
  const embeddedChunks = chunks.filter((c) =>
    vectorIndex.hasMatchingVector(c.id, c.contentHash, activeModel)
  ).length;
  const crossEncoder = await checkCrossEncoderServiceStatus();

  res.json({
    runtime,
    settings,
    crossEncoder,
    vectorIndex: vectorIndex.getStats(),
    indexSummary: {
      totalDocuments: docs.length,
      totalChunks: chunks.length,
      embeddedChunks,
    },
    lastTrace: lucentStore.getLastTrace(),
  });
});

// ML Evaluation Laboratory Routes
app.get('/api/evaluation', (_req, res) => {
  res.json({
    queries: lucentStore.getEvaluationQueries(),
    runs: lucentStore.getEvaluationRuns(),
  });
});

app.post('/api/evaluation/queries', (req, res) => {
  const body = req.body as Record<string, unknown>;
  const query = sanitizeShortString(body.query, 400);
  const category = sanitizeShortString(body.category, 80) || 'Custom';
  const notes = sanitizeShortString(body.notes, 300);

  const grades: Record<string, number> = {};
  if (body.relevanceGrades && typeof body.relevanceGrades === 'object') {
    for (const [docId, grade] of Object.entries(body.relevanceGrades as Record<string, unknown>)) {
      if (lucentStore.getDocumentById(docId) && typeof grade === 'number' && grade >= 0 && grade <= 3) {
        grades[docId] = Math.round(grade);
      }
    }
  }
  if (Array.isArray(body.relevantDocumentIds)) {
    for (const id of body.relevantDocumentIds) {
      if (typeof id === 'string' && lucentStore.getDocumentById(id) && !(id in grades)) {
        grades[id] = 2;
      }
    }
  }

  const relevantDocumentIds = Object.entries(grades)
    .filter(([, g]) => g > 0)
    .map(([id]) => id);

  if (!query) {
    res.status(400).json({ error: 'Evaluation query text is required.' });
    return;
  }
  if (relevantDocumentIds.length === 0) {
    res.status(400).json({ error: 'Assign at least one relevant document (grade 1–3) from the library.' });
    return;
  }

  const created = {
    id: `eval-q-${crypto.randomUUID().slice(0, 10)}`,
    query,
    relevantDocumentIds,
    relevanceGrades: grades,
    category,
    notes,
    isDemo: false,
  };

  lucentStore.addEvaluationQuery(created);
  res.status(201).json({ query: created });
});

app.patch('/api/evaluation/queries/:id', (req, res) => {
  const body = req.body as Record<string, unknown>;
  const patch: Record<string, unknown> = {};

  if (typeof body.query === 'string' && body.query.trim()) {
    patch.query = sanitizeShortString(body.query, 400);
  }
  if (typeof body.category === 'string') {
    patch.category = sanitizeShortString(body.category, 80) || 'Custom';
  }
  if (typeof body.notes === 'string') {
    patch.notes = sanitizeShortString(body.notes, 300);
  }
  if (body.relevanceGrades && typeof body.relevanceGrades === 'object') {
    const grades: Record<string, number> = {};
    for (const [docId, grade] of Object.entries(body.relevanceGrades as Record<string, unknown>)) {
      if (lucentStore.getDocumentById(docId) && typeof grade === 'number' && grade >= 0 && grade <= 3) {
        grades[docId] = Math.round(grade);
      }
    }
    const relevantDocumentIds = Object.entries(grades)
      .filter(([, g]) => g > 0)
      .map(([id]) => id);
    if (relevantDocumentIds.length === 0) {
      res.status(400).json({ error: 'At least one document must have relevance grade >= 1.' });
      return;
    }
    patch.relevanceGrades = grades;
    patch.relevantDocumentIds = relevantDocumentIds;
  }

  const updated = lucentStore.updateEvaluationQuery(req.params.id, patch);
  if (!updated) {
    res.status(404).json({ error: 'Evaluation query not found.' });
    return;
  }
  res.json({ query: updated });
});

app.delete('/api/evaluation/queries/:id', (req, res) => {
  const deleted = lucentStore.deleteEvaluationQuery(req.params.id);
  if (!deleted) {
    res.status(404).json({ error: 'Evaluation query not found.' });
    return;
  }
  res.json({ deleted: true });
});

app.post('/api/evaluation/import', (req, res) => {
  try {
    const body = req.body as { format?: string; content?: string };
    const format = body.format === 'csv' ? 'csv' : 'json';
    const content = typeof body.content === 'string' ? body.content : '';
    if (!content.trim()) {
      res.status(400).json({ error: 'Dataset content cannot be empty.' });
      return;
    }
    const imported = parseEvaluationDatasetImport(format, content);
    res.status(201).json({ importedCount: imported.length, queries: imported });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to parse evaluation dataset.';
    res.status(400).json({ error: message });
  }
});

app.post('/api/evaluation/run', async (req, res) => {
  try {
    const body = req.body as {
      methods?: EvaluationMethodId[];
      k?: number;
      datasetScope?: 'all' | 'human_only' | 'demo_only';
    };
    const requestedMethods = Array.isArray(body.methods)
      ? body.methods.filter((m): m is EvaluationMethodId => VALID_EVAL_METHODS.has(m))
      : (['bm25', 'dense', 'hybrid'] as EvaluationMethodId[]);
    const methods: EvaluationMethodId[] =
      requestedMethods.length > 0 ? requestedMethods : ['bm25', 'dense', 'hybrid'];
    const k = typeof body.k === 'number' ? body.k : 5;
    const datasetScope =
      body.datasetScope === 'human_only' || body.datasetScope === 'demo_only'
        ? body.datasetScope
        : 'all';

    const run = await runEvaluationSuite({ methods, k, datasetScope });
    res.json({ run });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to run evaluation experiment.';
    res.status(400).json({ error: message });
  }
});

app.get('/api/evaluation/runs/:id/report', (req, res) => {
  const run = lucentStore.getEvaluationRunById(req.params.id);
  if (!run) {
    res.status(404).json({ error: 'Experiment run not found.' });
    return;
  }
  const markdown = buildExperimentReportMarkdown(run);
  res.setHeader('Content-Type', 'text/markdown; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename="lucent-report-${run.id}.md"`);
  res.send(markdown);
});

// Analytics Route (100% real corpus, vector index, and logged search telemetry)
app.get('/api/analytics', (_req, res) => {
  const docs = lucentStore.getDocuments();
  const chunks = lucentStore.getChunks();
  const logs = lucentStore.getQueryLogs();
  const activeModel = lucentStore.getSettings().embeddingModel || serverConfig.embeddingModel;

  const byTypeMap = new Map<string, number>();
  const bySourceMap = new Map<string, number>();
  for (const d of docs) {
    byTypeMap.set(d.documentType, (byTypeMap.get(d.documentType) || 0) + 1);
    const srcLabel = d.source || 'Source not specified';
    bySourceMap.set(srcLabel, (bySourceMap.get(srcLabel) || 0) + 1);
  }

  const modes: SearchMode[] = ['keyword', 'semantic', 'hybrid'];
  const modeBreakdown = modes.map((mode) => {
    const modeLogs = logs.filter((l) => l.mode === mode);
    const avgLatencyMs =
      modeLogs.length > 0
        ? Math.round(modeLogs.reduce((s, l) => s + l.totalLatencyMs, 0) / modeLogs.length)
        : 0;
    return { mode, count: modeLogs.length, avgLatencyMs };
  });

  const catMap = new Map<string, number>();
  for (const log of logs) {
    const cat = log.queryCategory || 'unclassified';
    catMap.set(cat, (catMap.get(cat) || 0) + 1);
  }

  const successfulLogs = logs.filter((l) => !l.failed);
  const avgStageLatencies = emptyLatencies(0);
  if (successfulLogs.length > 0) {
    const n = successfulLogs.length;
    const sumField = (key: keyof StageLatencyBreakdown) =>
      Math.round(successfulLogs.reduce((s, l) => s + (l.latencies?.[key] || 0), 0) / n);
    avgStageLatencies.queryProcessingMs = sumField('queryProcessingMs');
    avgStageLatencies.queryClassificationMs = sumField('queryClassificationMs');
    avgStageLatencies.embeddingGenerationMs = sumField('embeddingGenerationMs');
    avgStageLatencies.bm25RetrievalMs = sumField('bm25RetrievalMs');
    avgStageLatencies.vectorRetrievalMs = sumField('vectorRetrievalMs');
    avgStageLatencies.hybridFusionMs = sumField('hybridFusionMs');
    avgStageLatencies.rerankingMs = sumField('rerankingMs');
    avgStageLatencies.answerGenerationMs = sumField('answerGenerationMs');
    avgStageLatencies.totalLatencyMs = sumField('totalLatencyMs');
  }

  const latencySeries = [...logs]
    .slice(0, 25)
    .reverse()
    .map((l, idx) => ({
      index: idx + 1,
      timestamp: l.timestamp,
      mode: l.mode,
      totalLatencyMs: l.totalLatencyMs,
      topScore: l.topScore,
    }));

  const snapshot: AnalyticsSnapshot = {
    corpus: {
      totalDocuments: docs.length,
      demoDocuments: docs.filter((d) => d.isDemo).length,
      userDocuments: docs.filter((d) => !d.isDemo).length,
      totalChunks: chunks.length,
      embeddedChunks: chunks.filter((c) =>
        vectorIndex.hasMatchingVector(c.id, c.contentHash, activeModel)
      ).length,
      failedEmbeddingDocs: docs.filter((d) => d.embeddingStatus === 'failed').length,
      totalWords: docs.reduce((s, d) => s + d.wordCount, 0),
      byType: Array.from(byTypeMap.entries())
        .map(([type, count]) => ({ type, count }))
        .sort((a, b) => b.count - a.count),
      bySource: Array.from(bySourceMap.entries())
        .map(([source, count]) => ({ source, count }))
        .sort((a, b) => b.count - a.count),
    },
    vectorIndex: vectorIndex.getStats(),
    queries: {
      totalQueries: logs.length,
      failedQueries: logs.filter((l) => l.failed).length,
      avgLatencyMs:
        logs.length > 0
          ? Math.round(logs.reduce((s, l) => s + l.totalLatencyMs, 0) / logs.length)
          : 0,
      zeroResultCount: logs.filter((l) => !l.failed && l.resultCount === 0).length,
      modeBreakdown,
      categoryBreakdown: Array.from(catMap.entries())
        .map(([category, count]) => ({ category, count }))
        .sort((a, b) => b.count - a.count),
      avgStageLatencies,
      latencySeries,
      recentQueries: logs.slice(0, 40),
    },
  };

  res.json(snapshot);
});

app.delete('/api/analytics/queries', (_req, res) => {
  lucentStore.clearQueryLogs();
  res.json({ cleared: true });
});

// Settings Routes
app.get('/api/settings', async (_req, res) => {
  const crossEncoder = await checkCrossEncoderServiceStatus();
  res.json({
    settings: lucentStore.getSettings(),
    runtime: getPublicRuntimeConfig(),
    crossEncoder,
  });
});

app.patch('/api/settings', (req, res) => {
  const body = req.body as Partial<EngineSettings>;
  const current = lucentStore.getSettings();
  const next: Partial<EngineSettings> = {};

  if (body.defaultMode && VALID_SEARCH_MODES.has(body.defaultMode)) {
    next.defaultMode = body.defaultMode;
  }
  if (body.defaultRerankerMode && VALID_RERANKER_MODES.has(body.defaultRerankerMode)) {
    next.defaultRerankerMode = body.defaultRerankerMode;
  }
  if (typeof body.defaultAnswerStyle === 'string') {
    next.defaultAnswerStyle = normalizeAnswerStyle(body.defaultAnswerStyle);
  }
  if (typeof body.topK === 'number' && Number.isFinite(body.topK)) {
    next.topK = Math.max(1, Math.min(20, Math.round(body.topK)));
  }
  if (typeof body.bm25Candidates === 'number' && Number.isFinite(body.bm25Candidates)) {
    next.bm25Candidates = Math.max(5, Math.min(100, Math.round(body.bm25Candidates)));
  }
  if (typeof body.semanticCandidates === 'number' && Number.isFinite(body.semanticCandidates)) {
    next.semanticCandidates = Math.max(5, Math.min(100, Math.round(body.semanticCandidates)));
  }
  if (typeof body.hybridCandidates === 'number' && Number.isFinite(body.hybridCandidates)) {
    next.hybridCandidates = Math.max(3, Math.min(50, Math.round(body.hybridCandidates)));
  }
  if (body.hybridFusionMethod === 'rrf' || body.hybridFusionMethod === 'weighted') {
    next.hybridFusionMethod = body.hybridFusionMethod;
  }
  if (typeof body.hybridAlpha === 'number' && Number.isFinite(body.hybridAlpha)) {
    next.hybridAlpha = Math.max(0, Math.min(1, Math.round(body.hybridAlpha * 100) / 100));
  }
  if (typeof body.bm25Weight === 'number' && Number.isFinite(body.bm25Weight)) {
    next.bm25Weight = Math.max(0, Math.min(5, Math.round(body.bm25Weight * 100) / 100));
  }
  if (typeof body.semanticWeight === 'number' && Number.isFinite(body.semanticWeight)) {
    next.semanticWeight = Math.max(0, Math.min(5, Math.round(body.semanticWeight * 100) / 100));
  }
  if (typeof body.rrfK === 'number' && Number.isFinite(body.rrfK)) {
    next.rrfK = Math.max(1, Math.min(200, Math.round(body.rrfK)));
  }
  if (typeof body.bm25K1 === 'number' && Number.isFinite(body.bm25K1)) {
    next.bm25K1 = Math.max(0.5, Math.min(3.0, Math.round(body.bm25K1 * 100) / 100));
  }
  if (typeof body.bm25B === 'number' && Number.isFinite(body.bm25B)) {
    next.bm25B = Math.max(0, Math.min(1, Math.round(body.bm25B * 100) / 100));
  }
  if (typeof body.chunkSize === 'number' && Number.isFinite(body.chunkSize)) {
    next.chunkSize = Math.max(60, Math.min(600, Math.round(body.chunkSize)));
  }
  if (typeof body.chunkOverlap === 'number' && Number.isFinite(body.chunkOverlap)) {
    const maxOverlap = Math.floor((next.chunkSize ?? current.chunkSize) * 0.5);
    next.chunkOverlap = Math.max(0, Math.min(maxOverlap, Math.round(body.chunkOverlap)));
  }
  if (typeof body.embeddingModel === 'string' && SUPPORTED_EMBEDDING_MODELS.includes(body.embeddingModel.trim())) {
    next.embeddingModel = body.embeddingModel.trim();
  }
  if (typeof body.embeddingBatchSize === 'number' && Number.isFinite(body.embeddingBatchSize)) {
    next.embeddingBatchSize = Math.max(1, Math.min(32, Math.round(body.embeddingBatchSize)));
  }
  if (typeof body.autoEmbedOnUpload === 'boolean') {
    next.autoEmbedOnUpload = body.autoEmbedOnUpload;
  }
  if (typeof body.rerankCandidateCount === 'number' && Number.isFinite(body.rerankCandidateCount)) {
    next.rerankCandidateCount = Math.max(2, Math.min(30, Math.round(body.rerankCandidateCount)));
  }
  if (typeof body.rerankBatchSize === 'number' && Number.isFinite(body.rerankBatchSize)) {
    next.rerankBatchSize = Math.max(1, Math.min(32, Math.round(body.rerankBatchSize)));
  }
  if (typeof body.rerankMaxSeqLength === 'number' && Number.isFinite(body.rerankMaxSeqLength)) {
    next.rerankMaxSeqLength = Math.max(128, Math.min(1024, Math.round(body.rerankMaxSeqLength)));
  }
  if (typeof body.rerankThreshold === 'number' && Number.isFinite(body.rerankThreshold)) {
    next.rerankThreshold = Math.max(-10, Math.min(1, Math.round(body.rerankThreshold * 100) / 100));
  }
  if (typeof body.enableRagByDefault === 'boolean') {
    next.enableRagByDefault = body.enableRagByDefault;
  }

  const updated = lucentStore.updateSettings(next);

  let rechunked: { documentsRechunked: number; totalChunks: number } | null = null;
  if (updated.chunkSize !== current.chunkSize || updated.chunkOverlap !== current.chunkOverlap) {
    rechunked = rechunkCorpus(updated.chunkSize, updated.chunkOverlap);
  }

  res.json({ settings: updated, rechunked });
});

// Express multer & global error handler
app.use((err: unknown, _req: express.Request, res: express.Response, next: express.NextFunction) => {
  if (err instanceof multer.MulterError) {
    res.status(400).json({ error: `File upload error: ${err.message}` });
    return;
  }
  if (err) {
    const message = err instanceof Error ? err.message : 'Internal server error.';
    res.status(500).json({ error: message });
    return;
  }
  next();
});

async function startServer() {
  if (serverConfig.nodeEnv !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(distDir, 'index.html'));
    });
  }

  app.listen(serverConfig.port, '0.0.0.0', () => {
    console.log(`[Lucent] Server running on http://0.0.0.0:${serverConfig.port}`);
  });
}

startServer();
