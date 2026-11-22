export type SearchMode = 'keyword' | 'semantic' | 'hybrid';

export type DocumentType =
  | 'Research Paper'
  | 'Technical Report'
  | 'Policy Brief'
  | 'Clinical Study'
  | 'Dataset Notes'
  | 'Uploaded Document';

export type HybridFusionMethod = 'rrf' | 'weighted';

export type RerankerMode = 'none' | 'cross_encoder' | 'llm';

export type QueryCategory =
  | 'factual'
  | 'comparative'
  | 'historical'
  | 'analytical'
  | 'explanatory';

export type AnswerStyle = 'concise' | 'detailed' | 'bullets' | 'comparative';

export type ProcessingStatus = 'pending' | 'processing' | 'indexed' | 'error';

export type EmbeddingStatus = 'pending' | 'embedding' | 'completed' | 'partial' | 'failed';

export type EvaluationMethodId =
  | 'bm25'
  | 'dense'
  | 'hybrid'
  | 'hybrid_cross_encoder'
  | 'hybrid_llm_rerank';

export interface ResearchDocument {
  id: string;
  title: string;
  source: string | null;
  documentType: DocumentType;
  publishedDate: string | null;
  authors: string[];
  sourceUrl: string | null;
  originalFilename: string | null;
  storedFilePath: string | null;
  mimeType: string;
  fileSizeBytes: number;
  pageCount: number | null;
  contentHash: string;
  content: string;
  summary: string;
  wordCount: number;
  chunkCount: number;
  embeddedChunkCount: number;
  processingStatus: ProcessingStatus;
  processingError: string | null;
  embeddingStatus: EmbeddingStatus;
  embeddingError: string | null;
  embeddingModel: string | null;
  isDemo: boolean;
  uploadedAt: string;
  createdAt: string;
  updatedAt: string;
}

export interface DocumentChunk {
  id: string;
  documentId: string;
  documentTitle: string;
  source: string | null;
  chunkIndex: number;
  pageNumber: number | null;
  text: string;
  startOffset: number;
  endOffset: number;
  wordCount: number;
  contentHash: string;
  hasEmbedding?: boolean;
  embeddingModel?: string | null;
}

export interface VectorIndexStats {
  backend: string;
  metric: string;
  vectorCount: number;
  dimensions: number | null;
  storageBytes: number;
  models: Array<{ model: string; count: number }>;
  persistedPath: string;
}

export interface SearchFilters {
  documentType?: string;
  source?: string;
  dateFrom?: string;
  dateTo?: string;
  includeDemo?: boolean;
}

export interface SearchRequestPayload {
  query: string;
  mode: SearchMode;
  rerankerMode?: RerankerMode;
  answerStyle?: AnswerStyle;
  filters?: SearchFilters;
  topK?: number;
  bm25Candidates?: number;
  semanticCandidates?: number;
  hybridCandidates?: number;
  bm25Weight?: number;
  semanticWeight?: number;
  rrfK?: number;
  generateAnswer?: boolean;
}

export interface QueryClassificationResult {
  category: QueryCategory;
  confidence: number;
  reasoning: string;
  model: string | null;
  status: 'classified' | 'fallback' | 'error';
  latencyMs: number;
}

export interface RetrievalExplanation {
  summary: string;
  matchedTerms: string[];
  bm25Score: number | null;
  bm25Rank: number | null;
  semanticScore: number | null;
  semanticRank: number | null;
  rrfScore: number | null;
  fusionMethod: string | null;
  hybridRank: number | null;
  originalPosition: number;
  newPosition: number;
  rankMovement: number; // positive means promoted up in rank
  rerankScore: number | null;
  rerankProvider: 'cross_encoder' | 'llm' | null;
  rerankModel: string | null;
  rerankExplanation: string | null;
  finalScore: number;
  chunkId: string;
  chunkIndex: number;
  pageNumber: number | null;
  totalChunksInDocument: number;
  matchingChunksCount: number;
}

export interface SearchResultItem {
  rank: number;
  documentId: string;
  chunkId: string;
  title: string;
  source: string | null;
  documentType: DocumentType;
  publishedDate: string | null;
  pageNumber: number | null;
  authors: string[];
  sourceUrl: string | null;
  originalFilename: string | null;
  hasUploadedFile: boolean;
  isDemo: boolean;
  excerpt: string;
  chunkText: string;
  relevanceScore: number;
  explanation: RetrievalExplanation;
}

export interface RagFinding {
  claim: string;
  evidenceType: 'fact' | 'interpretation';
  citationIndices: number[];
}

export interface RagSourceReference {
  citationIndex: number;
  chunkId: string;
  documentId: string;
  title: string;
  source: string | null;
  publishedDate: string | null;
  pageNumber: number | null;
  excerpt: string;
  sourceUrl: string | null;
  hasUploadedFile: boolean;
  isDemo: boolean;
}

export interface RagAnswer {
  status: 'generated' | 'skipped' | 'error' | 'no_evidence';
  model: string | null;
  answerStyle: AnswerStyle;
  queryCategory: QueryCategory | null;
  summary: string;
  keyFindings: RagFinding[];
  conflictingEvidence: string | null;
  evidenceGaps: string | null;
  sourcesUsed: RagSourceReference[];
  rejectedCitationCount: number;
  latencyMs: number;
  errorMessage?: string;
}

export interface PipelineStageTrace {
  stage:
    | 'query_processing'
    | 'query_classification'
    | 'embedding_generation'
    | 'lexical_retrieval'
    | 'semantic_retrieval'
    | 'hybrid_ranking'
    | 'neural_reranking'
    | 'rag_generation';
  label: string;
  status: 'executed' | 'skipped' | 'unavailable' | 'error';
  latencyMs: number;
  candidateCount: number | null;
  details: string;
}

export interface StageLatencyBreakdown {
  queryProcessingMs: number;
  queryClassificationMs: number;
  embeddingGenerationMs: number;
  bm25RetrievalMs: number;
  vectorRetrievalMs: number;
  hybridFusionMs: number;
  rerankingMs: number;
  answerGenerationMs: number;
  totalLatencyMs: number;
}

export interface RerankingMetrics {
  provider: RerankerMode;
  modelName: string | null;
  documentsReranked: number;
  latencyMs: number;
  avgTimePerDocMs: number;
  status: 'executed' | 'skipped' | 'error';
  errorMessage?: string;
}

export interface SearchPipelineTrace {
  queryId: string;
  query: string;
  mode: SearchMode;
  rerankerMode: RerankerMode;
  timestamp: string;
  totalLatencyMs: number;
  latencies: StageLatencyBreakdown;
  corpusDocumentsScanned: number;
  corpusChunksScanned: number;
  stages: PipelineStageTrace[];
  warnings: string[];
}

export interface SearchResponsePayload {
  queryId: string;
  query: string;
  mode: SearchMode;
  rerankerMode: RerankerMode;
  classification: QueryClassificationResult | null;
  appliedFilters: SearchFilters;
  totalMatches: number;
  results: SearchResultItem[];
  reranking: RerankingMetrics;
  answer: RagAnswer;
  trace: SearchPipelineTrace;
  warnings: string[];
}

export interface ModeComparisonEntry {
  method: EvaluationMethodId;
  label: string;
  status: 'completed' | 'error';
  errorMessage?: string;
  latencyMs: number;
  results: Array<{
    rank: number;
    documentId: string;
    title: string;
    score: number;
    originalRank?: number;
    rankMovement?: number;
  }>;
}

export interface SearchComparisonResponse {
  query: string;
  timestamp: string;
  corpusDocumentsScanned: number;
  methods: ModeComparisonEntry[];
}

export interface CrossEncoderServiceStatus {
  configured: boolean;
  serviceUrlMasked: string | null;
  healthy: boolean;
  modelName: string | null;
  modelVersion: string | null;
  device: string | null;
  maxSequenceLength: number | null;
  errorMessage: string | null;
}

export interface QueryLogEntry {
  id: string;
  query: string;
  mode: SearchMode;
  rerankerMode: RerankerMode;
  queryCategory: QueryCategory | null;
  resultCount: number;
  topScore: number | null;
  totalLatencyMs: number;
  latencies: StageLatencyBreakdown;
  ragStatus: RagAnswer['status'];
  failed: boolean;
  errorMessage?: string;
  timestamp: string;
  hasFilters: boolean;
}

export interface EvaluationQuery {
  id: string;
  query: string;
  relevantDocumentIds: string[];
  relevanceGrades: Record<string, number>; // 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant
  category: string;
  notes: string;
  isDemo: boolean;
}

export interface EvaluationQueryResult {
  queryId: string;
  query: string;
  relevantDocumentIds: string[];
  relevanceGrades: Record<string, number>;
  retrievedDocumentIds: string[];
  precisionAtK: number;
  recallAtK: number;
  mrr: number;
  averagePrecision: number;
  ndcgAtK: number;
  latencyMs: number;
  rankMovements?: Array<{ documentId: string; fromRank: number; toRank: number; delta: number }>;
}

export interface EvaluationModeReport {
  method: EvaluationMethodId;
  label: string;
  modelName: string;
  modelVersion: string;
  status: 'completed' | 'error';
  errorMessage?: string;
  k: number;
  avgPrecisionAtK: number;
  avgRecallAtK: number;
  meanReciprocalRank: number;
  meanAveragePrecision: number;
  avgNdcgAtK: number;
  avgLatencyMs: number;
  perQuery: EvaluationQueryResult[];
}

export interface EvaluationErrorAnalysis {
  zeroRelevantQueries: Array<{ method: EvaluationMethodId; queryId: string; query: string }>;
  rankedTooLow: Array<{
    method: EvaluationMethodId;
    queryId: string;
    query: string;
    documentId: string;
    rank: number;
  }>;
  poorRecallQueries: Array<{
    method: EvaluationMethodId;
    queryId: string;
    query: string;
    recallAtK: number;
  }>;
  rerankingRegressions: Array<{
    rerankerMethod: EvaluationMethodId;
    queryId: string;
    query: string;
    hybridNdcg: number;
    rerankedNdcg: number;
    delta: number;
  }>;
  keywordVsSemanticDivergence: Array<{
    queryId: string;
    query: string;
    bm25Ndcg: number;
    denseNdcg: number;
    winner: 'bm25' | 'dense' | 'tie';
  }>;
}

export interface EvaluationRunRecord {
  id: string;
  timestamp: string;
  datasetVersion: string;
  datasetLabel: string;
  k: number;
  queryCount: number;
  demoQueryCount: number;
  humanQueryCount: number;
  configSnapshot: EngineSettings;
  reports: EvaluationModeReport[];
  errorAnalysis: EvaluationErrorAnalysis;
}

export interface EngineSettings {
  defaultMode: SearchMode;
  defaultRerankerMode: RerankerMode;
  defaultAnswerStyle: AnswerStyle;
  topK: number;
  bm25Candidates: number;
  semanticCandidates: number;
  hybridCandidates: number;
  hybridFusionMethod: HybridFusionMethod;
  hybridAlpha: number;
  bm25Weight: number;
  semanticWeight: number;
  rrfK: number;
  bm25K1: number;
  bm25B: number;
  chunkSize: number;
  chunkOverlap: number;
  embeddingModel: string;
  embeddingBatchSize: number;
  autoEmbedOnUpload: boolean;
  rerankCandidateCount: number;
  rerankBatchSize: number;
  rerankMaxSeqLength: number;
  rerankThreshold: number;
  enableRagByDefault: boolean;
}

export interface PublicRuntimeConfig {
  hasGeminiApiKey: boolean;
  hasCrossEncoderService: boolean;
  generationModel: string;
  embeddingModel: string;
  supportedEmbeddingModels: string[];
  maxUploadBytes: number;
  supportedFileExtensions: string[];
  rateLimitPerMinute: number;
  environment: string;
}

export interface AnalyticsSnapshot {
  corpus: {
    totalDocuments: number;
    demoDocuments: number;
    userDocuments: number;
    totalChunks: number;
    embeddedChunks: number;
    failedEmbeddingDocs: number;
    totalWords: number;
    byType: Array<{ type: string; count: number }>;
    bySource: Array<{ source: string; count: number }>;
  };
  vectorIndex: VectorIndexStats;
  queries: {
    totalQueries: number;
    failedQueries: number;
    avgLatencyMs: number;
    zeroResultCount: number;
    modeBreakdown: Array<{ mode: SearchMode; count: number; avgLatencyMs: number }>;
    categoryBreakdown: Array<{ category: string; count: number }>;
    avgStageLatencies: StageLatencyBreakdown;
    latencySeries: Array<{
      index: number;
      timestamp: string;
      mode: SearchMode;
      totalLatencyMs: number;
      topScore: number | null;
    }>;
    recentQueries: QueryLogEntry[];
  };
}
