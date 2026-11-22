import fs from 'fs';
import path from 'path';
import { serverConfig, SUPPORTED_EMBEDDING_MODELS } from './config.ts';
import { vectorIndex } from './modules/vectorIndex.ts';
import type {
  DocumentChunk,
  EngineSettings,
  EvaluationQuery,
  EvaluationRunRecord,
  QueryLogEntry,
  ResearchDocument,
  SearchPipelineTrace,
} from '../src/types/lucent.ts';

interface PersistedState {
  documents: ResearchDocument[];
  chunks: DocumentChunk[];
  queryLogs: QueryLogEntry[];
  evaluationQueries: EvaluationQuery[];
  evaluationRuns: EvaluationRunRecord[];
  settings: EngineSettings;
  lastTrace: SearchPipelineTrace | null;
}

export const DEFAULT_SETTINGS: EngineSettings = {
  defaultMode: 'hybrid',
  defaultRerankerMode: 'none',
  defaultAnswerStyle: 'detailed',
  topK: 6,
  bm25Candidates: 25,
  semanticCandidates: 25,
  hybridCandidates: 10,
  hybridFusionMethod: 'rrf',
  hybridAlpha: 0.6,
  bm25Weight: 1.0,
  semanticWeight: 1.0,
  rrfK: 60,
  bm25K1: 1.5,
  bm25B: 0.75,
  chunkSize: 180,
  chunkOverlap: 35,
  embeddingModel: serverConfig.embeddingModel,
  embeddingBatchSize: 16,
  autoEmbedOnUpload: true,
  rerankCandidateCount: 10,
  rerankBatchSize: 8,
  rerankMaxSeqLength: 512,
  rerankThreshold: 0.0,
  enableRagByDefault: true,
};

const DATA_DIR = serverConfig.dataDir;
const UPLOADS_DIR = path.resolve(DATA_DIR, 'uploads');
const STATE_FILE = path.resolve(DATA_DIR, 'lucent-state.json');

function ensureDirectories(): void {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

class LucentStore {
  private state: PersistedState;

  constructor() {
    ensureDirectories();
    this.state = this.loadFromDisk();
  }

  private loadFromDisk(): PersistedState {
    if (fs.existsSync(STATE_FILE)) {
      try {
        const raw = fs.readFileSync(STATE_FILE, 'utf8');
        const parsed = JSON.parse(raw) as Partial<PersistedState>;
        const evalQueries = Array.isArray(parsed.evaluationQueries)
          ? parsed.evaluationQueries.map((q) => {
              const grades: Record<string, number> = q.relevanceGrades ? { ...q.relevanceGrades } : {};
              for (const docId of q.relevantDocumentIds || []) {
                if (typeof grades[docId] !== 'number') {
                  grades[docId] = 2;
                }
              }
              return { ...q, relevanceGrades: grades };
            })
          : [];
        const mergedSettings: EngineSettings = {
          ...DEFAULT_SETTINGS,
          ...(parsed.settings || {}),
        };
        if (!SUPPORTED_EMBEDDING_MODELS.includes(mergedSettings.embeddingModel)) {
          mergedSettings.embeddingModel = serverConfig.embeddingModel;
        }
        return {
          documents: Array.isArray(parsed.documents) ? parsed.documents : [],
          chunks: Array.isArray(parsed.chunks) ? parsed.chunks : [],
          queryLogs: Array.isArray(parsed.queryLogs) ? parsed.queryLogs : [],
          evaluationQueries: evalQueries,
          evaluationRuns: Array.isArray(parsed.evaluationRuns) ? parsed.evaluationRuns : [],
          settings: mergedSettings,
          lastTrace: parsed.lastTrace || null,
        };
      } catch (err) {
        console.error('[LucentStore] Failed to parse state file, initializing fresh store:', err);
      }
    }
    return {
      documents: [],
      chunks: [],
      queryLogs: [],
      evaluationQueries: [],
      evaluationRuns: [],
      settings: { ...DEFAULT_SETTINGS },
      lastTrace: null,
    };
  }

  public save(): void {
    ensureDirectories();
    const tmpFile = `${STATE_FILE}.tmp`;
    fs.writeFileSync(tmpFile, JSON.stringify(this.state, null, 2), 'utf8');
    fs.renameSync(tmpFile, STATE_FILE);
  }

  public getUploadsDir(): string {
    ensureDirectories();
    return UPLOADS_DIR;
  }

  public resolveSafeUploadPath(storedRelativeName: string): string | null {
    const safeBase = path.basename(storedRelativeName);
    const resolved = path.resolve(UPLOADS_DIR, safeBase);
    if (!resolved.startsWith(UPLOADS_DIR + path.sep)) {
      return null;
    }
    return resolved;
  }

  public getDocuments(): ResearchDocument[] {
    return this.state.documents;
  }

  public getDocumentById(id: string): ResearchDocument | undefined {
    return this.state.documents.find((doc) => doc.id === id);
  }

  public upsertDocument(doc: ResearchDocument, docChunks: DocumentChunk[]): void {
    this.state.documents = this.state.documents.filter((d) => d.id !== doc.id);
    this.state.documents.unshift(doc);
    this.state.chunks = this.state.chunks.filter((c) => c.documentId !== doc.id);
    this.state.chunks.push(...docChunks);
    this.save();
  }

  public updateDocumentRecord(id: string, patch: Partial<ResearchDocument>): ResearchDocument | null {
    const doc = this.getDocumentById(id);
    if (!doc) return null;
    Object.assign(doc, patch, { updatedAt: new Date().toISOString() });
    this.save();
    return doc;
  }

  public deleteDocument(id: string): boolean {
    const doc = this.getDocumentById(id);
    if (!doc) return false;

    if (doc.storedFilePath) {
      const safePath = this.resolveSafeUploadPath(doc.storedFilePath);
      if (safePath && fs.existsSync(safePath)) {
        try {
          fs.unlinkSync(safePath);
        } catch {
          // Ignore file removal error if already missing
        }
      }
    }

    vectorIndex.removeByDocumentId(id);
    this.state.documents = this.state.documents.filter((d) => d.id !== id);
    this.state.chunks = this.state.chunks.filter((c) => c.documentId !== id);
    this.save();
    return true;
  }

  public deleteDemoDocuments(): { deletedDocuments: number; deletedEvaluationQueries: number } {
    const demoDocIds = new Set(this.state.documents.filter((d) => d.isDemo).map((d) => d.id));
    for (const id of demoDocIds) {
      vectorIndex.removeByDocumentId(id);
    }

    const beforeDocs = this.state.documents.length;
    this.state.documents = this.state.documents.filter((d) => !d.isDemo);
    this.state.chunks = this.state.chunks.filter((c) => !demoDocIds.has(c.documentId));

    const beforeEval = this.state.evaluationQueries.length;
    this.state.evaluationQueries = this.state.evaluationQueries.filter((q) => !q.isDemo);

    const deletedDocuments = beforeDocs - this.state.documents.length;
    const deletedEvaluationQueries = beforeEval - this.state.evaluationQueries.length;

    if (deletedDocuments > 0 || deletedEvaluationQueries > 0) {
      this.save();
    }

    return { deletedDocuments, deletedEvaluationQueries };
  }

  public getChunks(): DocumentChunk[] {
    return this.state.chunks;
  }

  public getChunksByDocumentId(documentId: string): DocumentChunk[] {
    return this.state.chunks
      .filter((c) => c.documentId === documentId)
      .sort((a, b) => a.chunkIndex - b.chunkIndex);
  }

  public replaceDocumentChunks(documentId: string, newChunks: DocumentChunk[], embeddingModel: string): void {
    this.state.chunks = this.state.chunks.filter((c) => c.documentId !== documentId);
    this.state.chunks.push(...newChunks);
    const activeIds = new Set(newChunks.map((c) => c.id));
    vectorIndex.removeStaleChunksForDocument(documentId, activeIds);

    const doc = this.getDocumentById(documentId);
    if (doc) {
      const embeddedCount = newChunks.filter((c) =>
        vectorIndex.hasMatchingVector(c.id, c.contentHash, embeddingModel)
      ).length;
      doc.chunkCount = newChunks.length;
      doc.embeddedChunkCount = embeddedCount;
      doc.embeddingStatus =
        newChunks.length === 0
          ? 'pending'
          : embeddedCount === newChunks.length
            ? 'completed'
            : embeddedCount > 0
              ? 'partial'
              : 'pending';
      doc.embeddingError = null;
      doc.embeddingModel = embeddedCount > 0 ? embeddingModel : null;
      doc.updatedAt = new Date().toISOString();
    }
    this.save();
  }

  public syncDocumentEmbeddingStatus(
    documentId: string,
    embeddingModel: string,
    explicitError?: string | null
  ): ResearchDocument | null {
    const doc = this.getDocumentById(documentId);
    if (!doc) return null;

    const docChunks = this.getChunksByDocumentId(documentId);
    const embeddedCount = docChunks.filter((c) =>
      vectorIndex.hasMatchingVector(c.id, c.contentHash, embeddingModel)
    ).length;

    doc.chunkCount = docChunks.length;
    doc.embeddedChunkCount = embeddedCount;

    if (explicitError) {
      doc.embeddingStatus = embeddedCount > 0 ? 'partial' : 'failed';
      doc.embeddingError = explicitError;
      doc.embeddingModel = embeddedCount > 0 ? embeddingModel : null;
    } else if (docChunks.length > 0 && embeddedCount === docChunks.length) {
      doc.embeddingStatus = 'completed';
      doc.embeddingError = null;
      doc.embeddingModel = embeddingModel;
    } else if (embeddedCount > 0) {
      doc.embeddingStatus = 'partial';
      doc.embeddingError = null;
      doc.embeddingModel = embeddingModel;
    } else {
      doc.embeddingStatus = 'pending';
      doc.embeddingError = null;
      doc.embeddingModel = null;
    }

    doc.updatedAt = new Date().toISOString();
    this.save();
    return doc;
  }

  public getSettings(): EngineSettings {
    return { ...this.state.settings };
  }

  public updateSettings(partial: Partial<EngineSettings>): EngineSettings {
    this.state.settings = { ...this.state.settings, ...partial };
    this.save();
    return { ...this.state.settings };
  }

  public addQueryLog(entry: QueryLogEntry, trace: SearchPipelineTrace | null): void {
    this.state.queryLogs.unshift(entry);
    if (this.state.queryLogs.length > 300) {
      this.state.queryLogs = this.state.queryLogs.slice(0, 300);
    }
    if (trace) {
      this.state.lastTrace = trace;
    }
    this.save();
  }

  public getQueryLogs(): QueryLogEntry[] {
    return this.state.queryLogs;
  }

  public clearQueryLogs(): void {
    this.state.queryLogs = [];
    this.state.lastTrace = null;
    this.save();
  }

  public getLastTrace(): SearchPipelineTrace | null {
    return this.state.lastTrace;
  }

  public getEvaluationQueries(): EvaluationQuery[] {
    return this.state.evaluationQueries;
  }

  public setEvaluationQueries(queries: EvaluationQuery[]): void {
    this.state.evaluationQueries = queries;
    this.save();
  }

  public addEvaluationQuery(query: EvaluationQuery): void {
    this.state.evaluationQueries.unshift(query);
    this.save();
  }

  public updateEvaluationQuery(id: string, patch: Partial<EvaluationQuery>): EvaluationQuery | null {
    const existing = this.state.evaluationQueries.find((q) => q.id === id);
    if (!existing) return null;
    Object.assign(existing, patch);
    this.save();
    return existing;
  }

  public deleteEvaluationQuery(id: string): boolean {
    const before = this.state.evaluationQueries.length;
    this.state.evaluationQueries = this.state.evaluationQueries.filter((q) => q.id !== id);
    if (this.state.evaluationQueries.length !== before) {
      this.save();
      return true;
    }
    return false;
  }

  public getEvaluationRuns(): EvaluationRunRecord[] {
    return this.state.evaluationRuns;
  }

  public getEvaluationRunById(id: string): EvaluationRunRecord | undefined {
    return this.state.evaluationRuns.find((r) => r.id === id);
  }

  public addEvaluationRun(run: EvaluationRunRecord): void {
    this.state.evaluationRuns.unshift(run);
    if (this.state.evaluationRuns.length > 40) {
      this.state.evaluationRuns = this.state.evaluationRuns.slice(0, 40);
    }
    this.save();
  }
}

export const lucentStore = new LucentStore();
