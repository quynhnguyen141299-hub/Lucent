import fs from 'fs';
import path from 'path';
import { serverConfig } from '../config.ts';
import type { VectorIndexStats } from '../../src/types/lucent.ts';

export interface VectorRecordMeta {
  chunkId: string;
  documentId: string;
  contentHash: string;
  model: string;
  dimensions: number;
  offset: number; // start index in Float32Array buffer
  updatedAt: string;
}

export interface VectorSearchHit {
  chunkId: string;
  documentId: string;
  score: number;
}

export interface VectorIndexBackend {
  upsertBatch(
    records: Array<{
      chunkId: string;
      documentId: string;
      contentHash: string;
      model: string;
      vector: number[];
    }>
  ): void;
  removeByDocumentId(documentId: string): number;
  hasMatchingVector(chunkId: string, contentHash: string, model: string): boolean;
  getVectorMeta(chunkId: string): VectorRecordMeta | undefined;
  search(
    queryVector: number[],
    options: {
      model: string;
      allowedChunkIds?: Set<string>;
      topK?: number;
    }
  ): VectorSearchHit[];
  getStats(): VectorIndexStats;
}

function l2Normalize(vec: number[] | Float32Array): Float32Array {
  const out = new Float32Array(vec.length);
  let normSq = 0;
  for (let i = 0; i < vec.length; i++) {
    const v = vec[i];
    normSq += v * v;
  }
  const norm = Math.sqrt(normSq);
  if (norm === 0) return out;
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] / norm;
  }
  return out;
}

interface PersistedVectorMetaFile {
  version: number;
  backend: string;
  metric: string;
  records: VectorRecordMeta[];
}

/**
 * Local persistent FAISS IndexFlatIP-compatible vector store.
 * Stores L2-normalized Float32 vectors in a contiguous binary file (`vector-index.f32`)
 * paired with a compact metadata index (`vector-index.meta.json`).
 * Exact inner product over L2-normalized vectors equals cosine similarity.
 */
class FlatNormalizedVectorIndex implements VectorIndexBackend {
  private readonly metaPath: string;
  private readonly binPath: string;
  private records = new Map<string, VectorRecordMeta>();
  private vectors = new Map<string, Float32Array>();

  constructor() {
    fs.mkdirSync(serverConfig.dataDir, { recursive: true });
    this.metaPath = path.resolve(serverConfig.dataDir, 'vector-index.meta.json');
    this.binPath = path.resolve(serverConfig.dataDir, 'vector-index.f32');
    this.loadFromDisk();
  }

  private loadFromDisk(): void {
    if (!fs.existsSync(this.metaPath) || !fs.existsSync(this.binPath)) {
      return;
    }
    try {
      const metaRaw = fs.readFileSync(this.metaPath, 'utf8');
      const parsed = JSON.parse(metaRaw) as PersistedVectorMetaFile;
      const binBuffer = fs.readFileSync(this.binPath);
      const flat = new Float32Array(
        binBuffer.buffer,
        binBuffer.byteOffset,
        Math.floor(binBuffer.byteLength / 4)
      );

      for (const rec of parsed.records || []) {
        if (rec.offset + rec.dimensions <= flat.length && rec.dimensions > 0) {
          const slice = new Float32Array(rec.dimensions);
          slice.set(flat.subarray(rec.offset, rec.offset + rec.dimensions));
          this.records.set(rec.chunkId, rec);
          this.vectors.set(rec.chunkId, slice);
        }
      }
    } catch (err) {
      console.error('[VectorIndex] Failed to load persisted vector index, starting empty:', err);
      this.records.clear();
      this.vectors.clear();
    }
  }

  private persist(): void {
    fs.mkdirSync(serverConfig.dataDir, { recursive: true });
    const recordList = Array.from(this.records.values());
    const totalFloats = recordList.reduce((sum, r) => sum + r.dimensions, 0);
    const flat = new Float32Array(totalFloats);
    const serializedRecords: VectorRecordMeta[] = [];

    let offset = 0;
    for (const rec of recordList) {
      const vec = this.vectors.get(rec.chunkId);
      if (!vec || vec.length !== rec.dimensions) continue;
      flat.set(vec, offset);
      serializedRecords.push({
        ...rec,
        offset,
      });
      offset += rec.dimensions;
    }

    const metaPayload: PersistedVectorMetaFile = {
      version: 1,
      backend: 'Local FlatIP Float32 Index (FAISS-compatible)',
      metric: 'Cosine Similarity (L2-Normalized Inner Product)',
      records: serializedRecords,
    };

    const tmpBin = `${this.binPath}.tmp`;
    const tmpMeta = `${this.metaPath}.tmp`;
    fs.writeFileSync(tmpBin, Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength));
    fs.writeFileSync(tmpMeta, JSON.stringify(metaPayload, null, 2), 'utf8');
    fs.renameSync(tmpBin, this.binPath);
    fs.renameSync(tmpMeta, this.metaPath);
  }

  public upsertBatch(
    entries: Array<{
      chunkId: string;
      documentId: string;
      contentHash: string;
      model: string;
      vector: number[];
    }>
  ): void {
    if (entries.length === 0) return;
    const now = new Date().toISOString();

    for (const entry of entries) {
      if (!entry.vector || entry.vector.length === 0) continue;
      const normalized = l2Normalize(entry.vector);
      this.vectors.set(entry.chunkId, normalized);
      this.records.set(entry.chunkId, {
        chunkId: entry.chunkId,
        documentId: entry.documentId,
        contentHash: entry.contentHash,
        model: entry.model,
        dimensions: normalized.length,
        offset: 0,
        updatedAt: now,
      });
    }

    this.persist();
  }

  public removeByDocumentId(documentId: string): number {
    let removed = 0;
    for (const [chunkId, rec] of this.records.entries()) {
      if (rec.documentId === documentId) {
        this.records.delete(chunkId);
        this.vectors.delete(chunkId);
        removed++;
      }
    }
    if (removed > 0) {
      this.persist();
    }
    return removed;
  }

  public removeStaleChunksForDocument(documentId: string, activeChunkIds: Set<string>): number {
    let removed = 0;
    for (const [chunkId, rec] of this.records.entries()) {
      if (rec.documentId === documentId && !activeChunkIds.has(chunkId)) {
        this.records.delete(chunkId);
        this.vectors.delete(chunkId);
        removed++;
      }
    }
    if (removed > 0) {
      this.persist();
    }
    return removed;
  }

  public hasMatchingVector(chunkId: string, contentHash: string, model: string): boolean {
    const rec = this.records.get(chunkId);
    const vec = this.vectors.get(chunkId);
    return Boolean(rec && vec && rec.contentHash === contentHash && rec.model === model);
  }

  public getVectorMeta(chunkId: string): VectorRecordMeta | undefined {
    return this.records.get(chunkId);
  }

  public search(
    queryVector: number[],
    options: {
      model: string;
      allowedChunkIds?: Set<string>;
      topK?: number;
    }
  ): VectorSearchHit[] {
    if (!queryVector || queryVector.length === 0) return [];
    const qNorm = l2Normalize(queryVector);
    const qDim = qNorm.length;
    const hits: VectorSearchHit[] = [];

    for (const [chunkId, rec] of this.records.entries()) {
      if (rec.model !== options.model) continue;
      if (rec.dimensions !== qDim) continue;
      if (options.allowedChunkIds && !options.allowedChunkIds.has(chunkId)) continue;

      const vec = this.vectors.get(chunkId);
      if (!vec) continue;

      // Exact inner product over L2-normalized Float32 vectors
      let dot = 0;
      for (let i = 0; i < qDim; i++) {
        dot += qNorm[i] * vec[i];
      }

      hits.push({
        chunkId,
        documentId: rec.documentId,
        score: dot,
      });
    }

    hits.sort((a, b) => b.score - a.score);
    if (options.topK && options.topK > 0) {
      return hits.slice(0, options.topK);
    }
    return hits;
  }

  public getStats(): VectorIndexStats {
    const modelCounts = new Map<string, number>();
    let firstDim: number | null = null;
    let totalFloats = 0;

    for (const rec of this.records.values()) {
      if (firstDim === null) firstDim = rec.dimensions;
      totalFloats += rec.dimensions;
      modelCounts.set(rec.model, (modelCounts.get(rec.model) || 0) + 1);
    }

    return {
      backend: 'Local FlatIP Float32 Index (FAISS-compatible)',
      metric: 'Cosine Similarity (L2-Normalized Inner Product)',
      vectorCount: this.records.size,
      dimensions: firstDim,
      storageBytes: totalFloats * 4,
      models: Array.from(modelCounts.entries()).map(([model, count]) => ({ model, count })),
      persistedPath: path.basename(this.binPath),
    };
  }
}

export const vectorIndex = new FlatNormalizedVectorIndex();
