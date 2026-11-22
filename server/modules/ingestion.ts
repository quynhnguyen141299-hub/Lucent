import crypto from 'crypto';
import fs from 'fs';
import mammoth from 'mammoth';
import path from 'path';
import { extractText as extractPdfText, getMeta as getPdfMeta } from 'unpdf';
import { serverConfig } from '../config.ts';
import { getGeminiClient, normalizeGeminiError, withRetry } from '../geminiClient.ts';
import { DEMO_EVALUATION_QUERIES, DEMO_SAMPLE_DOCUMENTS } from '../sampleCorpus.ts';
import { lucentStore } from '../store.ts';
import { vectorIndex } from './vectorIndex.ts';
import type { DocumentChunk, DocumentType, ResearchDocument } from '../../src/types/lucent.ts';

const ALLOWED_DOCUMENT_TYPES: Set<DocumentType> = new Set([
  'Research Paper',
  'Technical Report',
  'Policy Brief',
  'Clinical Study',
  'Dataset Notes',
  'Uploaded Document',
]);

export const SUPPORTED_UPLOAD_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md'];
const ALLOWED_UPLOAD_EXTENSIONS = new Set(SUPPORTED_UPLOAD_EXTENSIONS);

export function sha256(text: string): string {
  return crypto.createHash('sha256').update(text, 'utf8').digest('hex');
}

export function normalizeDocumentType(input?: string): DocumentType {
  if (input && ALLOWED_DOCUMENT_TYPES.has(input as DocumentType)) {
    return input as DocumentType;
  }
  return 'Uploaded Document';
}

export function sanitizeHttpUrl(raw?: string | null): string | null {
  if (!raw || !raw.trim()) return null;
  try {
    const parsed = new URL(raw.trim());
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed.toString();
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Cleans extracted document text:
 * - Removes null bytes, zero-width artifacts, and non-printable control characters
 * - Strips HTML tags and decorative horizontal rules / redundant markdown formatting
 * - Repairs soft hyphenation across line breaks (common in PDFs)
 * - Normalizes whitespace while preserving paragraph structure
 */
export function cleanDocumentText(raw: string): string {
  return raw
    .replace(/\u0000/g, '')
    .replace(/[\u200B-\u200D\uFEFF]/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    // Rejoin hyphenated line wraps from PDF columns
    .replace(/([a-zA-Z])-\n([a-zA-Z])/g, '$1$2')
    // Remove HTML tags if pasted or embedded
    .replace(/<\/?[a-z][^>]*>/gi, ' ')
    // Strip markdown heading prefixes, emphasis stars, and horizontal rules
    .replace(/^\s*[-*_]{3,}\s*$/gm, '')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/`{1,3}([^`]+)`{1,3}/g, '$1')
    // Strip control characters except tab and newline
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, ' ')
    // Normalize horizontal whitespace
    .replace(/[ \t\f\v]+/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n[ \t]+/g, '\n')
    // Collapse 3+ consecutive newlines to paragraph break
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export function countWords(text: string): number {
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  return tokens.length;
}

export function buildDocumentSummary(content: string): string {
  const cleaned = content.replace(/\s+/g, ' ').trim();
  if (cleaned.length <= 240) return cleaned;
  const cut = cleaned.slice(0, 240);
  const lastPeriod = cut.lastIndexOf('.');
  if (lastPeriod > 110) {
    return cut.slice(0, lastPeriod + 1);
  }
  const lastSpace = cut.lastIndexOf(' ');
  return `${cut.slice(0, lastSpace > 0 ? lastSpace : 240)}...`;
}

export interface ExtractedPageSegment {
  pageNumber: number | null;
  text: string;
}

export interface ExtractedFilePayload {
  pages: ExtractedPageSegment[];
  fullText: string;
  pageCount: number | null;
  detectedTitle: string | null;
  detectedAuthors: string[];
}

function validateFileSignature(ext: string, buffer: Buffer): void {
  if (buffer.length === 0) {
    throw new Error('The uploaded file is empty (0 bytes).');
  }
  if (buffer.length > serverConfig.maxUploadBytes) {
    throw new Error(
      `File exceeds the maximum allowed size of ${Math.round(serverConfig.maxUploadBytes / (1024 * 1024))} MB.`
    );
  }

  if (ext === '.pdf') {
    const header = buffer.subarray(0, 5).toString('ascii');
    if (!header.startsWith('%PDF-')) {
      throw new Error('File has a .pdf extension but does not contain a valid PDF header.');
    }
  } else if (ext === '.docx') {
    // DOCX files are ZIP archives starting with PK\x03\x04
    if (buffer.length < 4 || buffer[0] !== 0x50 || buffer[1] !== 0x4b) {
      throw new Error('File has a .docx extension but does not appear to be a valid Word OpenXML archive.');
    }
  }
}

/**
 * Extracts text and page boundaries from PDF, DOCX, TXT, or Markdown uploads.
 * Preserves PDF page numbers and never invents missing metadata.
 */
export async function extractDocumentFromBuffer(
  originalName: string,
  buffer: Buffer
): Promise<ExtractedFilePayload> {
  const ext = path.extname(originalName).toLowerCase();
  if (!ALLOWED_UPLOAD_EXTENSIONS.has(ext)) {
    throw new Error(
      `Unsupported file type "${ext || 'unknown'}". Supported formats: PDF (.pdf), Word (.docx), Plain Text (.txt), and Markdown (.md).`
    );
  }

  validateFileSignature(ext, buffer);

  if (ext === '.pdf') {
    try {
      const uint8 = new Uint8Array(buffer);
      const { totalPages, text: rawPages } = await extractPdfText(uint8, { mergePages: false });
      const pages: ExtractedPageSegment[] = [];

      for (let i = 0; i < rawPages.length; i++) {
        const cleanedPage = cleanDocumentText(rawPages[i] || '');
        if (cleanedPage.length > 0) {
          pages.push({ pageNumber: i + 1, text: cleanedPage });
        }
      }

      if (pages.length === 0) {
        throw new Error(
          'No extractable text was found in this PDF. It may be a scanned image-only PDF without an embedded text layer.'
        );
      }

      let detectedTitle: string | null = null;
      let detectedAuthors: string[] = [];
      try {
        const meta = await getPdfMeta(new Uint8Array(buffer));
        if (meta.info?.Title && meta.info.Title.trim().length > 2) {
          detectedTitle = meta.info.Title.trim().slice(0, 240);
        }
        if (meta.info?.Author && meta.info.Author.trim().length > 1) {
          detectedAuthors = meta.info.Author.split(/[;,]/).map((a) => a.trim()).filter(Boolean).slice(0, 10);
        }
      } catch {
        // Embedded PDF metadata is optional
      }

      const fullText = pages.map((p) => p.text).join('\n\n');
      return {
        pages,
        fullText,
        pageCount: totalPages || pages.length,
        detectedTitle,
        detectedAuthors,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown PDF extraction failure.';
      if (msg.includes('No extractable text')) {
        throw err;
      }
      throw new Error(`PDF text extraction failed: ${msg}`);
    }
  }

  if (ext === '.docx') {
    try {
      const result = await mammoth.extractRawText({ buffer });
      const cleaned = cleanDocumentText(result.value || '');
      if (cleaned.length < 10) {
        throw new Error('The DOCX document contains no readable text content.');
      }
      return {
        pages: [{ pageNumber: null, text: cleaned }],
        fullText: cleaned,
        pageCount: null,
        detectedTitle: null,
        detectedAuthors: [],
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown DOCX extraction failure.';
      throw new Error(`DOCX text extraction failed: ${msg}`);
    }
  }

  // Plain text / Markdown
  const rawText = buffer.toString('utf8');
  const cleaned = cleanDocumentText(rawText);
  if (cleaned.length < 10) {
    throw new Error('The uploaded text file contains insufficient readable text (minimum 10 characters).');
  }

  return {
    pages: [{ pageNumber: null, text: cleaned }],
    fullText: cleaned,
    pageCount: null,
    detectedTitle: null,
    detectedAuthors: [],
  };
}

/**
 * Segments document pages into sentence-aware overlapping chunks while preserving
 * document title, source, and page number metadata on every chunk.
 */
export function segmentDocumentIntoChunks(params: {
  documentId: string;
  documentTitle: string;
  source: string | null;
  pages: ExtractedPageSegment[];
  chunkSizeWords: number;
  overlapWords: number;
}): DocumentChunk[] {
  const { documentId, documentTitle, source, pages, chunkSizeWords, overlapWords } = params;
  const safeChunkSize = Math.max(40, Math.min(800, Math.floor(chunkSizeWords)));
  const safeOverlap = Math.max(0, Math.min(Math.floor(safeChunkSize * 0.5), Math.floor(overlapWords)));

  const chunks: DocumentChunk[] = [];
  let globalChunkIndex = 0;
  let runningOffset = 0;

  for (const page of pages) {
    const pageText = cleanDocumentText(page.text);
    if (!pageText) continue;

    const sentences = pageText
      .split(/(?<=[.!?])\s+|\n{2,}/)
      .map((s) => s.trim())
      .filter(Boolean);

    if (sentences.length === 0) continue;

    let cursorSentence = 0;
    while (cursorSentence < sentences.length) {
      const selected: string[] = [];
      let currentWords = 0;
      let i = cursorSentence;

      while (i < sentences.length) {
        const sentenceWords = countWords(sentences[i]);
        if (selected.length > 0 && currentWords + sentenceWords > safeChunkSize) {
          break;
        }
        selected.push(sentences[i]);
        currentWords += sentenceWords;
        i++;
      }

      const chunkText = selected.join(' ').trim();
      const localStart = Math.max(0, pageText.indexOf(selected[0]));
      const startOffset = runningOffset + localStart;
      const endOffset = startOffset + chunkText.length;

      chunks.push({
        id: `${documentId}::chunk-${globalChunkIndex}`,
        documentId,
        documentTitle,
        source,
        chunkIndex: globalChunkIndex,
        pageNumber: page.pageNumber,
        text: chunkText,
        startOffset,
        endOffset,
        wordCount: countWords(chunkText),
        contentHash: sha256(chunkText),
      });

      globalChunkIndex++;

      if (i >= sentences.length) {
        break;
      }

      let overlapCount = 0;
      let nextCursor = i;
      while (nextCursor > cursorSentence + 1 && overlapCount < safeOverlap) {
        nextCursor--;
        overlapCount += countWords(sentences[nextCursor]);
      }
      cursorSentence = Math.max(cursorSentence + 1, nextCursor);
    }

    runningOffset += pageText.length + 2;
  }

  return chunks;
}

export interface IngestDocumentInput {
  id?: string;
  title: string;
  source?: string | null;
  documentType?: string;
  publishedDate?: string | null;
  authors?: string[];
  sourceUrl?: string | null;
  content: string;
  pages?: ExtractedPageSegment[];
  pageCount?: number | null;
  isDemo?: boolean;
  uploadedFile?: {
    originalName: string;
    mimeType: string;
    buffer: Buffer;
  };
}

export function ingestDocumentRecord(input: IngestDocumentInput): {
  document: ResearchDocument;
  chunks: DocumentChunk[];
  unchanged: boolean;
} {
  const settings = lucentStore.getSettings();
  const id = input.id || `doc-${crypto.randomUUID()}`;
  const cleanedContent = cleanDocumentText(input.content);

  if (!input.title.trim()) {
    throw new Error('Document title is required.');
  }
  if (!cleanedContent || cleanedContent.length < 20) {
    throw new Error('Document content must contain at least 20 characters of text.');
  }

  const normalizedTitle = input.title.trim().slice(0, 240);
  // Never invent a source or publication date if none was provided
  const normalizedSource = input.source && input.source.trim() ? input.source.trim().slice(0, 160) : null;
  const normalizedDate =
    input.publishedDate && /^\d{4}-\d{2}-\d{2}$/.test(input.publishedDate.trim())
      ? input.publishedDate.trim()
      : null;
  const contentHash = sha256(cleanedContent);

  const existingDoc = lucentStore.getDocumentById(id);
  const unchanged = Boolean(
    existingDoc &&
      existingDoc.contentHash === contentHash &&
      existingDoc.chunkCount > 0
  );

  let storedFilePath: string | null = existingDoc?.storedFilePath || null;
  let originalFilename: string | null = existingDoc?.originalFilename || null;
  let mimeType = existingDoc?.mimeType || 'text/plain; charset=utf-8';
  let fileSizeBytes = Buffer.byteLength(cleanedContent, 'utf8');

  if (input.uploadedFile) {
    const ext = path.extname(input.uploadedFile.originalName).toLowerCase();
    if (!ALLOWED_UPLOAD_EXTENSIONS.has(ext)) {
      throw new Error(`Unsupported file extension "${ext}".`);
    }
    const generatedName = `${id}${ext}`;
    const safeTarget = lucentStore.resolveSafeUploadPath(generatedName);
    if (!safeTarget) {
      throw new Error('Invalid file storage path.');
    }
    fs.writeFileSync(safeTarget, input.uploadedFile.buffer);
    storedFilePath = generatedName;
    originalFilename = path.basename(input.uploadedFile.originalName).slice(0, 180);
    mimeType = input.uploadedFile.mimeType || 'application/octet-stream';
    fileSizeBytes = input.uploadedFile.buffer.length;
  }

  const pages: ExtractedPageSegment[] =
    input.pages && input.pages.length > 0
      ? input.pages
      : [{ pageNumber: null, text: cleanedContent }];

  const chunks = segmentDocumentIntoChunks({
    documentId: id,
    documentTitle: normalizedTitle,
    source: normalizedSource,
    pages,
    chunkSizeWords: settings.chunkSize,
    overlapWords: settings.chunkOverlap,
  });

  const activeEmbeddingModel = settings.embeddingModel || serverConfig.embeddingModel;
  const alreadyEmbeddedCount = chunks.filter((c) =>
    vectorIndex.hasMatchingVector(c.id, c.contentHash, activeEmbeddingModel)
  ).length;

  const now = new Date().toISOString();

  const doc: ResearchDocument = {
    id,
    title: normalizedTitle,
    source: normalizedSource,
    documentType: normalizeDocumentType(input.documentType),
    publishedDate: normalizedDate,
    authors: (input.authors || [])
      .map((a) => a.trim())
      .filter(Boolean)
      .slice(0, 12),
    sourceUrl: sanitizeHttpUrl(input.sourceUrl),
    originalFilename,
    storedFilePath,
    mimeType,
    fileSizeBytes,
    pageCount: input.pageCount ?? null,
    contentHash,
    content: cleanedContent,
    summary: buildDocumentSummary(cleanedContent),
    wordCount: countWords(cleanedContent),
    chunkCount: chunks.length,
    embeddedChunkCount: alreadyEmbeddedCount,
    processingStatus: 'indexed',
    processingError: null,
    embeddingStatus:
      chunks.length > 0 && alreadyEmbeddedCount === chunks.length
        ? 'completed'
        : alreadyEmbeddedCount > 0
          ? 'partial'
          : 'pending',
    embeddingError: null,
    embeddingModel: alreadyEmbeddedCount > 0 ? activeEmbeddingModel : null,
    isDemo: Boolean(input.isDemo),
    uploadedAt: existingDoc?.uploadedAt || now,
    createdAt: existingDoc?.createdAt || now,
    updatedAt: now,
  };

  lucentStore.upsertDocument(doc, chunks);
  vectorIndex.removeStaleChunksForDocument(id, new Set(chunks.map((c) => c.id)));

  return { document: doc, chunks, unchanged };
}

/**
 * Generates dense embeddings for a batch of text strings via server-side Google Embedding API.
 * Uses configurable embedding model and batch size with exponential backoff retries.
 */
export async function embedTexts(
  texts: string[],
  overrideModel?: string,
  overrideBatchSize?: number
): Promise<{ vectors: number[][]; model: string }> {
  const settings = lucentStore.getSettings();
  const model = (overrideModel || settings.embeddingModel || serverConfig.embeddingModel).trim();
  const batchSize = Math.max(1, Math.min(32, Math.floor(overrideBatchSize || settings.embeddingBatchSize || 16)));

  if (texts.length === 0) return { vectors: [], model };
  const ai = getGeminiClient();
  const vectors: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const response = await withRetry(() =>
      ai.models.embedContent({
        model,
        contents: batch,
      })
    );

    const embeddings = response.embeddings || [];
    if (embeddings.length !== batch.length) {
      throw new Error(
        `Embedding response mismatch: expected ${batch.length} vectors, received ${embeddings.length}.`
      );
    }

    for (const emb of embeddings) {
      if (!emb.values || emb.values.length === 0) {
        throw new Error('Received an empty embedding vector from the embedding API.');
      }
      vectors.push(emb.values);
    }
  }

  return { vectors, model };
}

/**
 * Generates and stores vectors in the persistent Vector Index for chunks that lack
 * a matching (chunkId, contentHash, model) entry. Skips unchanged chunks automatically.
 */
export async function ensureChunksEmbedded(
  targetChunks?: DocumentChunk[],
  options?: { force?: boolean; model?: string }
): Promise<{
  embeddedNow: number;
  skippedUnchanged: number;
  totalTarget: number;
  model: string;
}> {
  const settings = lucentStore.getSettings();
  const model = (options?.model || settings.embeddingModel || serverConfig.embeddingModel).trim();
  const chunks = targetChunks || lucentStore.getChunks();

  const toEmbed = options?.force
    ? chunks
    : chunks.filter((c) => !vectorIndex.hasMatchingVector(c.id, c.contentHash, model));

  const skippedUnchanged = chunks.length - toEmbed.length;

  if (toEmbed.length === 0) {
    const touchedDocIds = new Set(chunks.map((c) => c.documentId));
    for (const docId of touchedDocIds) {
      lucentStore.syncDocumentEmbeddingStatus(docId, model, null);
    }
    return { embeddedNow: 0, skippedUnchanged, totalTarget: chunks.length, model };
  }

  const touchedDocIds = new Set(toEmbed.map((c) => c.documentId));
  for (const docId of touchedDocIds) {
    lucentStore.updateDocumentRecord(docId, {
      embeddingStatus: 'embedding',
      embeddingError: null,
    });
  }

  try {
    const batchSize = Math.max(1, Math.min(32, settings.embeddingBatchSize || 16));
    let embeddedNow = 0;

    for (let i = 0; i < toEmbed.length; i += batchSize) {
      const slice = toEmbed.slice(i, i + batchSize);
      const { vectors } = await embedTexts(
        slice.map((c) => c.text),
        model,
        batchSize
      );

      vectorIndex.upsertBatch(
        slice.map((chunk, idx) => ({
          chunkId: chunk.id,
          documentId: chunk.documentId,
          contentHash: chunk.contentHash,
          model,
          vector: vectors[idx],
        }))
      );
      embeddedNow += slice.length;
    }

    for (const docId of touchedDocIds) {
      lucentStore.syncDocumentEmbeddingStatus(docId, model, null);
    }

    return {
      embeddedNow,
      skippedUnchanged,
      totalTarget: chunks.length,
      model,
    };
  } catch (err) {
    const normalized = normalizeGeminiError(err);
    for (const docId of touchedDocIds) {
      lucentStore.syncDocumentEmbeddingStatus(docId, model, normalized.message);
    }
    throw normalized;
  }
}

export async function embedSingleDocument(
  documentId: string,
  options?: { force?: boolean }
): Promise<{
  document: ResearchDocument;
  embeddedNow: number;
  skippedUnchanged: number;
  model: string;
}> {
  const doc = lucentStore.getDocumentById(documentId);
  if (!doc) {
    throw new Error('Document not found.');
  }
  const chunks = lucentStore.getChunksByDocumentId(documentId);
  const result = await ensureChunksEmbedded(chunks, { force: options?.force });
  const updated = lucentStore.getDocumentById(documentId) || doc;
  return {
    document: updated,
    embeddedNow: result.embeddedNow,
    skippedUnchanged: result.skippedUnchanged,
    model: result.model,
  };
}

/**
 * Reprocesses an existing document: re-extracts original uploaded file if available
 * (or uses stored cleaned text), re-segments chunks with current settings, and refreshes embeddings.
 */
export async function reprocessDocument(
  documentId: string,
  options?: { forceReembed?: boolean }
): Promise<{
  document: ResearchDocument;
  embeddedNow: number;
  skippedUnchanged: number;
  embeddingWarning: string | null;
}> {
  const doc = lucentStore.getDocumentById(documentId);
  if (!doc) {
    throw new Error('Document not found.');
  }

  lucentStore.updateDocumentRecord(documentId, {
    processingStatus: 'processing',
    processingError: null,
  });

  const settings = lucentStore.getSettings();
  const activeModel = settings.embeddingModel || serverConfig.embeddingModel;

  try {
    let pages: ExtractedPageSegment[] = [{ pageNumber: null, text: doc.content }];
    let fullText = doc.content;
    let pageCount = doc.pageCount;

    if (doc.storedFilePath && doc.originalFilename) {
      const safePath = lucentStore.resolveSafeUploadPath(doc.storedFilePath);
      if (safePath && fs.existsSync(safePath)) {
        const buffer = fs.readFileSync(safePath);
        const extracted = await extractDocumentFromBuffer(doc.originalFilename, buffer);
        pages = extracted.pages;
        fullText = extracted.fullText;
        pageCount = extracted.pageCount;
      }
    }

    const newChunks = segmentDocumentIntoChunks({
      documentId: doc.id,
      documentTitle: doc.title,
      source: doc.source,
      pages,
      chunkSizeWords: settings.chunkSize,
      overlapWords: settings.chunkOverlap,
    });

    lucentStore.updateDocumentRecord(doc.id, {
      content: fullText,
      contentHash: sha256(fullText),
      summary: buildDocumentSummary(fullText),
      wordCount: countWords(fullText),
      pageCount,
      processingStatus: 'indexed',
      processingError: null,
    });

    lucentStore.replaceDocumentChunks(doc.id, newChunks, activeModel);

    let embeddedNow = 0;
    let skippedUnchanged = newChunks.length;
    let embeddingWarning: string | null = null;

    if (settings.autoEmbedOnUpload || options?.forceReembed) {
      try {
        const embedRes = await ensureChunksEmbedded(newChunks, {
          force: Boolean(options?.forceReembed),
          model: activeModel,
        });
        embeddedNow = embedRes.embeddedNow;
        skippedUnchanged = embedRes.skippedUnchanged;
      } catch (err) {
        const normalized = normalizeGeminiError(err);
        embeddingWarning = normalized.message;
      }
    }

    const updatedDoc = lucentStore.getDocumentById(doc.id) || doc;
    return {
      document: updatedDoc,
      embeddedNow,
      skippedUnchanged,
      embeddingWarning,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Document reprocessing failed.';
    lucentStore.updateDocumentRecord(documentId, {
      processingStatus: 'error',
      processingError: message,
    });
    throw err;
  }
}

export function rechunkCorpus(chunkSize: number, chunkOverlap: number): { documentsRechunked: number; totalChunks: number } {
  const docs = lucentStore.getDocuments();
  const settings = lucentStore.getSettings();
  const activeModel = settings.embeddingModel || serverConfig.embeddingModel;
  let totalChunks = 0;

  for (const doc of docs) {
    const chunks = segmentDocumentIntoChunks({
      documentId: doc.id,
      documentTitle: doc.title,
      source: doc.source,
      pages: [{ pageNumber: null, text: doc.content }],
      chunkSizeWords: chunkSize,
      overlapWords: chunkOverlap,
    });
    lucentStore.replaceDocumentChunks(doc.id, chunks, activeModel);
    totalChunks += chunks.length;
  }

  return { documentsRechunked: docs.length, totalChunks };
}

export function seedDemoCorpusIfEmpty(): void {
  if (lucentStore.getDocuments().length === 0) {
    restoreDemoCorpus();
  }
  if (lucentStore.getEvaluationQueries().length === 0) {
    lucentStore.setEvaluationQueries([...DEMO_EVALUATION_QUERIES]);
  }
}

export function restoreDemoCorpus(): { restoredCount: number } {
  let restoredCount = 0;
  for (const sample of DEMO_SAMPLE_DOCUMENTS) {
    if (!lucentStore.getDocumentById(sample.id)) {
      ingestDocumentRecord({
        id: sample.id,
        title: sample.title,
        source: sample.source,
        documentType: sample.documentType,
        publishedDate: sample.publishedDate,
        authors: sample.authors,
        sourceUrl: sample.sourceUrl,
        content: sample.content,
        isDemo: true,
      });
      restoredCount++;
    }
  }

  const existingEvalIds = new Set(lucentStore.getEvaluationQueries().map((q) => q.id));
  for (const demoQuery of DEMO_EVALUATION_QUERIES) {
    if (!existingEvalIds.has(demoQuery.id)) {
      lucentStore.addEvaluationQuery(demoQuery);
    }
  }

  return { restoredCount };
}
