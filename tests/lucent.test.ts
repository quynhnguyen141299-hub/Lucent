import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  computeAveragePrecision,
  computeGradedNdcgAtK,
  computeMrr,
  computePrecisionAtK,
  computeRecallAtK,
} from '../server/modules/evaluation.ts';
import { validateAndFilterCitations } from '../server/modules/generation.ts';
import {
  cleanDocumentText,
  sanitizeHttpUrl,
  segmentDocumentIntoChunks,
} from '../server/modules/ingestion.ts';
import { computeRrfScore, rankCandidates } from '../server/modules/ranking.ts';
import {
  applyValidatedRerankScores,
  CrossEncoderHttpProvider,
  RerankerServiceError,
} from '../server/modules/reranking.ts';
import { computeCosineSimilarity, runBm25Retrieval } from '../server/modules/retrieval.ts';
import { sanitizeShortString } from '../server/middleware/reliability.ts';
import type { DocumentChunk, ResearchDocument, SearchResultItem } from '../src/types/lucent.ts';

function makeDoc(id: string, title: string, content: string): ResearchDocument {
  return {
    id,
    title,
    source: 'Unit Test Corpus',
    documentType: 'Research Paper',
    publishedDate: '2025-01-01',
    authors: [],
    sourceUrl: null,
    originalFilename: null,
    storedFilePath: null,
    mimeType: 'text/plain',
    fileSizeBytes: content.length,
    pageCount: 1,
    contentHash: `hash-${id}`,
    content,
    summary: content.slice(0, 120),
    wordCount: content.split(/\s+/).length,
    chunkCount: 1,
    embeddedChunkCount: 0,
    processingStatus: 'indexed',
    processingError: null,
    embeddingStatus: 'pending',
    embeddingError: null,
    embeddingModel: null,
    isDemo: false,
    uploadedAt: '2025-01-01T00:00:00.000Z',
    createdAt: '2025-01-01T00:00:00.000Z',
    updatedAt: '2025-01-01T00:00:00.000Z',
  };
}

describe('Document Ingestion & Text Chunking', () => {
  it('cleans HTML tags, control chars, soft hyphen wraps, and normalizes whitespace', () => {
    const dirty = 'Hello <b>world</b>!\r\nRecip-\nrocal   rank\t\tfusion\u0000.\n\n\n\nSecond paragraph.';
    const cleaned = cleanDocumentText(dirty);
    assert.equal(cleaned, 'Hello world !\nReciprocal rank fusion.\n\nSecond paragraph.');
  });

  it('segments multi-page documents while preserving page numbers and overlap', () => {
    const longSentence = (n: number) =>
      `Sentence ${n} explains hybrid retrieval ranking mechanisms and document length normalization cleanly.`;
    const page1 = Array.from({ length: 12 }, (_, i) => longSentence(i + 1)).join(' ');
    const page2 = Array.from({ length: 6 }, (_, i) => longSentence(i + 20)).join(' ');

    const chunks = segmentDocumentIntoChunks({
      documentId: 'doc-1',
      documentTitle: 'Test Paper',
      source: 'Test Journal',
      pages: [
        { pageNumber: 1, text: page1 },
        { pageNumber: 2, text: page2 },
      ],
      chunkSizeWords: 60,
      overlapWords: 15,
    });

    assert.ok(chunks.length >= 3, 'Expected multiple overlapping chunks');
    assert.equal(chunks[0].pageNumber, 1);
    assert.equal(chunks[chunks.length - 1].pageNumber, 2);
    assert.equal(chunks[0].documentTitle, 'Test Paper');
    assert.ok(chunks[0].contentHash.length === 64);
  });

  it('rejects non-HTTP(S) URLs to prevent javascript: XSS links', () => {
    assert.equal(sanitizeHttpUrl('javascript:alert(1)'), null);
    assert.equal(sanitizeHttpUrl('data:text/html,hi'), null);
    assert.equal(sanitizeHttpUrl('https://example.org/paper'), 'https://example.org/paper');
  });
});

describe('BM25 Lexical Retrieval', () => {
  it('ranks relevant documents deterministically and computes positive scores', () => {
    const d1 = makeDoc(
      'd1',
      'Okapi BM25 Saturation',
      'Okapi BM25 uses term frequency saturation parameter k1 and document length normalization b.'
    );
    const d2 = makeDoc(
      'd2',
      'Ocean Alkalinity Enhancement',
      'Marine carbon dioxide removal requires coastal carbonate chemistry transects and floats.'
    );
    const docMap = new Map([
      [d1.id, d1],
      [d2.id, d2],
    ]);
    const chunks: DocumentChunk[] = [
      {
        id: 'd1::chunk-0',
        documentId: d1.id,
        documentTitle: d1.title,
        source: d1.source,
        chunkIndex: 0,
        pageNumber: 1,
        text: d1.content,
        startOffset: 0,
        endOffset: d1.content.length,
        wordCount: 13,
        contentHash: 'h1',
      },
      {
        id: 'd2::chunk-0',
        documentId: d2.id,
        documentTitle: d2.title,
        source: d2.source,
        chunkIndex: 0,
        pageNumber: 1,
        text: d2.content,
        startOffset: 0,
        endOffset: d2.content.length,
        wordCount: 12,
        contentHash: 'h2',
      },
    ];

    const run1 = runBm25Retrieval('BM25 length normalization', chunks, docMap, {
      bm25K1: 1.5,
      bm25B: 0.75,
      bm25Candidates: 10,
    });
    const run2 = runBm25Retrieval('BM25 length normalization', chunks, docMap, {
      bm25K1: 1.5,
      bm25B: 0.75,
      bm25Candidates: 10,
    });

    assert.equal(run1.candidates.length, 1);
    assert.equal(run1.candidates[0].document.id, 'd1');
    assert.ok(run1.candidates[0].bm25Score > 0);
    assert.deepEqual(run1.candidates[0].bm25Score, run2.candidates[0].bm25Score);
  });
});

describe('Vector Cosine Similarity & Reciprocal Rank Fusion (RRF)', () => {
  it('computes exact cosine similarity for aligned, orthogonal, and opposite vectors', () => {
    assert.ok(Math.abs(computeCosineSimilarity([1, 0, 0], [1, 0, 0]) - 1) < 1e-6);
    assert.ok(Math.abs(computeCosineSimilarity([1, 0], [0, 1]) - 0) < 1e-6);
    assert.ok(Math.abs(computeCosineSimilarity([1, 2], [-1, -2]) + 1) < 1e-6);
  });

  it('computes exact RRF formula RRF(d) = sum(w_i / (k + rank_i))', () => {
    const score = computeRrfScore({ bm25Rank: 1, semanticRank: 2, k: 60, bm25Weight: 1, semanticWeight: 1 });
    const expected = 1 / 61 + 1 / 62;
    assert.ok(Math.abs(score - expected) < 1e-9);
  });

  it('deduplicates multiple chunks from the same document during hybrid ranking', () => {
    const doc = makeDoc('d-dup', 'Hybrid Paper', 'Chunk A and Chunk B');
    const c1: DocumentChunk = {
      id: 'd-dup::chunk-0',
      documentId: 'd-dup',
      documentTitle: doc.title,
      source: doc.source,
      chunkIndex: 0,
      pageNumber: 1,
      text: 'Chunk A',
      startOffset: 0,
      endOffset: 7,
      wordCount: 2,
      contentHash: 'ha',
    };
    const c2: DocumentChunk = {
      id: 'd-dup::chunk-1',
      documentId: 'd-dup',
      documentTitle: doc.title,
      source: doc.source,
      chunkIndex: 1,
      pageNumber: 2,
      text: 'Chunk B',
      startOffset: 8,
      endOffset: 15,
      wordCount: 2,
      contentHash: 'hb',
    };

    const ranked = rankCandidates({
      mode: 'hybrid',
      lexicalCandidates: [
        { chunk: c1, document: doc, bm25Score: 3.2, matchedTerms: ['chunk'] },
        { chunk: c2, document: doc, bm25Score: 1.1, matchedTerms: ['chunk'] },
      ],
      semanticCandidates: [{ chunk: c1, document: doc, cosineSimilarity: 0.91 }],
      settings: {
        hybridFusionMethod: 'rrf',
        hybridAlpha: 0.5,
        rrfK: 60,
        bm25Weight: 1,
        semanticWeight: 1,
      },
      topK: 5,
    });

    assert.equal(ranked.results.length, 1);
    assert.equal(ranked.results[0].explanation.matchingChunksCount, 2);
  });
});

describe('Neural Reranking & Service Error Honesty', () => {
  it('tracks rank movement accurately and filters below threshold', () => {
    const makeItem = (id: string, rank: number): SearchResultItem => ({
      rank,
      documentId: id,
      chunkId: `${id}::chunk-0`,
      title: id,
      source: null,
      documentType: 'Research Paper',
      publishedDate: null,
      pageNumber: null,
      authors: [],
      sourceUrl: null,
      originalFilename: null,
      hasUploadedFile: false,
      isDemo: false,
      excerpt: 'excerpt',
      chunkText: 'excerpt',
      relevanceScore: 0.5,
      explanation: {
        summary: 's',
        matchedTerms: [],
        bm25Score: null,
        bm25Rank: null,
        semanticScore: null,
        semanticRank: null,
        rrfScore: null,
        fusionMethod: 'rrf',
        hybridRank: rank,
        originalPosition: rank,
        newPosition: rank,
        rankMovement: 0,
        rerankScore: null,
        rerankProvider: null,
        rerankModel: null,
        rerankExplanation: null,
        finalScore: 0.5,
        chunkId: `${id}::chunk-0`,
        chunkIndex: 0,
        pageNumber: null,
        totalChunksInDocument: 1,
        matchingChunksCount: 1,
      },
    });

    const reranked = applyValidatedRerankScores(
      [makeItem('doc-a', 1), makeItem('doc-b', 2), makeItem('doc-c', 3)],
      {
        providerType: 'llm',
        modelName: 'gemini-4-hydrogen',
        processingTimeMs: 120,
        scores: [
          { documentId: 'doc-a', chunkId: 'doc-a::chunk-0', score: 0.4, explanation: 'Partial' },
          { documentId: 'doc-b', chunkId: 'doc-b::chunk-0', score: 0.92, explanation: 'Direct answer' },
          { documentId: 'doc-c', chunkId: 'doc-c::chunk-0', score: 0.05, explanation: 'Irrelevant' },
        ],
      },
      0.2,
      5
    );

    assert.equal(reranked.length, 2);
    assert.equal(reranked[0].documentId, 'doc-b');
    assert.equal(reranked[0].explanation.originalPosition, 2);
    assert.equal(reranked[0].explanation.newPosition, 1);
    assert.equal(reranked[0].explanation.rankMovement, 1);
  });

  it('throws explicit RerankerServiceError when cross-encoder service URL is not configured', async () => {
    const provider = new CrossEncoderHttpProvider();
    await assert.rejects(
      () =>
        provider.rerank('test', [], {
          candidateCount: 5,
          batchSize: 8,
          maxSeqLength: 512,
          threshold: 0,
        }),
      (err: unknown) => err instanceof RerankerServiceError
    );
  });
});

describe('RAG Citation Validation', () => {
  it('strips hallucinated out-of-range citation indices and retains only verified chunk sources', () => {
    const evidence: SearchResultItem[] = [
      {
        rank: 1,
        documentId: 'doc-1',
        chunkId: 'doc-1::chunk-0',
        title: 'Verified Paper',
        source: 'Test Journal',
        documentType: 'Research Paper',
        publishedDate: '2025-01-01',
        pageNumber: 3,
        authors: [],
        sourceUrl: null,
        originalFilename: null,
        hasUploadedFile: false,
        isDemo: false,
        excerpt: 'Verified excerpt text.',
        chunkText: 'Verified excerpt text.',
        relevanceScore: 0.9,
        explanation: {
          summary: '',
          matchedTerms: [],
          bm25Score: null,
          bm25Rank: null,
          semanticScore: null,
          semanticRank: null,
          rrfScore: null,
          fusionMethod: null,
          hybridRank: 1,
          originalPosition: 1,
          newPosition: 1,
          rankMovement: 0,
          rerankScore: null,
          rerankProvider: null,
          rerankModel: null,
          rerankExplanation: null,
          finalScore: 0.9,
          chunkId: 'doc-1::chunk-0',
          chunkIndex: 0,
          pageNumber: 3,
          totalChunksInDocument: 2,
          matchingChunksCount: 1,
        },
      },
    ];

    const validated = validateAndFilterCitations(
      {
        summary: 'Grounded claim [1] and fabricated reference [9].',
        keyFindings: [
          { claim: 'Real finding', evidenceType: 'fact', citationIndices: [1, 99] },
          { claim: 'Unsupported hallucination', evidenceType: 'fact', citationIndices: [42] },
        ],
        usedSourceIndices: [1, 7],
      },
      evidence
    );

    assert.equal(validated.sanitizedSummary, 'Grounded claim [1] and fabricated reference .');
    assert.equal(validated.keyFindings.length, 1);
    assert.deepEqual(validated.keyFindings[0].citationIndices, [1]);
    assert.equal(validated.sourcesUsed.length, 1);
    assert.equal(validated.sourcesUsed[0].chunkId, 'doc-1::chunk-0');
    assert.ok(validated.rejectedCitationCount >= 3);
  });
});

describe('IR Evaluation Metrics', () => {
  it('computes exact Precision@K, Recall@K, MRR, MAP, and Graded NDCG@K', () => {
    const retrieved = ['d-miss', 'd-high', 'd-med', 'd-other'];
    const grades = { 'd-high': 3, 'd-med': 1 };

    assert.equal(computePrecisionAtK(retrieved, grades, 4), 0.5);
    assert.equal(computeRecallAtK(retrieved, grades, 4), 1.0);
    assert.equal(computeMrr(retrieved, grades), 0.5);

    // AP = ((1/2) + (2/3)) / 2 = 7/12
    const ap = computeAveragePrecision(retrieved, grades, 4);
    assert.ok(Math.abs(ap - 7 / 12) < 1e-6);

    // DCG = (2^3-1)/log2(3) + (2^1-1)/log2(4) = 7/log2(3) + 1/2
    // IDCG = (2^3-1)/log2(2) + (2^1-1)/log2(3) = 7 + 1/log2(3)
    const expectedNdcg =
      (7 / Math.log2(3) + 1 / Math.log2(4)) / (7 / Math.log2(2) + 1 / Math.log2(3));
    const actualNdcg = computeGradedNdcgAtK(retrieved, grades, 4);
    assert.ok(Math.abs(actualNdcg - expectedNdcg) < 1e-6);
  });

  it('sanitizes control characters and enforces max lengths on short inputs', () => {
    const clean = sanitizeShortString('hello\x00\x07  world', 8);
    assert.equal(clean, 'hello wo');
  });
});
