# Lucent – Semantic Search & AI Ranking Engine

**Find what matters. Understand why.**

Lucent is a full-stack information retrieval, neural ranking, and Retrieval-Augmented Generation (RAG) research platform engineered with React, TypeScript, Express, server-side Google Gemini models, a persistent local Float32 vector index, and an independently deployable Python FastAPI + PyTorch + Sentence Transformers cross-encoder microservice.

---

## Architecture Diagram

```text
+---------------------------------------------------------------------------+
|                     React + TypeScript SPA (Port 3000)                    |
|   Search  |  Document Library  |  Pipeline  |  Evaluation Lab  | Analytics|
+-------------------------------------+-------------------------------------+
                                      | HTTPS / JSON / Multipart Upload
                                      v
+---------------------------------------------------------------------------+
|                 Express Backend Server (server.ts)                        |
|  [Rate Limiter & Request Logger] -> [Input Validation & Path Confinement] |
|                                                                           |
|  +-----------------------+    +----------------------------------------+  |
|  | ingestion.ts          |    | retrieval.ts & ranking.ts              |  |
|  | - PDF (unpdf, pages)  |    | - Okapi BM25 (k1, b, IDF, norm)        |  |
|  | - DOCX (mammoth)      |    | - Dense FlatIP Vector Search (Cosine)  |  |
|  | - TXT/MD cleaning     |    | - Reciprocal Rank Fusion (RRF)         |  |
|  | - Sliding-window      |    | - Document Deduplication & Provenance  |  |
|  |   chunking & SHA-256  |    +-------------------+--------------------+  |
|  +-----------+-----------+                        |                       |
|              |                                    v                       |
|              v                        +-----------+--------------------+  |
|  +-----------+-----------+            | reranking.ts                   |  |
|  | vectorIndex.ts        |            | - CrossEncoderHttpProvider ----+--+--> Python FastAPI
|  | Persistent Float32    |            | - GeminiLlmRerankerProvider    |  |    PyTorch + MS MARCO
|  | FlatIP Index (.f32)   |            +-----------+--------------------+  |    Cross-Encoder
|  +-----------------------+                        |                       |    (services/cross_encoder)
|                                                   v                       |
|  +-----------------------+            +-----------+--------------------+  |
|  | evaluation.ts         |            | generation.ts (RAG Pipeline)   |  |
|  | P@K, R@K, MRR, MAP,   |            | - Query Intent Classification  |  |
|  | Graded NDCG@K, Reports|            | - Grounded Answer Synthesis    |  |
|  +-----------------------+            | - Strict Citation Validation   |  |
|                                       +-----------+--------------------+  |
+---------------------------------------------------+-----------------------+
                                                    | Server-side SDK only
                                                    v
                                     Google Gemini API (@google/genai)
                                     - gemini-embedding-2
                                     - gemini-4-hydrogen
```

---

## Installation & Quick Start

```bash
npm install
cp .env.example .env
# Add GEMINI_API_KEY in .env or Settings > Secrets
npm run dev
```

Run tests and type checks:

```bash
npx tsx --test tests/lucent.test.ts
npm run lint
npm run build
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | For semantic search & RAG | — | Server-side Google Gemini API credential (never exposed to browser). |
| `CROSS_ENCODER_SERVICE_URL` | Optional | `""` | Base URL of the external Python FastAPI MS MARCO cross-encoder service. |
| `CROSS_ENCODER_TIMEOUT_MS` | Optional | `15000` | Request timeout in ms when calling the cross-encoder microservice. |
| `LUCENT_GENERATION_MODEL` | Optional | `gemini-4-hydrogen` | Server-side model for query classification, optional LLM reranking, and RAG synthesis. |
| `LUCENT_EMBEDDING_MODEL` | Optional | `gemini-embedding-2` | Server-side Google embedding model for chunk and query vectors. |
| `LUCENT_DATA_DIR` | Optional | `./data` | Directory for persisted documents, uploads, state JSON, and Float32 vector index. |
| `LUCENT_RATE_LIMIT_PER_MIN` | Optional | `120` | Sliding-window API rate limit per client IP. |

---

## Models & Ranking Methodology

1. **Lexical Retrieval (Okapi BM25):** Deterministic sparse matching with stopword filtering, smoothed IDF `log(1 + (N - df + 0.5)/(df + 0.5))`, term saturation `k1` (default `1.5`), and document length normalization `b` (default `0.75`).
2. **Dense Vector Retrieval (`gemini-embedding-2` + Local FlatIP Float32 Index):** Chunk embeddings are L2-normalized and persisted as contiguous `Float32Array` binary records (`data/vector-index.f32`). Unchanged chunks are skipped via SHA-256 content hashes.
3. **Hybrid Fusion (Reciprocal Rank Fusion):** Combines lexical and dense candidate ranks via `RRF(d) = w_bm25 / (k + rank_bm25(d)) + w_sem / (k + rank_sem(d))` with configurable `k` (default `60`) and weights, plus optional weighted linear score blending.
4. **Second-Stage Reranking:**
   - **Cross-Encoder (`services/cross_encoder`):** Genuine pretrained Hugging Face MS MARCO cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) running in Python/FastAPI/PyTorch. Returns an explicit service error if unavailable (never silently substituted).
   - **LLM-Based Reranker (Optional):** Structured Gemini relevance judge clearly distinguished from a trained cross-encoder.
5. **RAG Pipeline (`gemini-4-hydrogen`):** Classifies query intent (`factual`, `comparative`, `historical`, `analytical`, `explanatory`), synthesizes grounded responses in 4 selectable formats, tags claims as `fact` vs `interpretation`, flags conflicting evidence and evidence gaps, and programmatically strips any citation index not grounded in the retrieved passages.

---

## Evaluation Methodology

The ML Evaluation Laboratory computes authentic offline IR metrics over human-labelled or clearly marked demonstration query sets with graded relevance labels (`0`–`3`):

- **Precision@K** and **Recall@K** (binary relevance `grade > 0`)
- **MRR** (Mean Reciprocal Rank)
- **MAP** (Mean Average Precision)
- **NDCG@K** using exponential graded gain `(2^rel - 1) / log2(rank + 1)` normalized by Ideal DCG
- Automated error analysis (zero-relevant queries, low-recall queries, documents ranked below top-3, reranking regressions vs. Hybrid RRF, and BM25 vs. Dense divergence) with downloadable Markdown experiment reports.

---

## Key REST API Endpoints

- `POST /api/search` — Execute Keyword, Semantic, or Hybrid search with optional reranking and RAG synthesis.
- `POST /api/search/compare` — Compare BM25, Dense, Hybrid, Cross-Encoder, and LLM Reranker side-by-side.
- `GET /api/documents`, `POST /api/documents`, `POST /api/documents/upload`, `POST /api/documents/:id/reprocess`, `POST /api/documents/:id/embed`, `GET /api/documents/:id/download`, `DELETE /api/documents/:id`
- `GET /api/pipeline/status`, `GET /api/vector-index/stats`, `GET /api/reranker/status`
- `GET /api/evaluation`, `POST /api/evaluation/queries`, `PATCH /api/evaluation/queries/:id`, `POST /api/evaluation/import`, `POST /api/evaluation/run`, `GET /api/evaluation/runs/:id/report`
- `GET /api/analytics`, `DELETE /api/analytics/queries`
- `GET /api/settings`, `PATCH /api/settings`

---

## Deploying the Python Cross-Encoder Service

See [`services/cross_encoder/README.md`](./services/cross_encoder/README.md) for local Docker and Google Cloud Run deployment instructions.

---

## Known Limitations

- The local FlatIP Float32 vector index performs exact linear inner-product scans; for corpora exceeding ~100,000 chunks, swap `VectorIndexBackend` in `server/modules/vectorIndex.ts` for an approximate nearest-neighbor store (FAISS IVF/HNSW, Qdrant, or pgvector).
- Image-only scanned PDFs without embedded text layers are rejected with an explicit extraction error because OCR is not bundled.
- The included sample corpus and initial benchmark queries are explicitly marked **Demonstration Data** and should be replaced or supplemented with domain-specific human-labelled judgments for production benchmarking.
