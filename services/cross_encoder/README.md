# Lucent Cross-Encoder Reranking Service (Python + FastAPI + PyTorch)

Standalone neural reranking microservice that scores `(query, passage)` pairs using a pretrained Hugging Face Sentence Transformers MS MARCO cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default).

## Architecture & Communication Flow

1. The React/TypeScript frontend sends search requests to Lucent's Node/Express backend (`POST /api/search` with `rerankerMode: "cross_encoder"`).
2. The Express backend executes BM25 + dense vector retrieval and Reciprocal Rank Fusion (RRF), selecting the top candidate chunks.
3. `server/modules/reranking.ts` (`CrossEncoderHttpProvider`) posts the query and candidate chunks to `${CROSS_ENCODER_SERVICE_URL}/rerank` over HTTPS with a configurable request timeout.
4. This Python FastAPI service runs joint transformer inference over `(query, passage)` pairs on GPU (CUDA/MPS) or CPU fallback and returns exact relevance scores and processing latency.
5. If this service is unreachable or unconfigured, Lucent surfaces an explicit service error and **never** substitutes fabricated scores.

## Local Execution

```bash
cd services/cross_encoder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

Run unit tests:

```bash
pytest -v
```

## Deploying to Google Cloud Run

```bash
gcloud builds submit services/cross_encoder \
  --tag gcr.io/<YOUR_PROJECT_ID>/lucent-cross-encoder

gcloud run deploy lucent-cross-encoder \
  --image gcr.io/<YOUR_PROJECT_ID>/lucent-cross-encoder \
  --region europe-west2 \
  --memory 2Gi \
  --cpu 2 \
  --port 8001 \
  --set-env-vars CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --allow-unauthenticated
```

After deployment, copy the resulting Cloud Run service URL and configure it on the Lucent Node backend via:

```bash
CROSS_ENCODER_SERVICE_URL="https://lucent-cross-encoder-xxxx-nw.a.run.app"
```
