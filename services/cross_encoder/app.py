"""Lucent Neural Cross-Encoder Reranking Microservice.

Loads a genuine pretrained Hugging Face / Sentence Transformers MS MARCO cross-encoder
model and exposes deterministic batch scoring endpoints for second-stage document reranking.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [lucent-cross-encoder] %(message)s",
)
logger = logging.getLogger("lucent.cross_encoder")

MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MODEL_VERSION = os.getenv("CROSS_ENCODER_MODEL_VERSION", "huggingface-ms-marco-v1")
DEFAULT_MAX_LENGTH = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))
DEFAULT_BATCH_SIZE = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "16"))
MAX_CANDIDATES_PER_REQUEST = int(os.getenv("CROSS_ENCODER_MAX_CANDIDATES", "100"))
MAX_BATCH_QUERIES = int(os.getenv("CROSS_ENCODER_MAX_BATCH_QUERIES", "20"))
INFERENCE_SEED = int(os.getenv("CROSS_ENCODER_SEED", "42"))
REQUESTED_DEVICE = os.getenv("CROSS_ENCODER_DEVICE", "auto").lower()


def resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_deterministic_inference(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CandidateDocument(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=200)
    chunk_id: Optional[str] = Field(default=None, max_length=250)
    text: str = Field(..., min_length=1, max_length=12000)


class RerankRequest(BaseModel):
    query: str = Field(..., max_length=1000)
    documents: List[CandidateDocument] = Field(default_factory=list)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=64)
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=64, le=1024)

    @field_validator("documents")
    @classmethod
    def check_doc_limit(cls, docs: List[CandidateDocument]) -> List[CandidateDocument]:
        if len(docs) > MAX_CANDIDATES_PER_REQUEST:
            raise ValueError(
                f"Candidate document count ({len(docs)}) exceeds limit of {MAX_CANDIDATES_PER_REQUEST}."
            )
        return docs


class RankedDocumentResult(BaseModel):
    document_id: str
    chunk_id: Optional[str] = None
    score: float
    rank: int


class RerankResponse(BaseModel):
    model_name: str
    model_version: str
    device: str
    processing_time_ms: float
    results: List[RankedDocumentResult]


class BatchRerankRequest(BaseModel):
    requests: List[RerankRequest] = Field(..., min_length=1, max_length=MAX_BATCH_QUERIES)


class BatchRerankResponse(BaseModel):
    model_name: str
    model_version: str
    device: str
    total_processing_time_ms: float
    responses: List[RerankResponse]


class CrossEncoderEngine:
    def __init__(self) -> None:
        self.model: Optional[CrossEncoder] = None
        self.device: str = resolve_device(REQUESTED_DEVICE)
        self.loaded_at: Optional[float] = None
        self.load_error: Optional[str] = None

    def load(self) -> None:
        configure_deterministic_inference(INFERENCE_SEED)
        try:
            logger.info("Loading pretrained cross-encoder '%s' on device '%s'...", MODEL_NAME, self.device)
            self.model = CrossEncoder(
                MODEL_NAME,
                max_length=DEFAULT_MAX_LENGTH,
                device=self.device,
            )
            if hasattr(self.model, "model"):
                self.model.model.eval()
            self.loaded_at = time.time()
            self.load_error = None
            logger.info("Cross-encoder ready on %s.", self.device)
        except Exception as exc:
            self.load_error = str(exc)
            logger.exception("Failed to load cross-encoder model: %s", exc)
            raise

    def score_request(self, req: RerankRequest) -> RerankResponse:
        if self.model is None:
            raise HTTPException(status_code=503, detail="Cross-encoder model is not loaded.")

        start = time.perf_counter()
        query = req.query.strip()
        if not query or len(req.documents) == 0:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            return RerankResponse(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                device=self.device,
                processing_time_ms=elapsed_ms,
                results=[],
            )

        pairs = [(query, doc.text) for doc in req.documents]

        # Enforce requested sequence length bound safely
        if hasattr(self.model, "max_length"):
            self.model.max_length = min(req.max_length, 1024)

        with torch.no_grad():
            raw_scores = self.model.predict(
                pairs,
                batch_size=req.batch_size,
                show_progress_bar=False,
            )

        scored = [
            (doc, float(score))
            for doc, score in zip(req.documents, raw_scores)
        ]
        # Deterministic sort: descending cross-encoder score, tie-break by document_id
        scored.sort(key=lambda item: (-item[1], item[0].document_id))

        ranked = [
            RankedDocumentResult(
                document_id=doc.document_id,
                chunk_id=doc.chunk_id,
                score=round(score, 6),
                rank=idx + 1,
            )
            for idx, (doc, score) in enumerate(scored)
        ]

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return RerankResponse(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            device=self.device,
            processing_time_ms=elapsed_ms,
            results=ranked,
        )


engine = CrossEncoderEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("CROSS_ENCODER_SKIP_LOAD", "false").lower() != "true":
        engine.load()
    yield


app = FastAPI(
    title="Lucent Cross-Encoder Reranking Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    if engine.model is None:
        raise HTTPException(
            status_code=503,
            detail=engine.load_error or "Model has not finished loading.",
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "device": engine.device,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "device": engine.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "max_length": DEFAULT_MAX_LENGTH,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "max_candidates_per_request": MAX_CANDIDATES_PER_REQUEST,
        "deterministic_seed": INFERENCE_SEED,
        "model_loaded": engine.model is not None,
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank_single(req: RerankRequest) -> RerankResponse:
    return engine.score_request(req)


@app.post("/rerank/batch", response_model=BatchRerankResponse)
def rerank_batch(batch_req: BatchRerankRequest) -> BatchRerankResponse:
    start = time.perf_counter()
    responses = [engine.score_request(item) for item in batch_req.requests]
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return BatchRerankResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        device=engine.device,
        total_processing_time_ms=elapsed_ms,
        responses=responses,
    )
