import os
from typing import List, Tuple

os.environ["CROSS_ENCODER_SKIP_LOAD"] = "true"

from fastapi.testclient import TestClient  # noqa: E402
import app as reranker_app  # noqa: E402


class DeterministicTestCrossEncoder:
    """Test double verifying request validation, sorting, batching, and error paths without downloading weights."""

    max_length = 512

    def predict(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
        show_progress_bar: bool = False,
    ) -> List[float]:
        scores: List[float] = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            scores.append(float(len(q_words.intersection(t_words))))
        return scores


client = TestClient(reranker_app.app)


def setup_function():
    reranker_app.engine.model = DeterministicTestCrossEncoder()
    reranker_app.engine.device = "cpu"


def test_health_and_model_info():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    info = client.get("/model-info")
    assert info.status_code == 200
    body = info.json()
    assert body["model_name"] == reranker_app.MODEL_NAME
    assert body["model_loaded"] is True


def test_rerank_orders_candidates_descending():
    res = client.post(
        "/rerank",
        json={
            "query": "reciprocal rank fusion bm25",
            "documents": [
                {"document_id": "doc-low", "text": "unrelated ocean chemistry passage"},
                {"document_id": "doc-high", "text": "reciprocal rank fusion combines bm25 scores"},
            ],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["document_id"] == "doc-high"
    assert data["results"][0]["rank"] == 1
    assert data["results"][1]["document_id"] == "doc-low"
    assert data["processing_time_ms"] >= 0


def test_empty_query_and_empty_documents():
    res = client.post("/rerank", json={"query": "   ", "documents": []})
    assert res.status_code == 200
    assert res.json()["results"] == []


def test_batch_rerank_endpoint():
    res = client.post(
        "/rerank/batch",
        json={
            "requests": [
                {
                    "query": "iron air battery",
                    "documents": [
                        {"document_id": "d1", "text": "iron air battery storage"},
                        {"document_id": "d2", "text": "clinical sepsis alert"},
                    ],
                },
                {
                    "query": "sepsis alert",
                    "documents": [
                        {"document_id": "d1", "text": "iron air battery storage"},
                        {"document_id": "d2", "text": "clinical sepsis alert"},
                    ],
                },
            ]
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body["responses"]) == 2
    assert body["responses"][0]["results"][0]["document_id"] == "d1"
    assert body["responses"][1]["results"][0]["document_id"] == "d2"


def test_service_unavailable_when_model_unloaded():
    reranker_app.engine.model = None
    res = client.post(
        "/rerank",
        json={"query": "test", "documents": [{"document_id": "d1", "text": "sample"}]},
    )
    assert res.status_code == 503
