"""Systematic retrieval quality evaluation — recall@k and MRR."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.json"
TOP_K_VALUES = [5, 8, 12]


def _has_credentials() -> bool:
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "ANTHROPIC_API_KEY"]
    return all(os.getenv(k) for k in required)


pytestmark = pytest.mark.skipif(
    not _has_credentials(),
    reason="Snowflake credentials not set — skipping integration eval",
)


@pytest.fixture(scope="module")
def retrieval_stack():
    """Build the hybrid retrieval stack once for the whole eval run."""
    from dotenv import load_dotenv
    load_dotenv()

    from insightlens.config import load_config
    from insightlens.embeddings.embedder import Embedder
    from insightlens.retrieval.hybrid_search import HybridSearchService
    from insightlens.retrieval.reranker import Reranker
    from insightlens.storage.chunk_repository import ChunkRepository
    from insightlens.storage.snowflake_client import open_connection

    cfg = load_config()
    embedder = Embedder(model=cfg.embedding_model)
    reranker = Reranker()

    with open_connection(cfg.snowflake) as conn:
        repo = ChunkRepository(conn)
        corpus = repo.get_all_chunks()

    return cfg, embedder, corpus, reranker


def _recall_at_k(retrieved_texts: list[str], keywords: list[str], k: int) -> bool:
    """Return True if any of the top-k chunks contain at least one expected keyword."""
    for text in retrieved_texts[:k]:
        text_lower = text.lower()
        if any(kw.lower() in text_lower for kw in keywords):
            return True
    return False


def _reciprocal_rank(retrieved_texts: list[str], keywords: list[str]) -> float:
    """Return 1/rank of the first chunk containing a keyword, 0 if none found."""
    for rank, text in enumerate(retrieved_texts, start=1):
        text_lower = text.lower()
        if any(kw.lower() in text_lower for kw in keywords):
            return 1.0 / rank
    return 0.0


def test_retrieval_quality(retrieval_stack):
    """Run the full golden Q&A eval and report recall@k and MRR."""
    from insightlens.retrieval.hybrid_search import HybridSearchService
    from insightlens.retrieval.vector_search import RetrievalRequest
    from insightlens.storage.chunk_repository import ChunkRepository
    from insightlens.storage.snowflake_client import open_connection

    cfg, embedder, corpus, reranker = retrieval_stack
    questions = json.loads(GOLDEN_QA_PATH.read_text())

    results: list[dict] = []

    with open_connection(cfg.snowflake) as conn:
        repo = ChunkRepository(conn)
        service = HybridSearchService(
            embedder=embedder,
            repository=repo,
            corpus_chunks=corpus,
            reranker=reranker,
        )

        for qa in questions:
            chunks = service.retrieve(
                RetrievalRequest(
                    query=qa["question"],
                    top_k=max(TOP_K_VALUES),
                    company_filter=qa.get("expected_company"),
                )
            )
            retrieved_texts = [c.chunk_text for c in chunks]
            keywords = qa["expected_keywords"]

            row = {
                "id": qa["id"],
                "question": qa["question"][:60] + "…",
                "rr": _reciprocal_rank(retrieved_texts, keywords),
            }
            for k in TOP_K_VALUES:
                row[f"recall@{k}"] = _recall_at_k(retrieved_texts, keywords, k)
            results.append(row)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'ID':<12} {'recall@5':>8} {'recall@8':>8} {'recall@12':>9} {'MRR':>6}  Question")
    print("-" * 72)
    for r in results:
        flags = "  ".join("✓" if r[f"recall@{k}"] else "✗" for k in TOP_K_VALUES)
        print(f"{r['id']:<12} {flags}   {r['rr']:.2f}  {r['question']}")

    print("=" * 72)
    mrr = sum(r["rr"] for r in results) / len(results)
    for k in TOP_K_VALUES:
        hits = sum(1 for r in results if r[f"recall@{k}"])
        print(f"recall@{k}: {hits}/{len(results)} = {hits/len(results):.0%}")
    print(f"MRR:       {mrr:.3f}")
    print("=" * 72)

    # Soft assertion: at least 50% recall@8 expected for a working system
    recall_8 = sum(1 for r in results if r["recall@8"]) / len(results)
    assert recall_8 >= 0.50, (
        f"recall@8 is only {recall_8:.0%} — retrieval quality is below the minimum threshold. "
        "Check chunking, BM25 index, or golden_qa keywords."
    )
