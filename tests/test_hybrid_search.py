"""Unit tests for version scoring, chunk-type scoring, and query classification."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from insightlens.retrieval.hybrid_search import HybridSearchService, _NUMERIC_QUERY_RE
from insightlens.retrieval.vector_search import RetrievalRequest
from insightlens.storage.chunk_repository import RetrievedChunk


def _chunk(chunk_id, doc_id, chunk_type="body", supersedes=None, text="sample"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        company="VICI",
        file_name="test.pdf",
        page_number=1,
        chunk_text=text,
        chunk_type=chunk_type,
        similarity=0.8,
        supersedes_document_id=supersedes,
        version_label=None,
        section_header=None,
        structured_content=None,
    )


class _Stub:
    def embed_query(self, q):
        return [0.0] * 768

    def search_similar(self, **_):
        return []


def _service():
    return HybridSearchService(_Stub(), _Stub(), [])


@pytest.mark.parametrize("query", [
    "What was VICI's FFO per share in Q3 2024?",
    "What is the cap rate on recent acquisitions?",
    "How much did NOI grow year-over-year?",
    "What are the key operating metrics?",
    "EBITDA margin expansion trend",
    "$6.2bn revenue target",
    "3.5x coverage ratio",
    "What is the dividend yield?",
])
def test_numeric_queries_are_detected(query):
    assert bool(_NUMERIC_QUERY_RE.search(query)), f"Expected numeric: {query!r}"


@pytest.mark.parametrize("query", [
    "What is the investment thesis for VICI?",
    "Describe the management team strategy.",
    "Why does VICI focus on experiential real estate?",
    "What are the key risks?",
    "Tell me about the portfolio composition.",
])
def test_narrative_queries_are_not_detected(query):
    assert not bool(_NUMERIC_QUERY_RE.search(query)), f"Expected narrative: {query!r}"



def test_current_version_gets_boost():
    svc = _service()
    current = _chunk("c1", "V2", supersedes="V1")
    scored = [(current, 1.0)]
    result = svc._apply_version_scores(scored)
    assert result[0][1] > 1.0, "Current-version chunk should score above 1.0"


def test_superseded_version_gets_penalty():
    svc = _service()
    old = _chunk("c_old", "V1")
    new = _chunk("c_new", "V2", supersedes="V1")
    scored = [(old, 1.0), (new, 1.0)]
    result = dict(svc._apply_version_scores(scored))
    assert result[old] < 1.0, "Superseded chunk should score below 1.0"
    assert result[new] > 1.0, "Current chunk should score above 1.0"


def test_unversioned_doc_score_unchanged():
    svc = _service()
    solo = _chunk("s1", "SOLO")
    result = svc._apply_version_scores([(solo, 1.0)])
    assert result[0][1] == pytest.approx(1.0), "Unversioned chunk score should not change"


def test_multi_hop_version_chain():
    """V1 → V2 → V3: only V3 is CURRENT; V1 and V2 are both SUPERSEDED."""
    svc = _service()
    v1 = _chunk("v1", "V1")
    v2 = _chunk("v2", "V2", supersedes="V1")
    v3 = _chunk("v3", "V3", supersedes="V2")
    scores = dict(svc._apply_version_scores([(v1, 1.0), (v2, 1.0), (v3, 1.0)]))
    assert scores[v3] > 1.0, "V3 is the current version — should be boosted"
    assert scores[v2] < 1.0, "V2 is superseded by V3 — should be penalised"
    assert scores[v1] < 1.0, "V1 is superseded by V2 — should be penalised"



def test_financial_table_boosted_on_numeric_query():
    svc = _service()
    table = _chunk("t1", "D1", chunk_type="financial_table")
    body  = _chunk("b1", "D1", chunk_type="body")
    req = RetrievalRequest(query="What is the FFO per share?", top_k=5)
    result = dict(svc._apply_chunk_type_scores([(table, 1.0), (body, 1.0)], req))
    assert result[table] > result[body], "financial_table should outscore body on numeric query"


def test_body_boosted_on_narrative_query():
    svc = _service()
    table = _chunk("t1", "D1", chunk_type="financial_table")
    body  = _chunk("b1", "D1", chunk_type="body")
    req = RetrievalRequest(query="What is the investment thesis?", top_k=5)
    result = dict(svc._apply_chunk_type_scores([(table, 1.0), (body, 1.0)], req))
    assert result[body] > result[table], "body should outscore financial_table on narrative query"


def test_preferred_chunk_types_override_auto_detection():
    svc = _service()
    table = _chunk("t1", "D1", chunk_type="financial_table")
    body  = _chunk("b1", "D1", chunk_type="body")
    req = RetrievalRequest(
        query="What is the investment thesis?",
        top_k=5,
        preferred_chunk_types=("financial_table",),
    )
    result = dict(svc._apply_chunk_type_scores([(table, 1.0), (body, 1.0)], req))
    assert result[table] > result[body], "Explicit preferred_chunk_types should override auto-detection"
