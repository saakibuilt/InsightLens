"""Retrieval: embed query → vector search → optional cross-encoder rerank."""
from __future__ import annotations

from dataclasses import dataclass

from insightlens.embeddings.embedder import Embedder
from insightlens.retrieval.reranker import Reranker
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk

_CANDIDATE_MULTIPLIER = 3


class RetrievalError(Exception):
    """Raised when the retrieval pipeline fails."""


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int
    company_filter: str | None = None
    preferred_chunk_types: tuple[str, ...] | None = None


class VectorSearchService:
    """Coordinates query embedding, vector lookup, and optional reranking."""

    def __init__(
        self,
        embedder: Embedder,
        repository: ChunkRepository,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._repository = repository
        self._reranker = reranker

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        if not request.query.strip():
            raise RetrievalError("Query is empty.")
        if request.top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {request.top_k}")

        query_vector = self._embedder.embed_query(request.query)

        candidate_k = request.top_k * _CANDIDATE_MULTIPLIER if self._reranker else request.top_k

        chunks = self._repository.search_similar(
            query_embedding=query_vector,
            top_k=candidate_k,
            company_filter=request.company_filter,
        )

        if self._reranker and chunks:
            chunks = self._reranker.rerank(request.query, chunks, request.top_k)

        return chunks
