"""Token-aware chunking with slide-deck awareness for investment documents."""
from __future__ import annotations

import json
from dataclasses import dataclass

import tiktoken

from insightlens.ingestion.pdf_parser import ParsedPage


class ChunkingError(Exception):
    """Raised when chunking parameters are invalid or text cannot be tokenized."""


@dataclass(frozen=True)
class TextChunk:
    text: str
    token_count: int
    page_number: int
    chunk_index: int
    section_header: str | None = None
    chunk_type: str = "body"
    structured_content: str | None = None


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveTokenChunker:
    """Splits text into chunks bounded by token count, preferring natural boundaries."""

    def __init__(self, chunk_size_tokens: int, overlap_tokens: int, encoding_name: str = "cl100k_base") -> None:
        if chunk_size_tokens <= 0:
            raise ChunkingError(f"chunk_size_tokens must be positive, got {chunk_size_tokens}")
        if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
            raise ChunkingError(
                f"overlap_tokens ({overlap_tokens}) must be in [0, chunk_size_tokens={chunk_size_tokens})"
            )

        self._chunk_size = chunk_size_tokens
        self._overlap = overlap_tokens
        self._encoder = tiktoken.get_encoding(encoding_name)

    def chunk_page(self, text: str, page_number: int, starting_chunk_index: int) -> list[TextChunk]:
        if not text.strip():
            return []

        pieces = self._split_recursively(text, _SEPARATORS)
        chunks = self._merge_pieces_to_chunks(pieces)

        return [
            TextChunk(
                text=chunk_text,
                token_count=len(self._encoder.encode(chunk_text)),
                page_number=page_number,
                chunk_index=starting_chunk_index + offset,
            )
            for offset, chunk_text in enumerate(chunks)
        ]

    def _split_recursively(self, text: str, separators: list[str]) -> list[str]:
        if len(self._encoder.encode(text)) <= self._chunk_size:
            return [text]

        if not separators:
            return self._hard_split_by_tokens(text)

        separator = separators[0]
        rest = separators[1:]

        if separator == "":
            return self._hard_split_by_tokens(text)

        parts = text.split(separator)
        result: list[str] = []
        for part in parts:
            if len(self._encoder.encode(part)) <= self._chunk_size:
                result.append(part)
            else:
                result.extend(self._split_recursively(part, rest))
        return result

    def _hard_split_by_tokens(self, text: str) -> list[str]:
        tokens = self._encoder.encode(text)
        return [
            self._encoder.decode(tokens[i : i + self._chunk_size])
            for i in range(0, len(tokens), self._chunk_size)
        ]

    def _merge_pieces_to_chunks(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current_tokens: list[int] = []

        for piece in pieces:
            piece_tokens = self._encoder.encode(piece)
            if not piece_tokens:
                continue

            if len(current_tokens) + len(piece_tokens) <= self._chunk_size:
                if current_tokens:
                    current_tokens.extend(self._encoder.encode(" "))
                current_tokens.extend(piece_tokens)
            else:
                if current_tokens:
                    chunks.append(self._encoder.decode(current_tokens))
                    overlap_start = max(0, len(current_tokens) - self._overlap)
                    current_tokens = current_tokens[overlap_start:] + piece_tokens
                else:
                    current_tokens = piece_tokens

        if current_tokens:
            chunks.append(self._encoder.decode(current_tokens))

        return chunks


class SlideAwareChunker:
    """Keeps each slide page as a coherent chunk unit instead of cutting at fixed token counts.

    Investment decks are slide-based: each page is one self-contained idea. Splitting
    across slide boundaries produces half-thoughts. This chunker:
      1. Merges tiny title-only pages (<30 tokens) into the following content page.
      2. Tags every chunk with the slide title as section_header.
      3. Creates a separate financial_table chunk for each extracted table, preserving
         the original row/column structure as JSON in structured_content.
    """

    _TITLE_SLIDE_TOKEN_THRESHOLD = 30

    def __init__(
        self,
        chunk_size_tokens: int,
        overlap_tokens: int,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._inner = RecursiveTokenChunker(chunk_size_tokens, overlap_tokens, encoding_name)
        self._encoder = tiktoken.get_encoding(encoding_name)

    def chunk_document(self, pages: list[ParsedPage]) -> list[TextChunk]:
        merged = self._merge_title_pages(pages)
        result: list[TextChunk] = []
        running_index = 0

        for page_number, text, section_header, page_tables, is_visual in merged:
            chunk_type = _detect_chunk_type(text, is_visual, bool(page_tables))

            # Text chunks from this slide
            if text.strip():
                raw_chunks = self._inner.chunk_page(text, page_number, running_index)
                for chunk in raw_chunks:
                    result.append(TextChunk(
                        text=chunk.text,
                        token_count=chunk.token_count,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        section_header=section_header,
                        chunk_type=chunk_type,
                    ))
                running_index += len(raw_chunks)

            # One structured chunk per extracted table on this slide
            for table in page_tables:
                flat = [[cell for cell in row] for row in table if row]
                if not flat:
                    continue
                table_text = _table_to_text(flat)
                if not table_text.strip():
                    continue
                result.append(TextChunk(
                    text=table_text,
                    token_count=len(self._encoder.encode(table_text)),
                    page_number=page_number,
                    chunk_index=running_index,
                    section_header=section_header,
                    chunk_type="financial_table",
                    structured_content=json.dumps(flat),
                ))
                running_index += 1

        return result

    def _merge_title_pages(
        self, pages: list[ParsedPage]
    ) -> list[tuple[int, str, str | None, tuple, bool]]:
        """Merge title-only slides into the next content slide.

        Returns list of (page_number, text, section_header, tables, is_visual).
        """
        result: list[tuple[int, str, str | None, tuple, bool]] = []
        pending_text: str = ""
        pending_header: str | None = None

        for page in pages:
            token_count = len(self._encoder.encode(page.text)) if page.text else 0

            # Build effective text for this page: vision content first (so BM25 and
            # the embedder see chart/map values), then the text-extracted content.
            vision = getattr(page, "vision_text", None)
            if vision:
                effective_text = (
                    f"[AI VISION EXTRACTION — chart/map/logo content]\n{vision}"
                    + (f"\n\n{page.text}" if page.text.strip() else "")
                ).strip()
            else:
                effective_text = page.text

            if token_count < self._TITLE_SLIDE_TOKEN_THRESHOLD and not pending_text:
                # Title-only page: carry it forward to attach to the next content page
                pending_text = effective_text
                pending_header = page.slide_title
            else:
                full_text = (pending_text + "\n\n" + effective_text).strip() if pending_text else effective_text
                header = pending_header if pending_header else page.slide_title
                result.append((page.page_number, full_text, header, page.tables, page.is_likely_visual))
                pending_text = ""
                pending_header = None

        if pending_text:
            result.append((pages[-1].page_number, pending_text, pending_header, (), False))

        return result


def _table_to_text(table: list[list[str]]) -> str:
    """Convert a table (list of rows) to pipe-delimited text for embedding."""
    return "\n".join(" | ".join(cell for cell in row) for row in table)


def _detect_chunk_type(text: str, is_likely_visual: bool, has_tables: bool) -> str:
    if is_likely_visual:
        return "chart_caption"
    if has_tables:
        return "financial_table"
    numeric_chars = sum(1 for c in text if c.isdigit() or c in ".,%-$")
    if text and numeric_chars / len(text) > 0.15:
        return "chart_caption"
    return "body"
