"""Walks data/raw_pdfs, parses + chunks + embeds + persists every PDF."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from insightlens.config import RAW_PDF_DIR, ConfigError, load_config
from insightlens.embeddings.embedder import EmbeddingError, Embedder
from insightlens.ingestion.chunker import ChunkingError, SlideAwareChunker
from insightlens.ingestion.document_metadata import extract_metadata
from insightlens.ingestion.pdf_parser import PDFParsingError, parse_pdf
from insightlens.storage.chunk_repository import (
    ChunkRecord,
    ChunkRepository,
    DocumentRecord,
    RepositoryError,
)
from insightlens.storage.snowflake_client import (
    SnowflakeConnectionError,
    open_connection,
)

"""@ 2026 Developed by Saksham Nirula"""


def _document_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}-{chunk_index:05d}"


def _process_pdf(
    path: Path,
    chunker: SlideAwareChunker,
    embedder: Embedder,
    repository: ChunkRepository,
) -> int:
    print(f"[ingest] Parsing {path.name}")
    parsed = parse_pdf(path)

    first_page_text = parsed.pages[0].text if parsed.pages else ""
    metadata = extract_metadata(path, first_page_text)

    category = path.parent.name.replace("_", " ").title()
    if category.lower() in ("raw pdfs", "raw_pdfs", "raw data", "raw_data", "data", "epstein"):
        category = metadata.company or path.stem.replace("_", " ").title()

    document_id = _document_id(path)
    repository.delete_document(document_id)
    repository.upsert_document(
        DocumentRecord(
            document_id=document_id,
            file_name=path.name[:255],
            company=category[:255],
            document_type=metadata.document_type or "Document",
            version_label=metadata.version_label,
            version_date=metadata.version_date,
            page_count=parsed.total_pages,
        )
    )

    slide_chunks = chunker.chunk_document(parsed.pages)

    if not slide_chunks:
        print(f"[ingest] {path.name} produced no chunks (empty or visual-only document)")
        return 0

    print(f"[ingest] Embedding {len(slide_chunks)} chunks from {path.name}")
    embeddings = embedder.embed_texts([c.text for c in slide_chunks])

    page_images = {p.page_number: getattr(p, 'page_image_b64', None) for p in parsed.pages}

    def _make_structured(chunk):
        img_b64 = page_images.get(chunk.page_number)
        if not img_b64:
            return chunk.structured_content
        data = {"image": img_b64}
        if chunk.structured_content:
            try:
                parsed_sc = json.loads(chunk.structured_content)
                if isinstance(parsed_sc, list):
                    data["table"] = parsed_sc
                elif isinstance(parsed_sc, dict):
                    data.update(parsed_sc)
            except Exception:
                pass
        return json.dumps(data)

    records = [
        ChunkRecord(
            chunk_id=_chunk_id(document_id, chunk.chunk_index),
            document_id=document_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            chunk_text=chunk.text,
            token_count=chunk.token_count,
            embedding=result.vector,
            section_header=chunk.section_header,
            chunk_type=chunk.chunk_type,
            structured_content=_make_structured(chunk),
        )
        for chunk, result in zip(slide_chunks, embeddings, strict=True)
    ]

    inserted = repository.insert_chunks(records)
    print(f"[ingest] Inserted {inserted} chunks for {path.name}")
    return inserted


def _set_document_relationships(repository: ChunkRepository) -> None:
    """For each company with multiple documents, mark the newer one as superseding the older."""
    docs = repository.get_all_documents()
    by_company: dict[str, list[DocumentRecord]] = defaultdict(list)
    for doc in docs:
        if doc.company:
            by_company[doc.company].append(doc)

    for company, company_docs in by_company.items():
        sorted_docs = sorted(company_docs, key=lambda d: d.version_date or date.min)
        for i in range(1, len(sorted_docs)):
            newer = sorted_docs[i]
            older = sorted_docs[i - 1]
            if newer.version_date and older.version_date and newer.version_date != older.version_date:
                repository.set_supersedes(newer.document_id, older.document_id)
                print(f"[ingest] {company}: {newer.file_name} supersedes {older.file_name}")


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[ingest] Configuration error: {exc}", file=sys.stderr)
        return 1

    pdf_paths = []
    extensions = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    
    for directory in (RAW_PDF_DIR, RAW_PDF_DIR.parent / "raw_data"):
        if directory.exists():
            for p in directory.rglob("*"):
                if p.is_file() and p.suffix.lower() in extensions:
                    pdf_paths.append(p)

    pdf_paths = sorted(list(dict.fromkeys(pdf_paths)))

    if not pdf_paths:
        print(f"[ingest] No PDFs found in {RAW_PDF_DIR} or {raw_data_dir}", file=sys.stderr)

    chunker = SlideAwareChunker(
        chunk_size_tokens=cfg.chunk_size_tokens,
        overlap_tokens=cfg.chunk_overlap_tokens,
    )

    print("[ingest] Loading embedding model (downloads ~90 MB on first run)...")
    embedder = Embedder(model=cfg.embedding_model)

    total_inserted = 0
    try:
        with open_connection(cfg.snowflake) as conn:
            repository = ChunkRepository(conn)

            valid_doc_ids = {_document_id(p) for p in pdf_paths}
            for doc in repository.get_all_documents():
                if doc.document_id not in valid_doc_ids:
                    print(f"[ingest] Pruning deleted file from database: {doc.file_name}")
                    repository.delete_document(doc.document_id)

            for path in pdf_paths:
                try:
                    total_inserted += _process_pdf(path, chunker, embedder, repository)
                except PDFParsingError as exc:
                    print(f"[ingest] Skipping {path.name}: {exc}", file=sys.stderr)
                except ChunkingError as exc:
                    print(f"[ingest] Chunking failed for {path.name}: {exc}", file=sys.stderr)
                    continue
                except EmbeddingError as exc:
                    print(f"[ingest] Embedding failed for {path.name}: {exc}", file=sys.stderr)
                    continue
                except RepositoryError as exc:
                    print(f"[ingest] Database write failed for {path.name}: {exc}", file=sys.stderr)
                    continue
                except Exception as exc:
                    print(f"[ingest] Unexpected error for {path.name}: {exc}", file=sys.stderr)
                    continue

            print("[ingest] Setting document version relationships...")
            _set_document_relationships(repository)

    except SnowflakeConnectionError as exc:
        print(f"[ingest] {exc}", file=sys.stderr)
        return 1

    print(f"[ingest] Done. Total chunks inserted: {total_inserted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
