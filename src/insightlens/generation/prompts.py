"""Prompt templates for answer generation."""
from __future__ import annotations

from datetime import date

from insightlens.storage.chunk_repository import RetrievedChunk

"""@ 2026 Developed by Saksham Nirula"""

SYSTEM_PROMPT = """You are a legal and investigative assistant that answers questions about case files and legal intelligence.

Rules you must follow:
1. Ground every factual claim in the provided sources. If the sources do not contain the answer, say so plainly — do not use training-data knowledge to fill gaps.
2. When sources from different documents or testimonies disagree on a fact or event, present both accounts separately with attribution. Never silently merge conflicting accounts.
3. When a source document is labeled STALE SOURCE or indicates it is an old historical file, explicitly flag the age of the data.
4. Cite sources inline using the format [Source N]. Each source corresponds to one entry in the source list.
5. Keep answers concise. Lead with the direct answer; supporting detail follows.
6. Ensure that you maintain neutrality and objectively state what the documents claim, attribute statements to the individuals making them, and do not present allegations as absolute facts unless the document explicitly establishes them as such.
7. If the user input is a person's name or a short keyword phrase, provide a comprehensive summary of what the sources say about that subject."""


_STALE_YEARS = 2


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "No source material was retrieved for this question. "
            "Reply that the corpus does not appear to contain information relevant to the question."
        )

    today = date.today()

    superseded_ids: set[str] = {
        chunk.supersedes_document_id
        for chunk in chunks
        if chunk.supersedes_document_id
    }

    source_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        company = chunk.company or "unknown subject"
        version = chunk.version_label or "unversioned"
        doc_type = chunk.document_type or "document"

        if chunk.supersedes_document_id:
            version_note = "CURRENT VERSION (supersedes an earlier version)"
        elif chunk.document_id in superseded_ids:
            version_note = "HISTORICAL VERSION (superseded by a more recent document in these sources)"
        else:
            version_note = f"version: {version}"

        stale_note = ""
        if chunk.version_date:
            age_years = (today - chunk.version_date).days / 365
            if age_years > _STALE_YEARS:
                stale_note = f" ⚠ STALE SOURCE (published {chunk.version_date.year}, data may be outdated)"

        slide = f", slide: {chunk.section_header}" if chunk.section_header else ""
        header = (
            f"[Source {index}] {chunk.file_name} "
            f"(subject: {company}, type: {doc_type}, {version_note}, "
            f"page: {chunk.page_number}{slide}{stale_note})"
        )
        source_blocks.append(f"{header}\n{chunk.chunk_text}")

    sources_text = "\n\n".join(source_blocks)
    return (
        f"Question: {question}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Provide an answer that follows the rules. Use [Source N] inline citations. If the question is just a short phrase or name, summarize the key information about it."
    )
