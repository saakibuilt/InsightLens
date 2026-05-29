"""Persistence operations for documents and chunks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import ProgrammingError

"""@ 2026 Developed by Saksham Nirula"""


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    file_name: str
    company: str | None
    document_type: str | None
    version_label: str | None
    version_date: date | None
    page_count: int
    supersedes_document_id: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    chunk_text: str
    token_count: int
    embedding: Sequence[float]
    section_header: str | None = None
    chunk_type: str = "body"
    structured_content: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    file_name: str
    company: str | None
    version_label: str | None
    page_number: int
    chunk_text: str
    similarity: float
    section_header: str | None = None
    chunk_type: str = "body"
    structured_content: str | None = None
    supersedes_document_id: str | None = None  # non-null → this doc is the newer version
    document_type: str | None = None
    version_date: date | None = None


class RepositoryError(Exception):
    """Raised when a database operation fails."""


class ChunkRepository:
    """Read and write operations for documents and chunks."""

    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def upsert_document(self, doc: DocumentRecord) -> None:
        sql = """
            MERGE INTO DOCUMENTS d
            USING (SELECT %s AS document_id) s
            ON d.document_id = s.document_id
            WHEN MATCHED THEN UPDATE SET
                file_name = %s,
                company = %s,
                document_type = %s,
                version_label = %s,
                version_date = %s,
                page_count = %s,
                supersedes_document_id = %s
            WHEN NOT MATCHED THEN INSERT
                (document_id, file_name, company, document_type, version_label,
                 version_date, page_count, supersedes_document_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            doc.document_id,
            doc.file_name, doc.company, doc.document_type, doc.version_label,
            doc.version_date, doc.page_count, doc.supersedes_document_id,
            doc.document_id, doc.file_name, doc.company, doc.document_type,
            doc.version_label, doc.version_date, doc.page_count, doc.supersedes_document_id,
        )
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, params)
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Failed to upsert document '{doc.file_name}' (id={doc.document_id}): {exc.msg}"
            ) from exc
        finally:
            cursor.close()

    def insert_chunks(self, chunks: Iterable[ChunkRecord]) -> int:
        rows = list(chunks)
        if not rows:
            return 0

        cursor = self._conn.cursor()
        inserted = 0
        try:
            for chunk in rows:
                vec = "[" + ",".join(str(float(v)) for v in chunk.embedding) + "]"
                sql = f"""
                    INSERT INTO CHUNKS
                        (chunk_id, document_id, page_number, chunk_index, chunk_text,
                         token_count, embedding, section_header, chunk_type, structured_content)
                    SELECT %s, %s, %s, %s, %s, %s,
                           {vec}::VECTOR(FLOAT, 384),
                           %s, %s, %s
                """
                cursor.execute(
                    sql,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.chunk_text,
                        chunk.token_count,
                        chunk.section_header,
                        chunk.chunk_type,
                        chunk.structured_content,
                    ),
                )
                inserted += 1
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Failed inserting chunks (succeeded: {inserted}/{len(rows)}). "
                f"Last chunk: {chunk.chunk_id}. Error: {exc.msg}"
            ) from exc
        finally:
            cursor.close()
        return inserted

    def delete_document(self, document_id: str) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute("DELETE FROM CHUNKS WHERE document_id = %s", (document_id,))
            cursor.execute("DELETE FROM DOCUMENTS WHERE document_id = %s", (document_id,))
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Failed to delete document {document_id}: {exc.msg}"
            ) from exc
        finally:
            cursor.close()

    def clear_all(self) -> None:
        """Delete all documents and chunks from the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("TRUNCATE TABLE CHUNKS")
            cursor.execute("TRUNCATE TABLE DOCUMENTS")
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Failed to clear database: {exc.msg}"
            ) from exc
        finally:
            cursor.close()

    def search_similar(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        company_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        vec = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
        base_sql = f"""
            SELECT
                c.chunk_id,
                c.document_id,
                d.file_name,
                d.company,
                d.version_label,
                c.page_number,
                c.chunk_text,
                VECTOR_COSINE_SIMILARITY(c.embedding, {vec}::VECTOR(FLOAT, 384)) AS similarity,
                c.section_header,
                c.chunk_type,
                c.structured_content,
                d.supersedes_document_id,
                d.document_type,
                d.version_date
            FROM CHUNKS c
            JOIN DOCUMENTS d ON c.document_id = d.document_id
        """
        params: list = []
        if company_filter:
            base_sql += " WHERE UPPER(d.company) = UPPER(%s)"
            params.append(company_filter)
        base_sql += " ORDER BY similarity DESC LIMIT %s"
        params.append(top_k)

        cursor = self._conn.cursor()
        try:
            cursor.execute(base_sql, tuple(params))
            rows = cursor.fetchall()
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Vector search failed (top_k={top_k}, company_filter={company_filter}): {exc.msg}"
            ) from exc
        finally:
            cursor.close()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                file_name=row[2],
                company=row[3],
                version_label=row[4],
                page_number=row[5],
                chunk_text=row[6],
                similarity=float(row[7]),
                section_header=row[8],
                chunk_type=row[9] or "body",
                structured_content=row[10],
                supersedes_document_id=row[11],
                document_type=row[12],
                version_date=row[13],
            )
            for row in rows
        ]

    def get_all_chunks(self) -> list[RetrievedChunk]:
        """Load every chunk (no embeddings) for BM25 corpus construction."""
        sql = """
            SELECT
                c.chunk_id,
                c.document_id,
                d.file_name,
                d.company,
                d.version_label,
                c.page_number,
                c.chunk_text,
                c.section_header,
                c.chunk_type,
                c.structured_content,
                d.supersedes_document_id,
                d.document_type,
                d.version_date
            FROM CHUNKS c
            JOIN DOCUMENTS d ON c.document_id = d.document_id
            ORDER BY c.chunk_id
        """
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        except ProgrammingError as exc:
            raise RepositoryError(f"Failed to load BM25 corpus: {exc.msg}") from exc
        finally:
            cursor.close()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                file_name=row[2],
                company=row[3],
                version_label=row[4],
                page_number=row[5],
                chunk_text=row[6],
                similarity=0.0,
                section_header=row[7],
                chunk_type=row[8] or "body",
                structured_content=row[9],
                supersedes_document_id=row[10],
                document_type=row[11],
                version_date=row[12],
            )
            for row in rows
        ]

    def get_all_documents(self) -> list[DocumentRecord]:
        """Return all documents — used to compute version supersession relationships."""
        sql = """
            SELECT document_id, file_name, company, document_type,
                   version_label, version_date, page_count, supersedes_document_id
            FROM DOCUMENTS
            ORDER BY company, version_date
        """
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        except ProgrammingError as exc:
            raise RepositoryError(f"Failed to list documents: {exc.msg}") from exc
        finally:
            cursor.close()

        return [
            DocumentRecord(
                document_id=row[0],
                file_name=row[1],
                company=row[2],
                document_type=row[3],
                version_label=row[4],
                version_date=row[5],
                page_count=row[6],
                supersedes_document_id=row[7],
            )
            for row in rows
        ]

    def set_supersedes(self, newer_document_id: str, older_document_id: str) -> None:
        """Mark newer_document_id as superseding older_document_id."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                "UPDATE DOCUMENTS SET supersedes_document_id = %s WHERE document_id = %s",
                (older_document_id, newer_document_id),
            )
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Failed to set supersedes relationship: {exc.msg}"
            ) from exc
        finally:
            cursor.close()

    def list_companies(self) -> list[str]:
        cursor = self._conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT company FROM DOCUMENTS WHERE company IS NOT NULL ORDER BY company")
            return [row[0] for row in cursor.fetchall()]
        except ProgrammingError as exc:
            raise RepositoryError(f"Failed to list companies: {exc.msg}") from exc
        finally:
            cursor.close()
