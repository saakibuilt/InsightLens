"""PDF text extraction with page-level metadata and table detection."""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pdfplumber

from insightlens.ingestion.vision_extractor import extract_visual_content

"""@ 2026 Developed by Saksham Nirula"""

_VISUAL_CHAR_THRESHOLD = 120

_VISION_CALL_THRESHOLD = 300

_TITLE_EXCLUDE_RE = re.compile(
    r"^\(?[\d]+\)?[\.\)]?\s*$|^page\s*\d+$", re.IGNORECASE
)

_FOOTNOTE_START_RE = re.compile(
    r"^[\(\*†‡§¶]?\d+[\.\)]\s+\S"
    r"|^\d+\s+[A-Z]"
    r"|^\([a-z]\)\s+\S"
    r"|^[Nn]otes?:?\s+\S"
    r"|^\*\s+\S|^†\s+\S"
)


class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    char_count: int
    is_likely_visual: bool
    slide_title: str | None = None
    tables: tuple = field(default_factory=tuple)  # tuple[tuple[tuple[str]]] — extracted tables
    vision_text: str | None = None               # Claude vision extraction for chart/map/logo pages
    page_image_b64: str | None = None            # Base64 encoded PNG of the page


@dataclass(frozen=True)
class ParsedDocument:
    file_path: Path
    pages: list[ParsedPage]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


def _tag_footnotes(text: str) -> str:
    """Prefix footnote lines with [FOOTNOTE] so retrieval can surface them precisely."""
    lines = text.splitlines()
    if len(lines) < 4:
        return text
    boundary = max(len(lines) * 3 // 4, len(lines) - 5)
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i >= boundary and stripped and _FOOTNOTE_START_RE.match(stripped):
            result.append(f"[FOOTNOTE] {line}")
        else:
            result.append(line)
    return "\n".join(result)


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract text and tables from a PDF, one entry per page."""
    if not path.exists():
        raise PDFParsingError(f"File not found: {path}")

    try:
        document = fitz.open(str(path))
    except fitz.FileDataError as exc:
        raise PDFParsingError(f"Corrupt or unreadable file: {path.name}. Error: {exc}") from exc

    plumber_doc = None
    try:
        if path.suffix.lower() == ".pdf":
            plumber_doc = pdfplumber.open(str(path))
    except Exception:
        pass

    pages: list[ParsedPage] = []
    try:
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            stripped = _tag_footnotes(text.strip())
            slide_title = _extract_slide_title(stripped)

            tables = ()
            if plumber_doc and index < len(plumber_doc.pages):
                try:
                    plumber_page = plumber_doc.pages[index]
                    raw_tables = plumber_page.extract_tables() or []
                    tables = tuple(
                        tuple(tuple(str(cell or "") for cell in row) for row in table if row)
                        for table in raw_tables if table
                    )
                except Exception:
                    pass

                vision_text: str | None = None
                if len(stripped) < _VISION_CALL_THRESHOLD:
                    vision_text = extract_visual_content(page) or None

                effective_text = stripped
                if not effective_text and vision_text:
                    effective_text = vision_text

                pix = page.get_pixmap(dpi=150)
                page_image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

                pages.append(
                    ParsedPage(
                        page_number=index + 1,
                        text=effective_text,
                        char_count=len(effective_text),
                        is_likely_visual=len(stripped) < _VISUAL_CHAR_THRESHOLD,
                        slide_title=slide_title,
                        tables=tables,
                        vision_text=vision_text,
                        page_image_b64=page_image_b64,
                    )
                )
    finally:
        document.close()
        if plumber_doc:
            plumber_doc.close()

    if not any(page.text for page in pages):
        if pages:
            first = pages[0]
            pages[0] = ParsedPage(
                page_number=first.page_number,
                text=f"Scanned Document / Exhibit: {path.name}. (Text is unavailable; requires OCR)",
                char_count=100,
                is_likely_visual=True,
                slide_title=path.name,
                page_image_b64=first.page_image_b64,
            )

    return ParsedDocument(file_path=path, pages=pages)


def _extract_slide_title(text: str) -> str | None:
    """Return the first short line as the likely slide title."""
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped
            and len(stripped) <= 80
            and len(stripped.split()) <= 12
            and not stripped.isdigit()
            and not _TITLE_EXCLUDE_RE.match(stripped)
        ):
            return stripped
    return None
