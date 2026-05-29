"""Heuristic detection of company, document type, and version from filenames and content."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DocumentMetadata:
    company: str | None
    document_type: str | None
    version_label: str | None
    version_date: date | None


_VERSION_PATTERNS = [
    re.compile(r"\bv(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bversion[\s_-]?(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})[\s_-]?(Q[1-4])\b", re.IGNORECASE),
    re.compile(r"\b(Q[1-4])[\s_-]?(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})\b"),
]

_DATE_PATTERN = re.compile(r"(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})")

_DOC_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("Court Document",        ["court", "case", "plaintiff", "defendant", "subpoena", "lawsuit", "complaint", "affidavit", "deposition", "exhibit"]),
    ("Merger Presentation",   ["merger"]),
    ("Investor Day",          ["morning session", "investor day"]),
    ("Q4 Update",             ["q4 2025", "q4-2025", "q4_2025", "q4 2024", "q4-2024"]),
    ("Quarterly Update",      ["10-q", "10q", "quarterly", "q1", "q2", "q3"]),
    ("Roadshow",              ["roadshow"]),
    ("Company Update",        ["company update", "company-update"]),
    ("Annual Report",         ["annual", "10-k", "10k"]),
    ("Third-Party Report",    ["impact of", "the impact", "brick and mortar"]),
    ("Research Report",       ["report", "analysis", "research"]),
    ("Investor Presentation", ["investor", "presentation", "deck", "pitch"]),
]


def extract_metadata(file_path: Path, first_page_text: str) -> DocumentMetadata:
    """Combine filename signals and first-page content to infer metadata."""
    stem = file_path.stem
    haystack = f"{stem} {first_page_text[:2000]}"

    return DocumentMetadata(
        company=_detect_company(stem, first_page_text),
        document_type=_detect_doc_type(haystack),
        version_label=_detect_version_label(stem),
        version_date=_detect_date(haystack),
    )


_DOC_TITLE_WORDS = {"company", "merger", "session", "morning", "impact", "the", "q1", "q2", "q3", "q4"}


def _detect_company(stem: str, first_page_text: str) -> str | None:
    underscore_pos = stem.find("_")
    hyphen_pos = stem.find("-")

    if underscore_pos == -1 and hyphen_pos == -1:
        candidate = stem.split()[0].strip() if stem else ""
    elif hyphen_pos != -1 and (underscore_pos == -1 or hyphen_pos < underscore_pos):
        candidate = stem[:hyphen_pos].strip()
    else:
        candidate = stem[:underscore_pos].strip()

    words = candidate.split()
    if len(words) > 2:
        candidate = words[0]
    elif len(words) == 2 and words[1].lower() in _DOC_TITLE_WORDS:
        candidate = words[0]

    if candidate and len(candidate) >= 2 and not candidate.isdigit():
        return candidate.title()

    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    for line in lines[:5]:
        if 2 <= len(line.split()) <= 6 and line[0].isupper():
            return line
    return None


def _detect_doc_type(haystack: str) -> str | None:
    lowered = haystack.lower()
    for label, keywords in _DOC_TYPE_PATTERNS:
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def _detect_version_label(stem: str) -> str | None:
    normalized = stem.replace("_", " ").replace("-", " ")
    for pattern in _VERSION_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return "_".join(group for group in match.groups() if group)
    return None


def _detect_date(haystack: str) -> date | None:
    match = _DATE_PATTERN.search(haystack)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None
