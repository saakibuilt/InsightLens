"""InsightLens — chat-style UI for investment document Q&A."""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Cloud runs `pip install -r requirements.txt` but NOT `pip install -e .`,
# so the local `insightlens` package isn't on sys.path by default.
# Insert src/ so imports work both locally and in the cloud.
sys.path.insert(0, str(Path(__file__).parents[2]))

import html
import json
import os
import re

import pandas as pd
import streamlit as st

# set_page_config must be the very first Streamlit call in the script.
st.set_page_config(page_title="InsightLens", layout="wide", page_icon="🔍")

# On Streamlit Cloud there is no .env file — secrets come from the dashboard.
# Locally, python-dotenv handles everything, so skip st.secrets to avoid the
# "No secrets found" warning that appears when no secrets.toml is present.
_env_file = Path(__file__).parents[3] / ".env"
if not _env_file.exists():
    try:
        for _k, _v in st.secrets.items():
            if isinstance(_v, str):
                os.environ.setdefault(_k, _v)
    except Exception:
        pass

from insightlens.config import ConfigError, load_config
from insightlens.embeddings.embedder import Embedder
from insightlens.generation.llm_client import ClaudeClient
from insightlens.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from insightlens.retrieval.hybrid_search import HybridSearchService
from insightlens.retrieval.reranker import Reranker
from insightlens.retrieval.vector_search import RetrievalRequest
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk
from insightlens.storage.snowflake_client import open_connection

"""@ 2026 Developed by Saksham Nirula"""
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Startup (cached — runs once per session) ───────────────────────────────────
@st.cache_resource
def _bootstrap():
    cfg = load_config()
    embedder = Embedder(model=cfg.embedding_model)
    llm = ClaudeClient(api_key=cfg.anthropic_api_key, model=cfg.generation_model)
    reranker = Reranker()
    return cfg, embedder, llm, reranker

@st.cache_data(ttl=300)
def _load_companies(_cfg):
    with open_connection(_cfg.snowflake) as conn:
        return ChunkRepository(conn).list_companies()

@st.cache_data(ttl=600)
def _load_corpus(_cfg):
    """Load all chunks for BM25 index construction. Refreshes every 10 minutes."""
    with open_connection(_cfg.snowflake) as conn:
        return ChunkRepository(conn).get_all_chunks()

try:
    cfg, embedder, llm, reranker = _bootstrap()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

try:
    companies = _load_companies(cfg)
except Exception as exc:
    st.error(f"Could not connect to Snowflake: {exc}")
    st.info("Check that your secrets are configured in the Streamlit Cloud dashboard.")
    st.stop()

with st.sidebar:
    st.html(
        "<div style='padding:4px 0 12px'>"
        "<span style='font-size:1.25rem;font-weight:700;color:#1e293b'>InsightLens</span>"
        "<br><span style='font-size:0.78rem;color:#64748b'>Investment Document Intelligence</span>"
        "</div>"
    )
    st.divider()
    company_choice = st.selectbox(
        "Company",
        options=["All companies"] + companies,
        help="Filter results to a single company's documents",
    )
    top_k = st.slider(
        "Sources per answer",
        min_value=3, max_value=15, value=max(cfg.retrieval_top_k, 8),
        help="How many source chunks are retrieved and cited",
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def _render_text(text: str) -> None:
    """Render body text: gradient-fade scroll for long passages."""
    escaped = html.escape(text)
    if len(text) > 800:
        st.html(
            "<div style='position:relative'>"
            f"<div style='max-height:220px;overflow-y:auto;font-size:0.875rem;"
            f"line-height:1.65;color:inherit;white-space:pre-wrap'>{escaped}</div>"
            "<div style='position:absolute;bottom:18px;left:0;right:0;height:36px;"
            "background:linear-gradient(to bottom,transparent,rgba(255,255,255,0.95));"
            "pointer-events:none'></div>"
            "<div style='text-align:center;font-size:0.7rem;color:#94a3b8'>"
            "↕ scroll for full text</div>"
            "</div>"
        )
    else:
        st.html(
            f"<div style='font-size:0.875rem;line-height:1.65;color:inherit;"
            f"white-space:pre-wrap'>{escaped}</div>"
        )


# ── Financial-body auto-detection ─────────────────────────────────────────────
_FIN_SIGNALS = re.compile(r"\$\s*[\d,]+|\(\s*[\d,]+\s*\)")
_FIN_VALUE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)|\(\s*([\d,]+(?:\.\d+)?)\s*\)|(—|-{1,2})")
_PERIOD_HEADER_RE = re.compile(
    r"(?:Q[1-4]\s+\d{4}|(?:December|March|June|September|October|January|April|July)"
    r"\s+\d{1,2},?\s+\d{4}|\d{4})",
    re.IGNORECASE,
)


def _try_render_financial_table_from_body(text: str) -> bool:
    """Parse financial-statement rows from raw body text and render as a table.

    Handles patterns like:
      FFO attributable to...   $311,007   $307,347   $301,769   $289,513
      Straight-line rent        (25,710)   (30,105)   (24,533)   (30,968)
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    # Collect data rows: label + ≥2 monetary values
    data_rows: list[list[str]] = []
    for line in lines:
        first_val = _FIN_SIGNALS.search(line)
        if not first_val:
            continue
        label = line[: first_val.start()].strip()
        if not label or len(label) < 3:
            continue
        rest = line[first_val.start():]
        vals: list[str] = []
        for m in _FIN_VALUE_RE.finditer(rest):
            if m.group(1):
                vals.append(f"${m.group(1)}")
            elif m.group(2):
                vals.append(f"({m.group(2)})")
            else:
                vals.append("—")
        if len(vals) >= 2:
            data_rows.append([label] + vals)

    if len(data_rows) < 3:
        return False

    # Normalise column count
    max_cols = max(len(r) for r in data_rows)
    data_rows = [(r + [""] * max_cols)[:max_cols] for r in data_rows]

    # Try to extract period labels from a header line near the top
    periods: list[str] = []
    for line in lines[:8]:
        if _FIN_SIGNALS.search(line):
            continue
        found = _PERIOD_HEADER_RE.findall(line)
        if found:
            periods = found
            break

    n_val = max_cols - 1
    col_headers = (periods[:n_val] if len(periods) >= n_val
                   else [f"Period {i + 1}" for i in range(n_val)])
    header = (["Metric"] + col_headers + [""] * max_cols)[:max_cols]

    # Render
    _th0 = ("border:1px solid #cbd5e1;padding:5px 10px;text-align:left;"
            "background:#f1f5f9;font-weight:600;font-size:0.75rem")
    _th  = ("border:1px solid #cbd5e1;padding:5px 10px;text-align:right;"
            "background:#f1f5f9;font-weight:600;font-size:0.75rem;white-space:nowrap")
    _td0 = "border:1px solid #e2e8f0;padding:4px 10px;text-align:left;font-size:0.78rem"
    _td  = ("border:1px solid #e2e8f0;padding:4px 10px;text-align:right;"
            "font-size:0.78rem;font-family:monospace")

    h_html = (
        f"<th style='{_th0}'>{html.escape(header[0])}</th>"
        + "".join(f"<th style='{_th}'>{html.escape(h)}</th>" for h in header[1:])
    )
    b_html = "".join(
        "<tr>"
        + f"<td style='{_td0}'>{html.escape(row[0])}</td>"
        + "".join(f"<td style='{_td}'>{html.escape(v)}</td>" for v in row[1:])
        + "</tr>"
        for row in data_rows
    )
    st.html(
        "<div style='overflow-x:auto;margin-top:4px'>"
        "<div style='font-size:0.7rem;font-weight:600;color:#64748b;margin-bottom:5px'>"
        "📋 FINANCIAL STATEMENT — auto-parsed table"
        "</div>"
        "<table style='border-collapse:collapse;width:100%'>"
        f"<thead><tr>{h_html}</tr></thead>"
        f"<tbody>{b_html}</tbody>"
        "</table></div>"
    )
    return True


def _render_body_content(text: str) -> None:
    """Render a body chunk as a structured artifact:
    title + 3-line preview + expandable full text.
    Financial reconciliation tables are auto-parsed into a real table.
    """
    # Auto-detect financial statement content (≥5 dollar/paren amounts)
    if len(_FIN_SIGNALS.findall(text)) >= 5:
        if _try_render_financial_table_from_body(text):
            return

    # Smart preview: title line + 2-3 excerpt lines + HTML <details> for the rest
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return

    title = lines[0]
    excerpt_lines = lines[1:4]
    has_more = len(lines) > 4 or sum(len(l) for l in lines[1:]) > 300

    title_html = (
        f"<div style='font-weight:600;font-size:0.875rem;color:#1e293b;"
        f"margin-bottom:5px;line-height:1.4'>{html.escape(title)}</div>"
    )
    excerpt_html = ""
    if excerpt_lines:
        excerpt = " ".join(excerpt_lines)[:300]
        if has_more:
            excerpt += "…"
        excerpt_html = (
            f"<div style='font-size:0.82rem;color:#475569;line-height:1.55;"
            f"margin-bottom:6px'>{html.escape(excerpt)}</div>"
        )

    if has_more:
        full_escaped = html.escape(text)
        expand_html = (
            "<details style='margin-top:2px'>"
            "<summary style='cursor:pointer;font-size:0.75rem;color:#4a7fcb;"
            "font-weight:500;user-select:none'>▶ Show full text</summary>"
            f"<div style='margin-top:8px;max-height:280px;overflow-y:auto;"
            f"font-size:0.8rem;line-height:1.6;white-space:pre-wrap;"
            f"color:inherit;padding-right:4px'>{full_escaped}</div>"
            "</details>"
        )
    else:
        full_escaped = html.escape(text)
        expand_html = (
            f"<div style='font-size:0.82rem;line-height:1.6;white-space:pre-wrap;"
            f"color:inherit'>{full_escaped}</div>"
        )

    st.html(title_html + excerpt_html + expand_html)


# ── Chart-line classification patterns ────────────────────────────────────────
# Require $ prefix OR unit suffix OR decimal — bare integers (page numbers
# like "14", "5") must NOT be treated as financial values.
_MONEY_RE = re.compile(
    r"(?:"
    r"[~\$]+[\d,\.]+\s*(?:bn|mn|mm|m|b|billion|million|trillion|%|×|x|k|bps)?"  # $X or $Xbn
    r"|[\d,\.]*\d\.\d[\d,\.]*\s*(?:bn|mn|mm|m|b|billion|million|trillion|%|×|x|k|bps)?"  # X.Y decimal
    r"|[\d,\.]+\s*(?:bn|mn|mm|m|b|billion|million|trillion|%|×|x|k|bps)"  # Xbn / X% (unit required)
    r")\s*$",
    re.IGNORECASE,
)
# Year-row patterns — "Q1 2024 Q2 2024" or "2021 2022 2023"
_YEAR_ROW_RE = re.compile(
    r"^(?:Q[1-4]\s+)?\d{4}(?:[\s\-–·]+(?:Q[1-4]\s+)?\d{4})+$"
)
_YEAR_SINGLE_RE = re.compile(r"^\d{4}$")
# Short lines (≤70 chars) containing an embedded dollar value
_INLINE_VALUE_RE = re.compile(
    r"^.{3,70}[~\$]\d[\d,\.]*\s*(?:bn|mn|mm|m|b|billion|million|trillion|%|bps)?.{0,20}$",
    re.IGNORECASE,
)
_FOOTNOTE_RE = re.compile(r"^\(?\d+\)?[\.\)]")


def _classify_chart_lines(
    text: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split chart text into (titles, values, year_rows, notes)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    titles, values, year_rows, notes = [], [], [], []
    seen_value = False
    for line in lines:
        if _FOOTNOTE_RE.match(line):
            continue
        if _MONEY_RE.match(line):
            values.append(line)
            seen_value = True
        elif _YEAR_ROW_RE.match(line) or _YEAR_SINGLE_RE.match(line):
            year_rows.append(line)
        elif not seen_value and _INLINE_VALUE_RE.match(line):
            values.append(line)
            seen_value = True
        elif not seen_value and len(titles) < 4:
            titles.append(line)
        else:
            notes.append(line)
    return titles, values, year_rows, notes


def _render_chart_text(text: str) -> None:
    """Render chart/infographic text as a structured info card."""
    titles, values, year_rows, notes = _classify_chart_lines(text)

    def e(s: str) -> str:
        return html.escape(s)

    title_html = ""
    if titles:
        title_html = (
            "<div style='font-size:0.9rem;font-weight:600;margin-bottom:8px;line-height:1.4'>"
            + "<br>".join(e(t) for t in titles)
            + "</div>"
        )

    values_html = ""
    if values:
        chips = "".join(
            f"<span style='display:inline-block;background:#dbeafe;color:#1e40af;"
            f"border-radius:4px;padding:2px 8px;margin:2px 3px;"
            f"font-weight:600;font-size:0.85rem'>{e(v)}</span>"
            for v in values
        )
        values_html = (
            "<div style='margin-bottom:6px'>"
            "<span style='font-size:0.72rem;color:#64748b;font-weight:600;"
            "letter-spacing:0.04em;display:block;margin-bottom:3px'>DATA POINTS</span>"
            + chips + "</div>"
        )

    # FIX 8 & 9: extract "Q1 2024" or plain "2024" tokens from year rows
    years_html = ""
    if year_rows:
        year_tags = ""
        for row in year_rows:
            tokens = re.findall(r"Q[1-4]\s+\d{4}|\d{4}", row)
            for tok in tokens:
                year_tags += (
                    f"<span style='display:inline-block;background:#f1f5f9;color:#475569;"
                    f"border-radius:3px;padding:1px 6px;margin:1px 2px;"
                    f"font-size:0.78rem;font-family:monospace'>{e(tok)}</span>"
                )
        years_html = (
            "<div style='margin-top:6px'>"
            "<span style='font-size:0.72rem;color:#64748b;font-weight:600;"
            "letter-spacing:0.04em;display:block;margin-bottom:3px'>PERIOD</span>"
            + year_tags + "</div>"
        )

    notes_html = ""
    if notes:
        notes_html = (
            "<div style='margin-top:8px;font-size:0.78rem;color:#64748b;"
            "line-height:1.5;white-space:pre-wrap'>"
            + "<br>".join(e(n) for n in notes)
            + "</div>"
        )

    if not title_html and not values_html and not years_html:
        body_html = (
            f"<div style='font-size:0.85rem;white-space:pre-wrap;"
            f"font-family:monospace;line-height:1.8'>{e(text)}</div>"
        )
    else:
        body_html = title_html + values_html + years_html + notes_html

    st.html(
        "<div style='background:#f0f6ff;border-left:3px solid #4a7fcb;"
        "border-radius:4px;padding:10px 14px;margin-top:4px'>"
        "<div style='font-size:0.7rem;font-weight:600;color:#4a7fcb;"
        "letter-spacing:0.05em;margin-bottom:8px'>"
        "📈 CHART / VISUAL SLIDE — data extracted from PDF; bar-to-year mapping is visual"
        "</div>"
        + body_html + "</div>"
    )


# ── Table validation ──────────────────────────────────────────────────────────

def _is_genuine_financial_table(rows: list) -> bool:
    """Return True only when structured_content looks like real financial data.

    Rejects:
    - TOC pages: cells are text strings + short page-number ints (1-3 digits)
    - Text slides: cells contain paragraph-length strings (>100 chars)
    - Mostly-empty tables: <25% of data cells contain any number
    """
    if not rows or len(rows) < 2:
        return False
    header, *data = rows
    if len(header) < 2 or not data:
        return False

    # Reject if ANY cell is a paragraph (>100 chars) — that's a text slide
    for row in rows[:6]:
        for cell in row:
            if len(str(cell).strip()) > 100:
                return False

    # Collect all non-empty data cells
    data_cells = [str(c).strip() for row in data for c in row if str(c).strip()]
    if not data_cells:
        return False

    # Reject TOC pattern: most cells are 1-3 digit integers (page numbers)
    short_ints = sum(1 for c in data_cells if c.isdigit() and len(c) <= 3)
    if short_ints / len(data_cells) > 0.40:
        return False

    # Must have meaningful numeric content ($, %, decimal, comma-separated)
    numeric = sum(1 for c in data_cells if re.search(r"[\$\%]|[\d,]{4,}|\d\.\d", c))
    if numeric / len(data_cells) < 0.20:
        return False

    return True


def _try_render_table(structured_content: str) -> bool:
    """Render structured_content as a pandas DataFrame (strict: genuine tables only)."""
    try:
        rows = json.loads(structured_content)
        if not _is_genuine_financial_table(rows):
            return False
        header, *data = rows
        safe_header = [str(h)[:120] for h in header]
        n = len(safe_header)
        normalized = [(list(row) + [""] * n)[:n] for row in data]
        df = pd.DataFrame(normalized, columns=safe_header)
        st.dataframe(df, use_container_width=True)
        return True
    except Exception:
        return False


def _render_structured_as_html_table(structured_content: str) -> bool:
    """HTML-table fallback for genuine financial tables that pandas can't handle."""
    try:
        rows = json.loads(structured_content)
        if not _is_genuine_financial_table(rows):
            return False
        header, *data = rows
        n = len(header)
        _th = ("border:1px solid #cbd5e1;padding:5px 10px;text-align:left;"
               "background:#f1f5f9;font-weight:600")
        _td = "border:1px solid #e2e8f0;padding:5px 10px;text-align:left"
        h_html = "<tr>" + "".join(
            f"<th style='{_th}'>{html.escape(str(h))}</th>" for h in header
        ) + "</tr>"
        b_html = "".join(
            "<tr>" + "".join(
                f"<td style='{_td}'>{html.escape(str(v))}</td>"
                for v in (list(row) + [""] * n)[:n]
            ) + "</tr>"
            for row in data
        )
        st.html(
            "<div style='overflow-x:auto;margin-top:4px'>"
            "<table style='border-collapse:collapse;font-size:0.82rem;width:100%'>"
            f"<thead>{h_html}</thead><tbody>{b_html}</tbody>"
            "</table></div>"
        )
        return True
    except Exception:
        return False


def _try_render_pipe_table(text: str) -> bool:
    """Render pipe-delimited text as an HTML table."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and "|" in ln]
    if len(lines) < 2:
        return False
    rows = [[c.strip() for c in ln.split("|")] for ln in lines]
    max_cols = max(len(r) for r in rows)
    if max_cols < 2:
        return False
    rows = [(r + [""] * max_cols)[:max_cols] for r in rows]
    header, *data = rows
    if not data:
        return False
    if not _is_genuine_financial_table([header] + data):
        return False
    _th = ("border:1px solid #cbd5e1;padding:5px 10px;text-align:left;"
           "background:#f1f5f9;font-weight:600")
    _td = "border:1px solid #e2e8f0;padding:5px 10px;text-align:left"
    h_html = "<tr>" + "".join(f"<th style='{_th}'>{html.escape(h)}</th>" for h in header) + "</tr>"
    b_html = "".join(
        "<tr>" + "".join(f"<td style='{_td}'>{html.escape(v)}</td>" for v in row) + "</tr>"
        for row in data
    )
    st.html(
        "<div style='overflow-x:auto;margin-top:4px'>"
        "<table style='border-collapse:collapse;font-size:0.82rem;width:100%'>"
        f"<thead>{h_html}</thead><tbody>{b_html}</tbody>"
        "</table></div>"
    )
    return True



def _resolve_content_kind(chunk: RetrievedChunk) -> str:
    """Classify what a chunk will actually render as — drives badge and dispatch.

    Returns one of: "table" | "statement" | "chart" | "text" | "reference"
    """
    if chunk.chunk_type == "financial_table" and chunk.structured_content:
        try:
            rows = json.loads(chunk.structured_content)
            if _is_genuine_financial_table(rows):
                return "table"
        except Exception:
            pass

    if chunk.chunk_type == "chart_caption":
        return "chart"

    if chunk.chunk_type == "financial_table":
        if len(_FIN_SIGNALS.findall(chunk.chunk_text)) >= 4:
            return "statement"
        _, vals, years, _ = _classify_chart_lines(chunk.chunk_text)
        if vals or years:
            return "chart"

    if chunk.chunk_type == "body":
        if len(_FIN_SIGNALS.findall(chunk.chunk_text)) >= 5:
            return "statement"

    return "text"


_BADGE = {
    "table":     "📊 ",
    "statement": "📋 ",
    "chart":     "📈 ",
    "text":      "",
    "reference": "",
}

_KIND_CHIP = {
    "table":     ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#166534;background:#dcfce7;border-radius:3px;padding:1px 6px'>"
                  "FINANCIAL TABLE</span>"),
    "statement": ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#1e40af;background:#dbeafe;border-radius:3px;padding:1px 6px'>"
                  "FINANCIAL STATEMENT</span>"),
    "chart":     ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#92400e;background:#fef3c7;border-radius:3px;padding:1px 6px'>"
                  "CHART / VISUAL</span>"),
    "text":      ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#374151;background:#f3f4f6;border-radius:3px;padding:1px 6px'>"
                  "TEXT</span>"),
    "reference": ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#6b7280;background:#f9fafb;border-radius:3px;padding:1px 6px'>"
                  "REFERENCE / INDEX</span>"),
}



def _render_chunk_content(chunk: RetrievedChunk, kind: str) -> None:
    """Render the content artifact for a source card.

    Architecture:
      table      → st.dataframe / bordered HTML table (genuine financial data)
      statement  → auto-parsed rows: Metric | Q1 | Q2 | Q3 | Q4
      chart      → title + value badges + period tags (bar/line charts, maps)
      text       → title + excerpt preview + ▶ expand (narrative slides)
      reference  → same as text but clearly labelled (TOC, definitions, appendix)
    """
    if kind == "table":
        if chunk.structured_content:
            if _try_render_table(chunk.structured_content):
                return
            if _render_structured_as_html_table(chunk.structured_content):
                return
        if _try_render_pipe_table(chunk.chunk_text):
            return
        kind = "statement"

    if kind == "statement":
        if _try_render_financial_table_from_body(chunk.chunk_text):
            return
        kind = "text"

    if kind == "chart":
        _render_chart_text(chunk.chunk_text)
        return

    _render_body_content(chunk.chunk_text)


def _render_sources(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        return
    st.markdown("---")
    st.markdown("#### Sources")
    for i, chunk in enumerate(chunks, start=1):
        kind    = _resolve_content_kind(chunk)
        badge   = _BADGE[kind]
        company = (chunk.company or "Unknown").upper()
        pct     = "—" if chunk.similarity == 0.0 else f"{chunk.similarity:.0%}"

        label = f"{badge}Source {i}  ·  {company}  ·  p.{chunk.page_number}  ·  {pct} match"

        with st.expander(label):
            fname     = chunk.file_name if len(chunk.file_name) <= 60 else chunk.file_name[:57] + "…"
            section   = chunk.section_header or ""
            version   = chunk.version_label or ""
            meta_parts = [f"📄 {html.escape(fname)}"]
            if section:
                meta_parts.append(f"Section: <em>{html.escape(section)}</em>")
            if version:
                meta_parts.append(f"Version: {html.escape(version)}")
            meta_parts.append(f"Page {chunk.page_number}")
            if chunk.similarity > 0.0:
                meta_parts.append(
                    f"<span style='color:#16a34a;font-weight:600'>{pct} match</span>"
                )

            st.html(
                "<div style='font-size:0.75rem;color:#64748b;line-height:1.8;"
                "border-bottom:1px solid #e2e8f0;padding-bottom:8px;margin-bottom:8px'>"
                + "  &nbsp;·&nbsp;  ".join(meta_parts)
                + "</div>"
            )

            st.html(_KIND_CHIP[kind] + "<div style='margin-bottom:8px'></div>")
            _render_chunk_content(chunk, kind)

if not st.session_state.messages:
    st.markdown("<div style='height:20vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.html(
            "<div style='text-align:center;padding:32px 24px;background:#f8fafc;"
            "border:1px solid #e2e8f0;border-radius:12px'>"
            "<div style='font-size:1.6rem;font-weight:700;color:#1e293b;"
            "letter-spacing:-0.5px;margin-bottom:6px'>InsightLens</div>"
            "<div style='font-size:0.9rem;color:#64748b;margin-bottom:20px'>"
            "Investment Document Intelligence</div>"
            "<div style='font-size:0.82rem;color:#94a3b8;line-height:1.8'>"
            "Ask about financials, operating metrics, strategy,<br>"
            "or cross-company comparisons across your uploaded documents."
            "</div>"
            "</div>"
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            _render_sources(msg["chunks"])

question = st.chat_input("Ask about your investment documents…")

_FOLLOWUP_RE = re.compile(
    r"\b(it|its|this|that|those|these|they|them|their|he|she|his|her|"
    r"the\s+company|the\s+fund|the\s+reit|the\s+same)\b",
    re.IGNORECASE,
)


def _contextualize_query(raw: str, history: list[dict]) -> str:
    """Rewrite a follow-up question so retrieval has full context.

    If the current query is short (<= 12 words) OR contains a pronoun / vague
    reference, prepend the previous user question as context so the embedder
    and BM25 index can match the right documents.

    Example:
      Previous: "What are VICI's key operating metrics?"
      Current:  "How does that compare to last quarter?"
      Retrieval query: "Context: What are VICI's key operating metrics?
                        Question: How does that compare to last quarter?"
    """
    user_turns = [m for m in history if m["role"] == "user"]
    if not user_turns:
        return raw

    words = raw.split()
    is_short = len(words) <= 12
    has_reference = bool(_FOLLOWUP_RE.search(raw))

    if not (is_short or has_reference):
        return raw  # self-contained question — leave unchanged

    prev_question = user_turns[-1]["content"]
    return f"Context: {prev_question}\nQuestion: {raw}"


_COMPANY_ALIASES: dict[str, str] = {
    "realty income":  "Realty Income",
    "digital realty": "Digital",
    "public storage": "Psa",
    "eastgroup":      "Egp",
    "east group":     "Egp",
    "simon property": "Simon",
    "simon":          "Simon",
    "bxp":            "Bxp",
    "vici":           "Vici",
    "dlr":            "Digital",
}


def _ensure_company_coverage(
    query: str,
    chunks: list[RetrievedChunk],
    retrieval: HybridSearchService,
) -> list[RetrievedChunk]:
    """Guarantee every company mentioned in a cross-company query has ≥1 chunk.

    After primary retrieval, if a company named in the query has zero chunks in
    the result set, run a targeted secondary retrieval (top_k=2, company_filter
    set to that company) and append those chunks so the LLM can address it.
    """
    if not companies:
        return chunks

    represented = {(c.company or "").lower() for c in chunks}
    query_lower = query.lower()

    company_lookup = {c.lower(): c for c in companies}

    missing: list[str] = []
    seen_missing: set[str] = set()

    for stored_lower, stored_name in company_lookup.items():
        if stored_lower in query_lower and stored_lower not in represented:
            if stored_lower not in seen_missing:
                missing.append(stored_name)
                seen_missing.add(stored_lower)

    for alias, canonical in _COMPANY_ALIASES.items():
        if alias in query_lower:
            actual = company_lookup.get(canonical.lower(), canonical)
            if actual.lower() not in represented and actual.lower() not in seen_missing:
                missing.append(actual)
                seen_missing.add(actual.lower())

    if not missing:
        return chunks

    augmented = list(chunks)
    seen_ids = {c.chunk_id for c in chunks}
    for company_name in missing:
        try:
            secondary = retrieval.retrieve(
                RetrievalRequest(query=query, top_k=2, company_filter=company_name)
            )
            for c in secondary:
                if c.chunk_id not in seen_ids:
                    augmented.append(c)
                    seen_ids.add(c.chunk_id)
        except Exception:
            pass

    return augmented


if question:
    company_filter = None if company_choice == "All companies" else company_choice

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieval_query = _contextualize_query(question, st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            corpus = _load_corpus(cfg)
            with open_connection(cfg.snowflake) as conn:
                repo = ChunkRepository(conn)
                retrieval = HybridSearchService(
                    embedder=embedder,
                    repository=repo,
                    corpus_chunks=corpus,
                    reranker=reranker,
                )
                chunks = retrieval.retrieve(
                    RetrievalRequest(
                        query=retrieval_query,
                        top_k=top_k,
                        company_filter=company_filter,
                    )
                )

                if not company_filter:
                    chunks = _ensure_company_coverage(retrieval_query, chunks, retrieval)

        user_prompt = build_user_prompt(question, chunks)
        answer_text = st.write_stream(llm.stream(SYSTEM_PROMPT, user_prompt))
        _render_sources(chunks)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "chunks": chunks,
    })
