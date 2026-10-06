#!/usr/bin/env python3
"""
ingest_cuad.py  —  Meethaq AI · CUAD v1 ingestion & legal-aware preprocessing
=============================================================================

Turns the Atticus Project's CUAD v1 corpus (or any folder of contracts) into
clean, structured records that the RAG pipeline can chunk and index.

What it does
────────────
1.  LOAD      CUAD_v1.json (SQuAD-style: title / context / 41 clause questions)
              and/or raw contract files (.txt / .pdf, any folder layout).
2.  CLEAN     Legal-specific cleaning (LegalTextCleaner):
                • signature blocks ("IN WITNESS WHEREOF…", By:/Name:/Title:, /s/)
                • page markers, running headers/footers, EDGAR cover noise,
                  "Confidential Treatment Requested" legends, table of contents
                • hard-wrapped lines re-joined
              Numbered clauses ("§4.2", "12.1", "Section 3.4", "(a)") are
              PROTECTED and are never removed by any rule.
              The "THIS AGREEMENT is made between …" preamble paragraph is KEPT
              on purpose: CUAD annotates Document Name / Parties / Agreement
              Date / Effective Date inside it, and an auditor needs them.
3.  STRUCTURE parse_sections(): ARTICLE / Section / 4.2 / 12.1 / "1." headings
              → a flat list of clause units, each with its hierarchy path
              ("ARTICLE 12 GENERAL > 12.1 Governing Law").
4.  ANNOTATE  CUAD answers (clause_type + text) are re-aligned to character
              spans in the CLEANED text (whitespace/punctuation-insensitive),
              and each is tagged with the section it falls in.
5.  EXPORT    JSONL (one contract per line) and/or direct indexing into the
              RAG pipeline.

Usage
─────
    # download CUAD_v1.json from HuggingFace automatically, process 25 contracts
    python ingest_cuad.py --download --limit 25 --out data/cuad_processed.jsonl

    # local copy of the dataset
    python ingest_cuad.py --json ./CUAD_v1/CUAD_v1.json --out data/cuad.jsonl

    # raw contracts only (no clause labels)
    python ingest_cuad.py --raw-dir ./CUAD_v1/full_contract_txt --out data/raw.jsonl

    # raw files + CUAD labels merged by file name
    python ingest_cuad.py --raw-dir ./CUAD_v1/full_contract_txt --json ./CUAD_v1/CUAD_v1.json

    # process AND index straight into ChromaDB
    python ingest_cuad.py --download --limit 25 --index --chroma-dir ./chroma_db

As a library
────────────
    from ingest_cuad import load_cuad_contracts
    docs = load_cuad_contracts("CUAD_v1.json", limit=5)
    # each doc: contract_name, contract_category, text, sections, annotations, stats
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import subprocess
import traceback
import json
import re
import sys
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
from typing import Optional

CUAD_HF_REPO = "theatticusproject/cuad"
CUAD_JSON_CANDIDATES = ("CUAD_v1/CUAD_v1.json", "CUAD_v1.json")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SHARED REGEXES
# ══════════════════════════════════════════════════════════════════════════════

# A line that starts a numbered clause / heading / sub-clause.
# Such lines are PROTECTED from every removal heuristic below.
CLAUSE_START_RE = re.compile(
    r"^\s*(?:"
    r"§+\s*\d"                                                       # §4.2
    r"|(?i:section|sec\.|article|clause|paragraph|part|schedule|exhibit|annex|appendix)"
    r"\s+[\dA-HIVXLC]"                                               # Section 3 / ARTICLE IV
    r"|\d{1,3}(?:\.\d{1,3})+\.?(?:\s|$)"                             # 12.1  4.2.3.
    r"|\d{1,3}\.\s"                                                  # 1. DEFINITIONS
    r"|\(\s*(?:[a-z]{1,3}|[ivxlc]{1,6}|\d{1,2}|[A-Z])\s*\)"          # (a) (iv) (2)
    r")"
)

STOP_HEADING_RE = re.compile(
    r"^\s*(?i:exhibit|schedule|annex|appendix|attachment|addendum|amendment|article)\b"
)

_ALNUM_RE = re.compile(r"[^\W_]")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LEGAL TEXT CLEANER
# ══════════════════════════════════════════════════════════════════════════════

class LegalTextCleaner:
    """
    Cleaning tuned for corporate contracts (SEC-filing exhibits, PDFs → text).

    Every rule is conservative and *line based*; lines that look like numbered
    clauses (CLAUSE_START_RE) are never dropped.  Paragraph de-duplication is
    deliberately NOT performed: identical boilerplate sentences legitimately
    repeat in contracts (e.g. in different exhibits) and removing them would
    silently delete obligations.

    After `clean()` the per-rule counters are available in `last_stats`.
    """

    _PAGE_MARKER_RES = [
        re.compile(r"^\s*[-–—]*\s*page\s+\d+(?:\s*(?:of|/)\s*\d+)?\s*[-–—]*\s*$", re.I),
        re.compile(r"^\s*\[\s*page\s*\d+\s*\]\s*$", re.I),
        re.compile(r"^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$"),                 # - 12 -
        re.compile(r"^\s*[-–—]?\s*[ivxl]{1,6}\s*[-–—]?\s*$"),            # lowercase roman page no.
    ]
    _LONE_NUMBER_RE = re.compile(r"^\s*(\d{1,4})\s*$")

    _COVER_RES = [
        re.compile(r"^\s*</?[A-Za-z][A-Za-z0-9_\-]*>\s*$"),                              # <PAGE> <TABLE> <S>
        re.compile(r"^\s*(?:ex|exhibit)[\s\-]*\d+(?:[.\-]\d+)*[a-z]?\s*$", re.I),         # EX-10.1
        re.compile(r"^\s*source:\s+.*\b\d{1,2}/\d{1,2}/\d{2,4}\s*$", re.I),               # EDGAR Online footer
        re.compile(r"^\W*confidential\s+treatment\s+(?:has\s+been\s+)?requested\b.*$", re.I),
        re.compile(
            r"^\W*(?:certain\s+)?(?:confidential\s+)?(?:information|portions)\s+"
            r"(?:contained\s+)?(?:in\s+)?this\s+(?:document|exhibit)\b.*(?:omitted|redacted).*$",
            re.I,
        ),
    ]

    _LEGEND_RE = re.compile(
        r"^\s*[\[(]?\s*(?:strictly\s+|highly\s+)?"
        r"(?:confidential|proprietary(?:\s*(?:and|&)\s*confidential)?|"
        r"privileged(?:\s*(?:and|&)\s*confidential)?|draft|execution\s+(?:copy|version)|"
        r"conformed\s+copy|for\s+internal\s+use\s+only|internal\s+use\s+only|do\s+not\s+copy)"
        r"\s*[\])]?\s*(?:[-–—].{0,60})?$",
        re.I,
    )
    _RULE_RE = re.compile(r"^\s*[_\-=~.#·•]{5,}\s*$")
    _WITNESSETH_RE = re.compile(r"^\s*witnesseth\s*:?\s*$", re.I)

    _TOC_HEADING_RE = re.compile(r"^\s*(?:table\s+of\s+contents|contents|index)\s*$", re.I)
    _DOT_LEADER_RE = re.compile(r"^(?=.*[A-Za-z]).*?(?:[.·…_]\s?){4,}\s*\d{1,3}\s*$")
    _TOC_ENTRY_RE = re.compile(r"^\s*.{2,110}?\s\d{1,3}\s*$")

    _WITNESS_RE = re.compile(r"^\s*in\s+witness\s+whereof\b", re.I)
    _AGREED_RE = re.compile(r"^\s*(?:accepted\s+and\s+)?agreed(?:\s+and\s+accepted)?(?:\s+to)?\s*[:.]?\s*$", re.I)
    _SIG_STRONG_RE = re.compile(r"^\s*(?:by\s*[:.]|/s/|_{5,}\s*(?:\(.*\))?\s*$)", re.I)
    _SIG_LABEL_RE = re.compile(
        r"^\s*(?:name|title|its|date|signature|signed|print(?:ed)?\s+name|witness|"
        r"authorized\s+(?:signatory|representative)|address|tel(?:ephone)?|fax|facsimile|"
        r"e-?mail|attn|attention)\s*[:.]",
        re.I,
    )

    def __init__(
        self,
        strip_signatures: bool = True,
        strip_cover_noise: bool = True,
        strip_running_headers: bool = True,
        strip_toc: bool = True,
        unwrap_lines: bool = True,
    ):
        self.strip_signatures = strip_signatures
        self.strip_cover_noise = strip_cover_noise
        self.strip_running_headers = strip_running_headers
        self.strip_toc = strip_toc
        self.unwrap_lines = unwrap_lines
        self.last_stats: Counter = Counter()

    # ── public API ────────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        stats: Counter = Counter()
        lines = self._normalise(text).split("\n")

        if self.strip_toc:
            lines = self._drop_toc(lines, stats)
        if self.strip_cover_noise:
            lines = self._drop_cover_noise(lines, stats)
        lines = self._drop_page_markers(lines, stats)
        if self.strip_running_headers:
            lines = self._drop_running_headers(lines, stats)
        if self.strip_signatures:
            lines = self._drop_signature_blocks(lines, stats)
        lines = self._drop_legends(lines, stats)
        if self.unwrap_lines:
            lines = self._unwrap(lines, stats)

        out = "\n".join(lines)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        self.last_stats = stats
        return out

    # ── step 0: character-level normalisation ─────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)                 # ligatures, NBSP, full-width
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\f", "\n\n")                          # form feed = page break
        text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
        text = text.replace("\t", " ")
        text = re.sub(r"[ ]{2,}", " ", text)
        return "\n".join(ln.strip() for ln in text.split("\n"))

    # ── step 1: table of contents ─────────────────────────────────────────────

    def _drop_toc(self, lines: list[str], stats: Counter) -> list[str]:
        n = len(lines)
        drop: set[int] = set()

        # (a) explicit "TABLE OF CONTENTS" region
        horizon = min(n, max(300, n // 4))
        i = 0
        while i < horizon:
            if self._TOC_HEADING_RE.match(lines[i]):
                j, entries = i + 1, 0
                region: list[int] = []
                while j < n:
                    t = lines[j]
                    if not t:
                        region.append(j)
                    elif self._DOT_LEADER_RE.match(t) or (len(t) <= 110 and self._TOC_ENTRY_RE.match(t)):
                        region.append(j)
                        entries += 1
                    else:
                        break
                    j += 1
                if entries >= 5:
                    drop.update(region)
                    drop.add(i)
                    i = j
                    continue
            i += 1

        # (b) dot-leader lines anywhere ("Termination .......... 14")
        for k, t in enumerate(lines):
            if t and self._DOT_LEADER_RE.match(t):
                drop.add(k)

        stats["toc_lines"] += len(drop)
        return [t for k, t in enumerate(lines) if k not in drop]

    # ── step 2: EDGAR / cover noise ───────────────────────────────────────────

    def _drop_cover_noise(self, lines: list[str], stats: Counter) -> list[str]:
        out = []
        for t in lines:
            if t and len(t) < 250 and not CLAUSE_START_RE.match(t) and any(r.match(t) for r in self._COVER_RES):
                stats["cover_noise"] += 1
                continue
            out.append(t)
        return out

    # ── step 3: page markers ──────────────────────────────────────────────────

    def _drop_page_markers(self, lines: list[str], stats: Counter) -> list[str]:
        drop: set[int] = set()
        lone: list[tuple[int, int]] = []                           # (line index, value)
        for i, t in enumerate(lines):
            if not t:
                continue
            if any(r.match(t) for r in self._PAGE_MARKER_RES):
                drop.add(i)
                continue
            m = self._LONE_NUMBER_RE.match(t)
            if m:
                lone.append((i, int(m.group(1))))

        # A lone number is a page number only if it sits next to its neighbour in a
        # 1,2,3… sequence.  Isolated numbers ("30", "90") may be table cells → kept.
        for k, (i, v) in enumerate(lone):
            prev_v = lone[k - 1][1] if k > 0 else None
            next_v = lone[k + 1][1] if k + 1 < len(lone) else None
            if (prev_v is not None and 0 < v - prev_v <= 3) or (next_v is not None and 0 < next_v - v <= 3):
                drop.add(i)

        stats["page_markers"] += len(drop)
        return [t for i, t in enumerate(lines) if i not in drop]

    # ── step 4: running headers / footers ─────────────────────────────────────

    def _drop_running_headers(self, lines: list[str], stats: Counter) -> list[str]:
        n = len(lines)
        if n < 80:
            return lines
        buckets: dict[str, list[int]] = {}
        for i, t in enumerate(lines):
            if not (6 <= len(t) <= 100) or t.endswith(":") or CLAUSE_START_RE.match(t):
                continue
            if sum(c.isalpha() for c in t) < 3:
                continue
            key = re.sub(r"\d+", "#", t.lower())
            buckets.setdefault(key, []).append(i)
        drop: set[int] = set()
        for idxs in buckets.values():
            # a real running header repeats ≥4 times and is spread over the document
            if len(idxs) >= 4 and (idxs[-1] - idxs[0]) >= 0.5 * n:
                drop.update(idxs)
        stats["running_headers"] += len(drop)
        return [t for i, t in enumerate(lines) if i not in drop]

    # ── step 5: signature blocks ──────────────────────────────────────────────

    def _short_unlabeled(self, t: str) -> bool:
        """Short, name/entity/title-like line (e.g. 'ACME CORP.', 'Chief Executive Officer')."""
        if not t or CLAUSE_START_RE.match(t) or STOP_HEADING_RE.match(t):
            return False
        words = t.split()
        if len(t) > 60 or len(words) > 8:
            return False
        if t.endswith(".") and len(words) > 5:
            return False
        return True

    def _drop_signature_blocks(self, lines: list[str], stats: Counter) -> list[str]:
        n = len(lines)
        remove = [False] * n

        for i in range(n):
            if remove[i] or not self._SIG_STRONG_RE.match(lines[i]):
                continue

            # forward: labelled lines + a few short name/title lines
            end, gap, k = i, 0, i + 1
            while k < n:
                t = lines[k]
                if not t:
                    k += 1
                    continue
                if self._SIG_STRONG_RE.match(t) or self._SIG_LABEL_RE.match(t):
                    end, gap = k, 0
                elif self._short_unlabeled(t) and gap < 4:
                    gap += 1
                    end = k
                else:
                    break
                k += 1

            # backward: entity names / "AGREED:" lines directly above the first marker
            start, back, seen = i, i - 1, 0
            while back >= 0 and seen < 6:
                t = lines[back]
                if not t:
                    back -= 1
                    continue
                if self._short_unlabeled(t) or self._SIG_LABEL_RE.match(t) or self._AGREED_RE.match(t):
                    start = back
                    seen += 1
                    back -= 1
                else:
                    break

            # the "IN WITNESS WHEREOF…" paragraph a few lines above
            w, steps = start - 1, 0
            while w >= 0 and steps < 5:
                t = lines[w]
                if self._WITNESS_RE.match(t):
                    start = w
                    break
                if CLAUSE_START_RE.match(t):
                    break
                if t:
                    steps += 1
                w -= 1

            for j in range(start, end + 1):
                remove[j] = True

        # stand-alone "IN WITNESS WHEREOF" paragraph (signature page absent)
        for i in range(n):
            if not remove[i] and self._WITNESS_RE.match(lines[i]):
                j = i
                while j < n and lines[j] and j - i < 4 and not CLAUSE_START_RE.match(lines[j]) or j == i:
                    remove[j] = True
                    j += 1
                    if j >= n:
                        break

        stats["signature_lines"] += sum(remove)
        return [t for i, t in enumerate(lines) if not remove[i]]

    # ── step 6: legends, rules, WITNESSETH ────────────────────────────────────

    def _drop_legends(self, lines: list[str], stats: Counter) -> list[str]:
        out = []
        for t in lines:
            if t and not CLAUSE_START_RE.match(t) and (
                self._RULE_RE.match(t) or self._LEGEND_RE.match(t) or self._WITNESSETH_RE.match(t)
            ):
                stats["legends_rules"] += 1
                continue
            out.append(t)
        return out

    # ── step 7: un-wrap hard line breaks ──────────────────────────────────────

    @staticmethod
    def _unwrap(lines: list[str], stats: Counter) -> list[str]:
        out: list[str] = []
        for t in lines:
            if (
                out and out[-1] and t
                and t[0].islower()                                  # continues the sentence
                and out[-1][-1] not in ".:;?!"                      # previous line did not end it
            ):
                joiner = "" if out[-1].endswith("-") else " "
                out[-1] = out[-1] + joiner + t
                stats["unwrapped"] += 1
            else:
                out.append(t)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONTRACT STRUCTURE  (section hierarchy)
# ══════════════════════════════════════════════════════════════════════════════

_ROMAN = r"[IVXLC]{1,7}"
_WORDNUM = (
    r"(?i:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
)
_RE_ARTICLE = re.compile(
    rf"^\s*(?P<kw>ARTICLE|Article|PART|Part)\s+(?P<num>{_ROMAN}|\d{{1,3}}|{_WORDNUM})\b[\s.:\-–—]*(?P<title>.*)$"
)
_RE_SECTION_KW = re.compile(
    r"^\s*(?P<kw>SECTION|Section|SEC\.|Sec\.|§§?)\s*(?P<num>\d{1,3}(?:\.\d{1,3})*)[A-Za-z]?\s*[.:\-–—]?\s*(?P<title>.*)$"
)
_RE_DECIMAL = re.compile(r"^\s*(?P<num>\d{1,3}(?:\.\d{1,3})+)\.?\s+(?P<title>\S.*)$")
_RE_INTEGER = re.compile(r"^\s*(?P<num>\d{1,3})\.\s+(?P<title>[A-Z“\"(].*)$")


def _title_ok(title: str, allow_empty: bool = True) -> bool:
    if not title:
        return allow_empty
    return not title[0].islower()


def _clean_title(raw: str, limit: int = 70) -> str:
    raw = raw.strip()
    first = re.split(r"(?<=[.:;])\s", raw, maxsplit=1)[0]
    first = first.strip(" .:;-–—\"“”'")
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0]
    return first


def _match_heading(s: str, prev: str, prev_was_heading: bool) -> Optional[dict]:
    if not s:
        return None
    m = _RE_ARTICLE.match(s)
    if m and _title_ok(m.group("title")) and len(s) < 160:
        return {"kind": "article", "number": m.group("num"), "title": _clean_title(m.group("title")),
                "depth": 1, "kw": m.group("kw").upper()}
    m = _RE_SECTION_KW.match(s)
    if m and _title_ok(m.group("title")):
        # "Section 4.2 …" at line start is often a mid-sentence cross-reference:
        # require a clean break before it.
        if not prev or prev[-1] in '.;:)"”' or prev_was_heading:
            num = m.group("num")
            return {"kind": "section", "number": num, "title": _clean_title(m.group("title")),
                    "depth": num.count(".") + 1, "kw": "Section"}
    m = _RE_DECIMAL.match(s)
    if m and _title_ok(m.group("title"), allow_empty=False):
        num = m.group("num")
        return {"kind": "decimal", "number": num, "title": _clean_title(m.group("title")),
                "depth": num.count(".") + 1, "kw": ""}
    m = _RE_INTEGER.match(s)
    if m:
        return {"kind": "integer", "number": m.group("num"), "title": _clean_title(m.group("title")),
                "depth": 1, "kw": ""}
    return None


def parse_sections(text: str) -> list[dict]:
    """
    Split `text` into a flat, gap-free list of clause units, each carrying its
    place in the contract hierarchy.

    Returns dicts with:
        index, level, number, title, label, path, start, end
    where [start, end) are character offsets into `text`, and `path` is e.g.
    "ARTICLE 12 GENERAL PROVISIONS > 12.1 Governing Law".
    A unit runs from one heading to the next heading of ANY depth.
    """
    lines = text.split("\n")
    heads: list[dict] = []
    pos, prev, prev_was_heading = 0, "", False
    for i, ln in enumerate(lines):
        s = ln.strip()
        h = _match_heading(s, prev, prev_was_heading)
        if h:
            if h["kind"] == "article" and not h["title"] and i + 1 < len(lines):
                nxt = lines[i + 1].strip()                          # title on the following line
                if 0 < len(nxt) <= 80 and _match_heading(nxt, "", True) is None:
                    h["title"] = _clean_title(nxt)
            h["start"] = pos + (len(ln) - len(ln.lstrip()))
            heads.append(h)
        prev_was_heading = h is not None
        prev = s
        pos += len(ln) + 1

    if not heads:
        return [{"index": 0, "level": 0, "number": "", "title": "", "label": "", "path": "",
                 "start": 0, "end": len(text)}]

    has_articles = any(h["kind"] == "article" for h in heads)
    units: list[dict] = []

    if text[: heads[0]["start"]].strip():
        units.append({"index": 0, "level": 0, "number": "", "title": "Preamble", "label": "Preamble",
                      "path": "Preamble", "start": 0, "end": heads[0]["start"]})

    stack: list[tuple[int, str]] = []
    for h in heads:
        if h["kind"] == "article":
            level = 1
            label = f'{h["kw"]} {h["number"]} {h["title"]}'.strip()
        else:
            level = h["depth"] + (1 if has_articles else 0)
            prefix = "Section " if h["kind"] == "section" else ""
            label = f'{prefix}{h["number"]} {h["title"]}'.strip()
        label = label[:100]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, label))
        units.append({"index": 0, "level": level, "number": h["number"], "title": h["title"],
                      "label": label, "path": " > ".join(lab for _, lab in stack),
                      "start": h["start"], "end": 0})

    for k, u in enumerate(units):
        u["index"] = k
        u["end"] = units[k + 1]["start"] if k + 1 < len(units) else len(text)
    return units


def section_for_offset(sections: list[dict], offset: int) -> dict:
    starts = [s["start"] for s in sections]
    return sections[max(0, bisect_right(starts, offset) - 1)]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONTRACT CATEGORY (CUAD "contract type") — folder name or title heuristic
# ══════════════════════════════════════════════════════════════════════════════

_CATEGORY_KEYWORDS = [
    ("Affiliate Agreements", ("affiliate",)),
    ("Co-Branding", ("co-brand", "cobrand", "co branding")),
    ("Development", ("development",)),
    ("Distributor", ("distributor", "distribution")),
    ("Endorsement", ("endorsement",)),
    ("Franchise", ("franchise",)),
    ("Hosting", ("hosting",)),
    ("Joint Venture", ("joint venture",)),
    ("Maintenance", ("maintenance",)),
    ("Manufacturing", ("manufactur",)),
    ("Marketing", ("marketing",)),
    ("Non-Compete/Exclusivity", ("non-compet", "noncompet", "exclusivity")),
    ("Outsourcing", ("outsourc",)),
    ("Promotion", ("promotion",)),
    ("Reseller", ("reseller",)),
    ("Sponsorship", ("sponsor",)),
    ("Strategic Alliance", ("strategic alliance", "alliance")),
    ("Supply", ("supply",)),
    ("Transportation", ("transportation",)),
    ("License Agreements", ("license", "licence")),
    ("Service", ("service",)),
    ("IP", ("intellectual property",)),
]


def infer_contract_category(title: str) -> str:
    """Best-effort contract type from the title suffix (…-EX-10-DISTRIBUTOR AGREEMENT)."""
    suffix = re.split(r"[-–]", title)[-1].lower()
    for name, keys in _CATEGORY_KEYWORDS:
        if any(k in suffix for k in keys):
            return name
    return "Other"


def _category_from_path(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        parts = ()
    parts = [p for p in parts if not re.match(r"(?i)part[_\s-]*[ivx\d]+$", p)]
    return parts[-1].replace("_", " ").strip() if parts else ""


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CUAD JSON  +  ANNOTATION ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def download_cuad_json(cache_dir: Optional[str] = None) -> Path:
    """Fetch CUAD_v1.json from the HuggingFace Hub (cached after the first call)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:                                       # pragma: no cover
        raise ImportError("pip install huggingface_hub") from exc
    last_err: Optional[Exception] = None
    for fname in CUAD_JSON_CANDIDATES:
        try:
            return Path(hf_hub_download(repo_id=CUAD_HF_REPO, filename=fname,
                                        repo_type="dataset", cache_dir=cache_dir))
        except Exception as exc:                                     # noqa: BLE001
            last_err = exc
    raise RuntimeError(
        "Could not download CUAD_v1.json from HuggingFace "
        f"({CUAD_HF_REPO}). Download it manually from "
        "https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1 "
        f"and pass --json <path>.  Last error: {last_err}"
    )


def _clause_type_from_qa(qa: dict) -> str:
    qid = qa.get("id", "")
    if "__" in qid:
        return qid.rsplit("__", 1)[1].strip()
    m = re.search(r'related to "([^"]+)"', qa.get("question", ""))
    return m.group(1).strip() if m else "Unknown"


def read_cuad_json(path: str | Path) -> list[dict]:
    """
    Parse CUAD_v1.json into raw records:
        {"title", "context", "annotations": [{"clause_type", "text", "orig_start"}]}
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records = []
    for article in payload["data"]:
        title = article["title"]
        paragraphs = article.get("paragraphs", [])
        for p_idx, para in enumerate(paragraphs):
            anns = []
            for qa in para.get("qas", []):
                ctype = _clause_type_from_qa(qa)
                for ans in qa.get("answers", []):
                    if ans.get("text", "").strip():
                        anns.append({"clause_type": ctype, "text": ans["text"],
                                     "orig_start": int(ans.get("answer_start", 0))})
            name = title if p_idx == 0 else f"{title}#p{p_idx + 1}"
            records.append({"title": name, "context": para["context"], "annotations": anns})
    return records


def _squash(text: str) -> tuple[str, list[int]]:
    """Lower-cased alphanumerics only + map back to original character offsets."""
    idx = [m.start() for m in _ALNUM_RE.finditer(text)]
    return "".join(text[i] for i in idx).lower(), idx


def _find_nearest(hay: str, needle: str, est: int, cap: int = 60) -> Optional[tuple[int, int]]:
    best, best_d, pos, seen = None, None, hay.find(needle), 0
    while pos != -1 and seen < cap:
        d = abs(pos - est)
        if best_d is None or d < best_d:
            best, best_d = pos, d
        pos = hay.find(needle, pos + 1)
        seen += 1
    return (best, best + len(needle)) if best is not None else None


def align_annotations(raw_text: str, clean_text: str, anns: list[dict]) -> tuple[list[dict], int]:
    """
    Re-locate CUAD answers inside the CLEANED text.

    Matching is done on lower-cased alphanumerics only, so differences in
    whitespace, quotes, hyphenation and punctuation introduced by cleaning do
    not matter.  Long answers interrupted by removed page markers fall back to
    head/tail anchoring.

    Returns (aligned annotations with 'start'/'end', number of unaligned answers).
    """
    if not anns:
        return [], 0
    clean_sq, clean_idx = _squash(clean_text)
    raw_idx = [m.start() for m in _ALNUM_RE.finditer(raw_text)]
    ratio = len(clean_sq) / max(1, len(raw_idx))

    aligned, missed = [], 0
    for a in anns:
        ans_sq = "".join(_ALNUM_RE.findall(a["text"])).lower()
        if len(ans_sq) < 2:
            missed += 1
            continue
        est = int(bisect_left(raw_idx, a.get("orig_start", 0)) * ratio)
        span = _find_nearest(clean_sq, ans_sq, est)
        if span is None and len(ans_sq) > 120:
            head, tail = ans_sq[:50], ans_sq[-50:]
            h = _find_nearest(clean_sq, head, est)
            if h:
                t = clean_sq.find(tail, h[0] + len(head), h[0] + int(len(ans_sq) * 1.8) + 300)
                if t != -1:
                    span = (h[0], t + len(tail))
        if span is None:
            missed += 1
            continue
        start, end = clean_idx[span[0]], clean_idx[span[1] - 1] + 1
        aligned.append({"clause_type": a["clause_type"], "text": clean_text[start:end],
                        "start": start, "end": end})
    aligned.sort(key=lambda x: (x["start"], x["end"]))
    return aligned, missed


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CONTRACT PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_contract(
    title: str,
    raw_text: str,
    annotations: Optional[list[dict]] = None,
    cleaner: Optional[LegalTextCleaner] = None,
    source_path: Optional[str] = None,
    category: Optional[str] = None,
) -> dict:
    """Clean → structure → align labels.  Returns the record the pipeline indexes."""
    cleaner = cleaner or LegalTextCleaner()
    cleaned = cleaner.clean(raw_text)
    sections = parse_sections(cleaned)
    aligned, missed = align_annotations(raw_text, cleaned, annotations or [])
    for a in aligned:
        sec = section_for_offset(sections, a["start"])
        a["section_label"] = sec["label"]
        a["section_path"] = sec["path"]
    return {
        "contract_name": title,
        "contract_category": category or infer_contract_category(title),
        "source_path": source_path or "",
        "text": cleaned,
        "sections": sections,
        "annotations": aligned,
        "clause_types_present": sorted({a["clause_type"] for a in aligned}),
        "stats": {
            "raw_chars": len(raw_text),
            "clean_chars": len(cleaned),
            "n_sections": len(sections),
            "n_annotations": len(aligned),
            "n_unaligned": missed,
            "cleaner": dict(cleaner.last_stats),
        },
    }


def _norm_key(name: str) -> str:
    return "".join(_ALNUM_RE.findall(Path(name).stem if "." in name[-6:] else name)).lower()


def load_raw_contracts(raw_dir: str | Path, cuad_records: Optional[list[dict]] = None,
                       cleaner: Optional[LegalTextCleaner] = None,
                       limit: Optional[int] = None) -> list[dict]:
    """
    Load .txt / .pdf contracts from `raw_dir` (recursive).  If `cuad_records`
    is given, labels are merged in by (normalised) file name.
    """
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    files: dict[str, Path] = {}
    for ext in (".pdf", ".txt"):                                     # .txt wins over .pdf
        for p in sorted(root.rglob(f"*{ext}")):
            files[str(p.with_suffix("")).lower()] = p
    labels = {_norm_key(r["title"]): r["annotations"] for r in (cuad_records or [])}

    docs = []
    for key in sorted(files):
        p = files[key]
        raw = _read_raw_file(p)
        if not raw.strip():
            continue
        anns = labels.get(_norm_key(p.stem), [])
        docs.append(prepare_contract(p.stem, raw, anns, cleaner, str(p), _category_from_path(p, root) or None))
        if limit and len(docs) >= limit:
            break
    if not docs:
        raise ValueError(f"No readable .txt/.pdf contracts found in {root}")
    return docs


def _read_raw_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz                                              # PyMuPDF
        except ImportError as exc:                                   # pragma: no cover
            raise ImportError("pip install pymupdf") from exc
        with fitz.open(str(path)) as pdf:
            return "\n".join(page.get_text() for page in pdf)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_cuad_contracts(
    json_path: str | Path,
    cleaner: Optional[LegalTextCleaner] = None,
    titles: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Load CUAD_v1.json → prepared contract records (cleaned, structured, labelled)."""
    cleaner = cleaner or LegalTextCleaner()
    records = read_cuad_json(json_path)
    if titles:
        wanted = set(titles)
        records = [r for r in records if r["title"] in wanted]
    if limit:
        records = records[:limit]
    return [prepare_contract(r["title"], r["context"], r["annotations"], cleaner) for r in records]


def save_jsonl(docs: list[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".new")
    with open(temporary, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, out)
    return out


def load_processed(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(docs: list[dict]) -> None:
    raw = sum(d["stats"]["raw_chars"] for d in docs)
    clean = sum(d["stats"]["clean_chars"] for d in docs)
    anns = sum(d["stats"]["n_annotations"] for d in docs)
    missed = sum(d["stats"]["n_unaligned"] for d in docs)
    rules: Counter = Counter()
    for d in docs:
        rules.update(d["stats"]["cleaner"])
    types: Counter = Counter()
    cats: Counter = Counter()
    for d in docs:
        cats[d["contract_category"]] += 1
        types.update(a["clause_type"] for a in d["annotations"])

    print("\n" + "═" * 64)
    print(f" INGEST SUMMARY   ({len(docs)} contracts)")
    print("═" * 64)
    print(f" characters      : {raw:,} raw → {clean:,} cleaned ({100 * (1 - clean / max(1, raw)):.1f}% removed)")
    print(f" sections        : {sum(d['stats']['n_sections'] for d in docs):,}")
    print(f" clause labels   : {anns:,} aligned, {missed:,} could not be aligned")
    print(f" cleaner rules   : {dict(rules)}")
    print(f" categories      : {dict(cats.most_common(8))}")
    print(f" top clause types: {dict(types.most_common(8))}")
    print("═" * 64)


def _index_in_child(out: Path, chroma_dir: str, collection: str) -> None:
    """Catch native access violations and enforce explicit worker completion."""
    from vector_store import ensure_disk_space, resolve_chroma_dir
    directory = resolve_chroma_dir(chroma_dir)
    ensure_disk_space(directory)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable, "-B", "-u", str(Path(__file__).resolve()),
        "--processed", str(out.resolve()), "--index-worker",
        "--chroma-dir", directory, "--collection", collection,
    ]
    print(f"[Indexing] Starting supervised worker; database={directory}; collection={collection}", flush=True)
    result = subprocess.run(command, env=environment, cwd=str(Path(__file__).resolve().parent))
    if result.returncode:
        code = result.returncode
        raise RuntimeError(
            f"Indexing worker failed with exit code {code} (0x{code & 0xffffffff:08X}). "
            "See the last flushed stage/native traceback above. Check disk space, "
            "available memory, cached ONNX files and competing database writers; "
            "the JSONL export was preserved."
        )


def _index_processed_in_worker(path: Path, chroma_dir: str, collection: str) -> int:
    from rag_pipeline import RAGPipeline
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    faulthandler.enable(all_threads=True)
    docs = load_processed(path)
    pipeline = None
    try:
        pipeline = RAGPipeline(chroma_dir=chroma_dir, collection_name=collection)
        pipeline.index_prepared(docs)
        print(f"[Indexing] Completed and persisted {pipeline.stats()['indexed_chunks']} chunks.", flush=True)
        return 0
    finally:
        if pipeline is not None:
            pipeline.close()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest & preprocess CUAD v1 for the Meethaq AI RAG pipeline.")
    ap.add_argument("--json", type=Path, help="path to CUAD_v1.json")
    ap.add_argument("--download", action="store_true", help="download CUAD_v1.json from HuggingFace")
    ap.add_argument("--raw-dir", type=Path, help="folder of raw .txt/.pdf contracts")
    ap.add_argument("--out", type=Path, default=Path("data/cuad_processed.jsonl"), help="output JSONL path")
    ap.add_argument("--limit", type=int, help="process only the first N contracts")
    ap.add_argument("--titles", nargs="*", help="process only these contract titles (JSON mode)")
    ap.add_argument("--index", action="store_true", help="also index into ChromaDB via RAGPipeline")
    ap.add_argument("--chroma-dir", default=None, help="persistent DB directory, anchored to this project")
    ap.add_argument("--processed", type=Path, help="index an existing prepared JSONL export")
    ap.add_argument("--index-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--collection", default=os.environ.get("MEETHAQ_COLLECTION", "meethaq_contracts"))
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if args.index_worker:
        if args.processed is None:
            ap.error("--index-worker requires --processed")
        return _index_processed_in_worker(args.processed, args.chroma_dir, args.collection)
    if args.processed is not None:
        if not args.index:
            ap.error("--processed requires --index")
        _index_in_child(args.processed, args.chroma_dir, args.collection)
        return 0

    cleaner = LegalTextCleaner()
    json_path = args.json
    if json_path is None and args.download:
        print("[ingest] downloading CUAD_v1.json from HuggingFace …")
        json_path = download_cuad_json()
        print(f"[ingest] using {json_path}")

    if args.raw_dir:
        records = read_cuad_json(json_path) if json_path else None
        docs = load_raw_contracts(args.raw_dir, records, cleaner, args.limit)
    elif json_path:
        docs = load_cuad_contracts(json_path, cleaner, args.titles, args.limit)
    else:
        ap.error("provide --json, --download or --raw-dir")
        return 2

    _print_summary(docs)
    out = save_jsonl(docs, args.out)
    print(f"[ingest] wrote {len(docs)} contracts → {out}")

    if args.index:
        _index_in_child(out, args.chroma_dir, args.collection)
    return 0


if __name__ == "__main__":
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    faulthandler.enable(all_threads=True)
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
