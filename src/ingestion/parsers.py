"""Multi-format company document parsers (.pdf via pymupdf, .md, .txt, .csv via stdlib)."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.config import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class ParsedSection:
    """Intermediate text unit with provenance before recursive chunking."""

    source_file: str
    page_or_section: str
    text: str


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_TXT_SECTION_RE = re.compile(r"^\[(.+?)\]\s*$", re.MULTILINE)


def parse_pdf(file_path: Path) -> list[ParsedSection]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    sections: list[ParsedSection] = []
    source_name = file_path.name
    with pymupdf.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                sections.append(
                    ParsedSection(
                        source_file=source_name,
                        page_or_section=f"Page {page_num}",
                        text=page_text,
                    )
                )
    return sections


def parse_markdown(file_path: Path) -> list[ParsedSection]:
    """Split a Markdown file by headings while preserving the heading inside each block."""
    raw_text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw_text:
        return []

    source_name = file_path.name
    matches = list(_MD_HEADING_RE.finditer(raw_text))
    if not matches:
        return [ParsedSection(source_file=source_name, page_or_section="Document Body", text=raw_text)]

    sections: list[ParsedSection] = []
    if matches[0].start() > 0:
        preamble = raw_text[: matches[0].start()].strip()
        if preamble:
            sections.append(
                ParsedSection(source_file=source_name, page_or_section="Preamble", text=preamble)
            )

    for idx, match in enumerate(matches):
        heading_title = match.group(2).strip()
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        block_text = raw_text[start:end].strip()
        if block_text:
            sections.append(
                ParsedSection(
                    source_file=source_name,
                    page_or_section=f"Section: {heading_title}",
                    text=block_text,
                )
            )
    return sections


def parse_txt(file_path: Path) -> list[ParsedSection]:
    """Parse plain text files, tracking [SECTION ...] blocks if present."""
    raw_text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw_text:
        return []

    source_name = file_path.name
    matches = list(_TXT_SECTION_RE.finditer(raw_text))
    if not matches:
        return [ParsedSection(source_file=source_name, page_or_section="Text Document", text=raw_text)]

    sections: list[ParsedSection] = []
    if matches[0].start() > 0:
        header_block = raw_text[: matches[0].start()].strip()
        if header_block:
            sections.append(
                ParsedSection(
                    source_file=source_name,
                    page_or_section="Document Header",
                    text=header_block,
                )
            )

    for idx, match in enumerate(matches):
        sec_title = match.group(1).strip()
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        block_text = raw_text[start:end].strip()
        if block_text:
            sections.append(
                ParsedSection(
                    source_file=source_name,
                    page_or_section=sec_title,
                    text=block_text,
                )
            )
    return sections


def parse_csv(file_path: Path, rows_per_section: int = 3) -> list[ParsedSection]:
    """Serialize CSV rows with explicit column headers so every chunk remains self-contained."""
    source_name = file_path.name
    sections: list[ParsedSection] = []

    with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        batch_lines: list[str] = []
        batch_start_row = 1

        for row_idx, row in enumerate(reader, start=1):
            kv_pairs = [
                f"{col.strip()}: {(val or '').strip()}"
                for col, val in row.items()
                if col is not None and (val or "").strip() != ""
            ]
            if not kv_pairs:
                continue
            batch_lines.append(f"Record {row_idx} -> " + " | ".join(kv_pairs))

            if len(batch_lines) >= rows_per_section:
                row_label = (
                    f"Row {batch_start_row}"
                    if batch_start_row == row_idx
                    else f"Rows {batch_start_row}-{row_idx}"
                )
                sections.append(
                    ParsedSection(
                        source_file=source_name,
                        page_or_section=row_label,
                        text="\n".join(batch_lines),
                    )
                )
                batch_lines = []
                batch_start_row = row_idx + 1

        if batch_lines:
            end_row = batch_start_row + len(batch_lines) - 1
            row_label = (
                f"Row {batch_start_row}"
                if batch_start_row == end_row
                else f"Rows {batch_start_row}-{end_row}"
            )
            sections.append(
                ParsedSection(
                    source_file=source_name,
                    page_or_section=row_label,
                    text="\n".join(batch_lines),
                )
            )

    return sections


def parse_file(file_path: Path) -> list[ParsedSection]:
    """Dispatch a file to the appropriate format parser."""
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{suffix}' for {file_path.name}")
    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".md":
        return parse_markdown(file_path)
    if suffix == ".txt":
        return parse_txt(file_path)
    if suffix == ".csv":
        return parse_csv(file_path)
    raise ValueError(f"Unhandled extension: {suffix}")
