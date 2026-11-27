"""Multi-format company document parsers (.pdf, .xlsx/.xlsm, .png/.jpg/.jpeg/.webp, .md, .txt, .csv)."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import openpyxl
from PIL import Image, UnidentifiedImageError
import pymupdf
from rapidocr_onnxruntime import RapidOCR

from src.config import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS


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


def _format_row_label(start_row: int, end_row: int, sheet_name: str | None = None) -> str:
    """Format a human-readable row range label, optionally scoped to an Excel sheet."""
    row_part = f"Row {start_row}" if start_row == end_row else f"Rows {start_row}-{end_row}"
    if sheet_name:
        return f"Sheet '{sheet_name}' ({row_part})"
    return row_part


def _batch_tabular_rows(
    source_name: str,
    rows: Iterable[tuple[int, dict[str, str]]],
    rows_per_section: int = 3,
    sheet_name: str | None = None,
) -> list[ParsedSection]:
    """Group tabular records into self-contained text blocks with explicit column names."""
    sections: list[ParsedSection] = []
    batch_lines: list[str] = []
    batch_indices: list[int] = []
    header_prefix = f"[Sheet: {sheet_name}]\n" if sheet_name else ""

    for row_idx, row_dict in rows:
        kv_pairs = [
            f"{col}: {val}"
            for col, val in row_dict.items()
            if col and val
        ]
        if not kv_pairs:
            continue
        batch_lines.append(f"Record {row_idx} -> " + " | ".join(kv_pairs))
        batch_indices.append(row_idx)

        if len(batch_lines) >= rows_per_section:
            sections.append(
                ParsedSection(
                    source_file=source_name,
                    page_or_section=_format_row_label(batch_indices[0], batch_indices[-1], sheet_name),
                    text=header_prefix + "\n".join(batch_lines),
                )
            )
            batch_lines = []
            batch_indices = []

    if batch_lines:
        sections.append(
            ParsedSection(
                source_file=source_name,
                page_or_section=_format_row_label(batch_indices[0], batch_indices[-1], sheet_name),
                text=header_prefix + "\n".join(batch_lines),
            )
        )

    return sections


def parse_csv(file_path: Path, rows_per_section: int = 3) -> list[ParsedSection]:
    """Serialize CSV rows with explicit column headers so every chunk remains self-contained."""
    source_name = file_path.name
    with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        cleaned_rows = (
            (
                row_idx,
                {
                    col.strip(): (val or "").strip()
                    for col, val in row.items()
                    if col is not None
                },
            )
            for row_idx, row in enumerate(reader, start=1)
        )
        return _batch_tabular_rows(
            source_name=source_name,
            rows=cleaned_rows,
            rows_per_section=rows_per_section,
        )


def parse_excel(file_path: Path, rows_per_section: int = 3) -> list[ParsedSection]:
    """Parse multi-sheet Excel (.xlsx/.xlsm) workbooks into header-annotated sections."""
    source_name = file_path.name
    sections: list[ParsedSection] = []
    workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    try:
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            row_iter = sheet.iter_rows(values_only=True)

            # Locate the first non-empty row as column headers
            headers: list[str] = []
            for raw_row in row_iter:
                if raw_row and any(cell is not None and str(cell).strip() != "" for cell in raw_row):
                    headers = [
                        str(cell).strip() if cell is not None and str(cell).strip() else f"Column_{idx + 1}"
                        for idx, cell in enumerate(raw_row)
                    ]
                    break

            if not headers:
                continue

            tabular_rows: list[tuple[int, dict[str, str]]] = []
            for row_idx, raw_row in enumerate(row_iter, start=1):
                if not raw_row:
                    continue
                row_dict: dict[str, str] = {}
                for col_idx, cell in enumerate(raw_row):
                    if cell is None:
                        continue
                    cell_str = str(cell).strip()
                    if not cell_str:
                        continue
                    col_name = headers[col_idx] if col_idx < len(headers) else f"Column_{col_idx + 1}"
                    row_dict[col_name] = cell_str
                if row_dict:
                    tabular_rows.append((row_idx, row_dict))

            sections.extend(
                _batch_tabular_rows(
                    source_name=source_name,
                    rows=tabular_rows,
                    rows_per_section=rows_per_section,
                    sheet_name=sheet_name,
                )
            )
    finally:
        workbook.close()

    return sections


@lru_cache(maxsize=1)
def get_ocr_engine() -> RapidOCR:
    """Load and cache the local CPU RapidOCR ONNX engine."""
    return RapidOCR()


def _reconstruct_spatial_lines(ocr_results: list) -> str:
    """Group OCR bounding boxes into top-to-bottom rows and left-to-right columns."""
    items: list[tuple[float, float, float, float, str]] = []
    heights: list[float] = []

    for entry in ocr_results:
        box, text, _conf = entry[0], str(entry[1]).strip(), entry[2]
        if not text:
            continue
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        y_mid = (y_min + y_max) / 2.0
        h = max(1.0, y_max - y_min)
        heights.append(h)
        items.append((y_mid, x_min, x_max, h, text))

    if not items:
        return ""

    median_h = float(np.median(heights)) if heights else 16.0
    row_tol = max(10.0, median_h * 0.55)

    # Sort top-to-bottom first, then group into horizontal rows
    items.sort(key=lambda it: (it[0], it[1]))
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for it in items:
        if not rows or abs(it[0] - rows[-1][0][0]) > row_tol:
            rows.append([it])
        else:
            rows[-1].append(it)

    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda it: it[1])
        parts: list[str] = [row[0][4]]
        prev_x_max = row[0][2]
        for cell in row[1:]:
            gap = cell[1] - prev_x_max
            # Wide horizontal gap indicates separate table columns / chart labels
            sep = " | " if gap >= max(24.0, median_h * 1.4) else " "
            parts.append(sep + cell[4])
            prev_x_max = max(prev_x_max, cell[2])
        lines.append("".join(parts))

    return "\n".join(lines)


def _describe_text_free_image(rgb_arr: np.ndarray, width: int, height: int, fmt: str, mode: str) -> str:
    """Generate a concise factual visual summary when an image has no readable OCR text."""
    orientation = "landscape" if width > height else ("portrait" if height > width else "square")
    mean_rgb = rgb_arr.mean(axis=(0, 1))
    std_rgb = float(rgb_arr.std())
    luma = float(0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2])
    tone = "bright" if luma > 180 else ("dark" if luma < 75 else "mid-tone")
    complexity = "uniform/minimal graphic" if std_rgb < 12.0 else "visual scene or text-free diagram"
    return (
        f"No readable text detected in this image ({orientation} {complexity}, {tone} palette, "
        f"mean RGB ({int(mean_rgb[0])}, {int(mean_rgb[1])}, {int(mean_rgb[2])}), {width}x{height}px {fmt} {mode})."
    )


def parse_image(file_path: Path) -> list[ParsedSection]:
    """Validate image headers via Pillow and extract spatially structured text via RapidOCR."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise ValueError(f"Empty or missing image file: {file_path.name}")

    allowed_formats = {"PNG", "JPEG", "WEBP"}
    try:
        with Image.open(file_path) as probe_img:
            img_format = (probe_img.format or "").upper()
            probe_img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or corrupted image '{file_path.name}': {exc}") from exc

    if img_format not in allowed_formats:
        raise ValueError(
            f"Unsupported image header '{img_format}' in '{file_path.name}' (expected PNG/JPEG/WEBP)."
        )

    try:
        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode
            rgb_arr = np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Failed decoding image '{file_path.name}': {exc}") from exc

    ocr_engine = get_ocr_engine()
    ocr_results, _ = ocr_engine(rgb_arr)
    extracted_text = _reconstruct_spatial_lines(ocr_results) if ocr_results else ""
    alnum_count = sum(1 for ch in extracted_text if ch.isalnum())

    header_line = f"[Image: {file_path.name} | Format: {img_format} | Dimensions: {width}x{height}px]"
    if alnum_count < 3:
        body = _describe_text_free_image(rgb_arr, width, height, img_format, mode)
    else:
        body = extracted_text

    return [
        ParsedSection(
            source_file=file_path.name,
            page_or_section=f"Image ({width}x{height} {img_format})",
            text=f"{header_line}\n{body}",
        )
    ]


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
    if suffix in (".xlsx", ".xlsm"):
        return parse_excel(file_path)
    if suffix in IMAGE_EXTENSIONS:
        return parse_image(file_path)
    raise ValueError(f"Unhandled extension: {suffix}")
