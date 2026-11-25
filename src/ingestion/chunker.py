"""Framework-free recursive character text splitter with overlap."""

from __future__ import annotations

import hashlib
from typing import Sequence

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


def _split_with_separator(text: str, separator: str) -> list[str]:
    """Split text while keeping trailing punctuation/newlines attached to the preceding piece."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    if len(parts) == 1:
        return parts
    # Keep separator attached to each piece except the last so sentences/lines stay readable
    result: list[str] = []
    for piece in parts[:-1]:
        combined = piece + separator
        if combined:
            result.append(combined)
    if parts[-1]:
        result.append(parts[-1])
    return result


def recursive_split_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Recursively split text into overlapping chunks <= chunk_size characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    # Pick the highest-priority separator present in text
    chosen_sep = ""
    next_separators: Sequence[str] = ()
    for idx, sep in enumerate(separators):
        if sep == "" or sep in cleaned:
            chosen_sep = sep
            next_separators = separators[idx + 1 :]
            break

    raw_splits = _split_with_separator(cleaned, chosen_sep)

    # Recursively break down any piece that still exceeds chunk_size
    atomic_pieces: list[str] = []
    for piece in raw_splits:
        if len(piece) <= chunk_size:
            atomic_pieces.append(piece)
        elif next_separators:
            atomic_pieces.extend(
                recursive_split_text(
                    piece,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=next_separators,
                )
            )
        else:
            # Hard slice fallback if no separators remain
            step = chunk_size - chunk_overlap
            for start in range(0, len(piece), step):
                atomic_pieces.append(piece[start : start + chunk_size])

    # Merge atomic pieces into chunks with overlap
    chunks: list[str] = []
    current_pieces: list[str] = []
    current_len = 0

    for piece in atomic_pieces:
        piece_len = len(piece)
        if current_len + piece_len > chunk_size and current_pieces:
            chunk_str = "".join(current_pieces).strip()
            if chunk_str:
                chunks.append(chunk_str)

            # Retain trailing pieces up to chunk_overlap characters while keeping >=1 piece if possible
            while len(current_pieces) > 1 and (
                current_len > chunk_overlap or current_len + piece_len > chunk_size
            ):
                removed = current_pieces.pop(0)
                current_len -= len(removed)

            # If a single trailing piece exceeds chunk_overlap, slice a word-aligned trailing window
            if len(current_pieces) == 1 and current_len > chunk_overlap:
                tail = current_pieces[0][-chunk_overlap:]
                space_idx = tail.find(" ")
                if 0 <= space_idx < len(tail) - 1:
                    tail = tail[space_idx + 1 :]
                tail = tail.strip()
                if tail and len(tail) + 1 + piece_len <= chunk_size:
                    current_pieces = [tail + " "]
                    current_len = len(current_pieces[0])
                else:
                    current_pieces.clear()
                    current_len = 0

            # Final guard if leftover overlap still prevents fitting the incoming piece
            if current_pieces and current_len + piece_len > chunk_size:
                current_pieces.clear()
                current_len = 0

        current_pieces.append(piece)
        current_len += piece_len

    if current_pieces:
        final_chunk = "".join(current_pieces).strip()
        if final_chunk and (not chunks or final_chunk != chunks[-1]):
            chunks.append(final_chunk)

    return chunks


def make_chunk_id(source_file: str, chunk_index: int, text: str) -> str:
    """Create a deterministic unique chunk ID from filename, index, and content hash."""
    digest = hashlib.sha1(f"{source_file}::{chunk_index}::{text}".encode("utf-8")).hexdigest()[:10]
    return f"{source_file}::chunk_{chunk_index:04d}_{digest}"
