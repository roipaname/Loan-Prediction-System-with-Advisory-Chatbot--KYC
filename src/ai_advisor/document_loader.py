"""
src/ai_advisor/document_loader.py
==================================
Shared document loading and chunking utilities.

Reads .txt, .md, .pdf, and .docx files from a directory and splits them
into overlapping word-level chunks suitable for both the TF-IDF store and
the ChromaDB vector store.

Public API
----------
  load_documents(directory, chunk_size, chunk_overlap) -> list[dict]

Each returned dict has the form:
  {
    "text"    : str,          — chunk text
    "id"      : str,          — "<stem>_chunk_<index>"
    "metadata": {
        "source"      : str,  — original filename (stem)
        "source_file" : str,  — full filename with extension
        "chunk_index" : int,  — 0-based position within the file
        "total_chunks": int,  — total chunks from this file
    }
  }
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Union

from loguru import logger as log


# ---------------------------------------------------------------------------
# Word-level chunker
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int=512,
    chunk_overlap: int=64,
    source: str='',
) -> List[Dict]:
    """
    Split text into overlapping word-level chunks.

    Parameters
    ----------
    text          : raw document text
    chunk_size    : target chunk length in words
    chunk_overlap : number of words to repeat between consecutive chunks
    source        : filename stem used to build chunk IDs and metadata

    Returns
    -------
    List of chunk dicts with text, id, and metadata.
    """
    words = text.split()
    if not words:
        return []

    step    = max(chunk_size - chunk_overlap, 1)
    starts  = range(0, len(words), step)
    raw_chunks = [" ".join(words[i : i + chunk_size]) for i in starts]

    # Drop any trailing chunks that are almost empty (< 10 words)
    chunks = [c for c in raw_chunks if len(c.split()) >= 10]
    total  = len(chunks)

    return [
        {
            "text": chunk,
            "id":   f"{source}_chunk_{idx}",
            "metadata": {
                "source":       source,
                "source_file":  source,   # overwritten by caller with real extension
                "chunk_index":  idx,
                "total_chunks": total,
            },
        }
        for idx, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# File-type readers
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())
    except Exception as exc:
        log.warning("pypdf failed for %s (%s); falling back to empty string.", path.name, exc)
        return ""


def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore

        doc   = docx.Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)
    except Exception as exc:
        log.warning("python-docx failed for %s (%s); returning empty string.", path.name, exc)
        return ""


_READERS = {
    ".txt":  _read_txt,
    ".md":   _read_txt,
    ".pdf":  _read_pdf,
    ".docx": _read_docx,
}


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_documents(
    directory: Union[str, Path],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Load and chunk all supported documents in a directory.

    Supported file types: .txt, .md, .pdf, .docx

    Parameters
    ----------
    directory    : path to the directory containing strategy documents
    chunk_size   : target chunk length in words
    chunk_overlap: word overlap between consecutive chunks

    Returns
    -------
    List of chunk dicts (text, id, metadata), sorted by source then index.
    Raises ValueError if the directory contains no readable files.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    all_chunks: List[Dict] = []
    file_count = 0

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in _READERS:
            continue

        reader = _READERS[path.suffix.lower()]
        log.info("Loading  %s …", path.name)

        raw_text = reader(path)
        if not raw_text.strip():
            log.warning("  %s: extracted no text; skipping.", path.name)
            continue

        chunks = _chunk_text(
            raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=path.stem,
        )
        # Set the real filename in metadata
        for c in chunks:
            c["metadata"]["source_file"] = path.name

        all_chunks.extend(chunks)
        file_count += 1
        log.info("  %s → %d chunks", path.name, len(chunks))

    log.info(
        "Document loader: %d files, %d total chunks (chunk_size=%d, overlap=%d)",
        file_count, len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks
