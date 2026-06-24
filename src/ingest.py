"""Document ingestion: load -> clean -> chunk.

This module implements the structured data-cleaning workflow. Raw enterprise
documents are noisy: inconsistent whitespace, boilerplate headers/footers, page
markers, and stray control characters all hurt retrieval quality. We normalise
the text before chunking so the embeddings capture content, not formatting noise.

None of these functions call an external API, so they are fast and unit-testable.
"""

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


# --- Cleaning --------------------------------------------------------------
# Common boilerplate patterns to strip. In a real corpus this list grows to
# match the document sources (intranet exports, PDFs, wiki dumps, etc.).
_BOILERPLATE_PATTERNS = [
    r"(?im)^\s*page \d+ of \d+\s*$",     # "Page 3 of 12"
    r"(?im)^\s*confidential\s*$",         # standalone confidentiality stamps
    r"(?im)^\s*-{3,}\s*$",                # horizontal rule lines
    r"(?im)^\s*\[?internal use only\]?\s*$",
]


def clean_text(text: str) -> str:
    """Normalise a raw document string.

    Steps:
      1. Standardise line endings.
      2. Remove known boilerplate lines.
      3. Strip control characters.
      4. Collapse excessive blank lines and trailing whitespace.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for pattern in _BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text)

    # Remove non-printable control characters except newline and tab.
    text = re.sub(r"[^\S\n\t]+", " ", text)            # collapse runs of spaces
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Collapse 3+ newlines into a paragraph break, trim line whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


# --- Loading ---------------------------------------------------------------
def load_documents(data_dir: Path = config.DATA_DIR) -> List[Document]:
    """Load and clean every .md / .txt file in the data directory.

    Returns a list of LangChain Document objects with cleaned content and a
    `source` metadata field so answers can be traced back to their origin.
    """
    data_dir = Path(data_dir)
    paths = sorted([*data_dir.glob("*.md"), *data_dir.glob("*.txt")])
    if not paths:
        raise FileNotFoundError(f"No .md or .txt documents found in {data_dir}")

    documents: List[Document] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        if cleaned:
            documents.append(
                Document(page_content=cleaned, metadata={"source": path.name})
            )
    return documents


# --- Chunking --------------------------------------------------------------
def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split cleaned documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_chunk(data_dir: Path = config.DATA_DIR) -> List[Document]:
    """Convenience wrapper: load + clean + chunk in one call."""
    return chunk_documents(load_documents(data_dir))
