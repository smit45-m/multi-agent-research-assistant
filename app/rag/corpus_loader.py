"""
Reference corpus loader.

Indexes the bundled technical reference corpus (data/corpus/*.md) into the
vector store so the knowledge base is populated out of the box — used by
the benchmark suite and available to the API at startup.
"""

import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from app.rag.vector_store import VectorStoreManager
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_SECTION_RE = re.compile(r"^## ", re.MULTILINE)


def _split_markdown_sections(text: str, source: str) -> List[Document]:
    """
    Split a Markdown reference file into one chunk per '## ' section.

    Corpus files are written as self-contained topic sections (~1-1.5 KB
    each); keeping a section intact in a single chunk preserves the full
    technical context for retrieval instead of cutting topics mid-paragraph.
    """
    parts = _SECTION_RE.split(text)
    docs: List[Document] = []
    for part in parts:
        body = part.strip()
        if len(body) < 80:  # skip the file title/preamble stub
            continue
        title = body.splitlines()[0].strip()
        docs.append(
            Document(
                page_content=f"## {body}" if not body.startswith("#") else body,
                metadata={
                    "source": source,
                    "section": title,
                    "chunk_tier": "markdown_section",
                },
            )
        )
    return docs


def default_corpus_dir() -> Path:
    """Location of the bundled reference corpus."""
    return Path(__file__).parent.parent.parent / "data" / "corpus"


def ensure_corpus_indexed(
    vector_store: VectorStoreManager, corpus_dir: Optional[Path] = None
) -> int:
    """
    Index the reference corpus into the vector store if it is empty.

    Args:
        vector_store: Initialized vector store manager.
        corpus_dir: Corpus directory override (defaults to data/corpus).

    Returns:
        Number of chunks indexed in this call (0 if already populated
        or no corpus files exist).
    """
    corpus_dir = corpus_dir or default_corpus_dir()
    if not corpus_dir.exists():
        return 0

    try:
        existing = vector_store.get_document_count()
    except Exception:  # noqa: BLE001 - treat unknown state as empty
        existing = 0
    if existing > 0:
        logger.info(
            "Vector store already holds %d chunks; skipping corpus indexing.",
            existing,
        )
        return 0

    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        return 0

    all_chunks: List[Document] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
            all_chunks.extend(_split_markdown_sections(text, source=path.name))
        except Exception as exc:  # noqa: BLE001 - skip unreadable files
            logger.warning("Failed to index corpus file %s: %s", path, exc)

    if not all_chunks:
        return 0

    vector_store.add_documents(all_chunks)
    try:
        vector_store.save()
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        logger.warning("Could not persist vector store: %s", exc)

    logger.info(
        "Indexed reference corpus: %d files -> %d chunks.",
        len(files),
        len(all_chunks),
    )
    return len(all_chunks)
