"""Tests for the bundled reference corpus loader."""

from pathlib import Path

from app.rag.corpus_loader import (
    _split_markdown_sections,
    default_corpus_dir,
    ensure_corpus_indexed,
)


class FakeVectorStore:
    """Minimal stand-in capturing added documents."""

    def __init__(self, existing: int = 0):
        self._existing = existing
        self.added: list = []
        self.saved = False

    def get_document_count(self) -> int:
        return self._existing

    def add_documents(self, documents):  # type: ignore[no-untyped-def]
        self.added.extend(documents)
        return [str(i) for i in range(len(documents))]

    def save(self) -> None:
        self.saved = True


def test_default_corpus_dir_exists_and_has_domain_files():
    corpus = default_corpus_dir()
    assert corpus.exists()
    files = list(corpus.glob("*.md"))
    # One reference file per major benchmark domain group.
    assert len(files) >= 7


def test_split_markdown_sections_keeps_topics_intact():
    text = (
        "# Title\n\n"
        "## First Topic\n\n"
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho sigma tau upsilon.\n\n"
        "## Second Topic\n\n"
        "One two three four five six seven eight nine ten eleven "
        "twelve thirteen fourteen fifteen sixteen seventeen.\n"
    )
    docs = _split_markdown_sections(text, source="sample.md")
    assert len(docs) == 2
    assert docs[0].metadata["section"] == "First Topic"
    assert "Alpha beta gamma" in docs[0].page_content
    assert docs[1].metadata["source"] == "sample.md"


def test_ensure_corpus_indexed_populates_empty_store():
    store = FakeVectorStore(existing=0)
    count = ensure_corpus_indexed(store)  # type: ignore[arg-type]
    assert count > 0
    assert len(store.added) == count
    assert store.saved


def test_ensure_corpus_indexed_skips_populated_store():
    store = FakeVectorStore(existing=42)
    count = ensure_corpus_indexed(store)  # type: ignore[arg-type]
    assert count == 0
    assert store.added == []


def test_ensure_corpus_indexed_handles_missing_dir(tmp_path: Path):
    store = FakeVectorStore(existing=0)
    count = ensure_corpus_indexed(
        store,  # type: ignore[arg-type]
        corpus_dir=tmp_path / "does-not-exist",
    )
    assert count == 0
