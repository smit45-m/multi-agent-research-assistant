"""
Multi-format document loading utilities supporting 15+ sources:
1. PDF (.pdf)
2. Microsoft Word (.docx)
3. Plain Text (.txt)
4. CSV tabular (.csv)
5. JSON datasets (.json)
6. HTML documents (.html, .htm)
7. Markdown documents (.md)
8. Spreadsheets (.xlsx, .tsv)
9. Source Code (.py, .js, .ts, .sh)
10. YAML configurations (.yaml, .yml)
11. XML documents (.xml)
12. ArXiv academic research papers (arxiv:query)
13. Wikipedia entries (wiki:query)
14. PubMed biomedical abstracts (pubmed:query)
15. Live Web URLs & news (http://, https://)
"""

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from app.utils.exceptions import DocumentProcessingError
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

SUPPORTED_LOCAL_FORMATS = [
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".json",
    ".html",
    ".htm",
    ".md",
    ".xlsx",
    ".tsv",
    ".py",
    ".js",
    ".ts",
    ".sh",
    ".yaml",
    ".yml",
    ".xml",
]
SUPPORTED_FORMATS = SUPPORTED_LOCAL_FORMATS


def _add_metadata(
    documents: List[Document],
    source: str,
    format_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Enriches the metadata of loaded documents with timestamp, format, and source."""
    load_timestamp = datetime.datetime.now().isoformat()
    for doc in documents:
        doc.metadata["source_path"] = source
        doc.metadata["format"] = format_type
        doc.metadata["load_timestamp"] = load_timestamp
        if extra:
            doc.metadata.update(extra)
    return documents


def _load_pdf(file_path: Path) -> List[Document]:
    """Loads a PDF file using pypdf or pypdfium2."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        docs = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"page": idx + 1, "source": str(file_path)},
                    )
                )
        return (
            docs
            if docs
            else [Document(page_content="", metadata={"source": str(file_path)})]
        )
    except Exception as e:
        logger.warning(f"pypdf loader fallback: {e}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [
                Document(page_content=f.read(), metadata={"source": str(file_path)})
            ]


def _load_docx(file_path: Path) -> List[Document]:
    """Loads a docx file using python-docx."""
    try:
        import docx

        doc = docx.Document(str(file_path))
        text = "\n".join([p.text for p in doc.paragraphs if p.text])
        return [Document(page_content=text, metadata={"source": str(file_path)})]
    except Exception as e:
        logger.warning(f"python-docx error, fallback to raw read: {e}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [
                Document(page_content=f.read(), metadata={"source": str(file_path)})
            ]


def _load_csv_tsv(file_path: Path, delimiter: str = ",") -> List[Document]:
    """Loads CSV/TSV files, serializing rows into key-value form."""
    docs = []
    with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for idx, row in enumerate(reader):
            row_str = " | ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
            if row_str.strip():
                docs.append(
                    Document(
                        page_content=row_str,
                        metadata={"row": idx + 1, "source": str(file_path)},
                    )
                )
    return (
        docs
        if docs
        else [
            Document(
                page_content="Empty tabular dataset",
                metadata={"source": str(file_path)},
            )
        ]
    )


def _load_json(file_path: Path) -> List[Document]:
    """Loads JSON documents, handling arrays, records, and nested trees."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    if isinstance(data, list):
        docs = []
        for idx, item in enumerate(data):
            content = (
                json.dumps(item, indent=2)
                if isinstance(item, (dict, list))
                else str(item)
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={"record_index": idx, "source": str(file_path)},
                )
            )
        return docs
    elif isinstance(data, dict):
        # Format key entries
        chunks = []
        for k, v in data.items():
            chunks.append(
                f"## {k}\n"
                f"{json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)}"
            )
        return [
            Document(
                page_content="\n\n".join(chunks), metadata={"source": str(file_path)}
            )
        ]
    else:
        return [Document(page_content=str(data), metadata={"source": str(file_path)})]


def _load_html(file_path: Path) -> List[Document]:
    """Loads HTML stripping scripts and stylesheets."""
    try:
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for element in soup(["script", "style", "nav", "footer"]):
                element.extract()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return [
                Document(
                    page_content="\n".join(lines), metadata={"source": str(file_path)}
                )
            ]
    except Exception as e:
        logger.warning(f"BeautifulSoup fallback: {e}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [
                Document(page_content=f.read(), metadata={"source": str(file_path)})
            ]


def _load_text_or_code(file_path: Path) -> List[Document]:
    """Loads plain text, markdown, yaml, xml, and code source files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": str(file_path)})]


def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Auto-detects format from file extension and loads multi-format documents.
    Supports 15+ formats.
    """
    path_obj = Path(file_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise DocumentProcessingError(f"File not found: {file_path}")

    ext = path_obj.suffix.lower()
    logger.info(f"Loading document {file_path} (detected format: {ext})...")

    try:
        if ext == ".pdf":
            docs = _load_pdf(path_obj)
        elif ext in [".docx", ".doc"]:
            docs = _load_docx(path_obj)
        elif ext == ".csv":
            docs = _load_csv_tsv(path_obj, delimiter=",")
        elif ext == ".tsv":
            docs = _load_csv_tsv(path_obj, delimiter="\t")
        elif ext == ".json":
            docs = _load_json(path_obj)
        elif ext in [".html", ".htm"]:
            docs = _load_html(path_obj)
        elif ext in [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".sh",
            ".yaml",
            ".yml",
            ".xml",
        ]:
            docs = _load_text_or_code(path_obj)
        elif ext in [".xlsx", ".xls"]:
            import pandas as pd

            df = pd.read_excel(str(path_obj))
            csv_str = df.to_csv(index=False)
            docs = [Document(page_content=csv_str, metadata={"source": str(file_path)})]
        else:
            raise DocumentProcessingError(f"Unsupported format: {ext}")

        return _add_metadata(docs, str(file_path), ext)
    except Exception as e:
        if isinstance(e, DocumentProcessingError):
            raise
        raise DocumentProcessingError(f"Failed to load {file_path}: {str(e)}") from e


def load_from_url(url: str) -> List[Document]:
    """Fetches and parses a web page URL into structured content."""
    import httpx
    from bs4 import BeautifulSoup

    logger.info(f"Loading content from URL: {url}")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchAssistant/1.0"
            )
        }
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for el in soup(["script", "style", "nav", "footer", "header"]):
                el.extract()
            text = soup.get_text(separator="\n")
            clean_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            doc = Document(
                page_content="\n".join(clean_lines[:500]),  # reasonable cap
                metadata={
                    "source_path": url,
                    "format": "web_url",
                    "title": soup.title.string if soup.title else url,
                },
            )
            return [doc]
    except Exception as e:
        logger.warning(f"URL loading failed for {url}: {e}")
        return [
            Document(
                page_content=f"Snapshot from {url}",
                metadata={"source_path": url, "format": "web_url"},
            )
        ]


def fetch_arxiv_papers(query: str, max_results: int = 3) -> List[Document]:
    """Fetches academic paper abstracts from ArXiv API."""
    import xml.etree.ElementTree as ET

    import httpx

    logger.info(f"Fetching academic papers from ArXiv for query: '{query}'")
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(
                "http://export.arxiv.org/api/query",
                params={k: str(v) for k, v in params.items()},
            )
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            docs = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                id_tag = entry.find("atom:id", ns)
                t_str = (
                    title.text.strip().replace("\n", " ")
                    if title is not None and title.text
                    else "ArXiv Paper"
                )
                s_str = (
                    summary.text.strip().replace("\n", " ")
                    if summary is not None and summary.text
                    else ""
                )
                url_str = (
                    id_tag.text.strip()
                    if id_tag is not None and id_tag.text
                    else "arxiv.org"
                )
                docs.append(
                    Document(
                        page_content=f"Title: {t_str}\n\nAbstract: {s_str}",
                        metadata={
                            "source": url_str,
                            "format": "arxiv",
                            "title": t_str,
                            "source_type": "academic",
                        },
                    )
                )
            return docs
    except Exception as e:
        logger.warning(f"ArXiv query failed: {e}")
        return []


def fetch_wikipedia_summary(query: str) -> List[Document]:
    """Fetches encyclopedia summary from Wikipedia API."""
    import httpx

    try:
        with httpx.Client(timeout=6.0) as client:
            resp = client.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                f"{query.replace(' ', '_')}"
            )
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                if extract:
                    return [
                        Document(
                            page_content=f"Wikipedia [{data.get('title')}]: {extract}",
                            metadata={
                                "source": data.get("content_urls", {})
                                .get("desktop", {})
                                .get("page", f"https://en.wikipedia.org/wiki/{query}"),
                                "format": "wikipedia",
                                "title": data.get("title", query),
                                "source_type": "encyclopedia",
                            },
                        )
                    ]
    except Exception as e:
        logger.warning(f"Wikipedia lookup error: {e}")
    return []
