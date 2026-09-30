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
import os
import json
import csv
import datetime
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

from langchain_core.documents import Document
from app.utils.logger import setup_logger
from app.utils.exceptions import DocumentProcessingError

logger = setup_logger(__name__, "INFO")

SUPPORTED_LOCAL_FORMATS = [
    ".pdf", ".docx", ".txt", ".csv", ".json", 
    ".html", ".htm", ".md", ".xlsx", ".tsv",
    ".py", ".js", ".ts", ".sh", ".yaml", ".yml", ".xml",
    ".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif",
    ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"
]
SUPPORTED_FORMATS = SUPPORTED_LOCAL_FORMATS


def _add_metadata(
    documents: List[Document],
    source: str,
    format_type: str,
    extra: Optional[Dict[str, Any]] = None
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
    """Loads a PDF file using pypdf or PyMuPDF (fitz), with multimodal vision OCR fallback for slide decks / image-only PDFs."""
    docs = []
    # 1. Try pypdf text extraction
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        for idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"page": idx + 1, "source": str(file_path)}
                ))
    except Exception as e:
        logger.warning(f"pypdf extraction error: {e}")

    # 2. If text was extracted, return it
    total_chars = sum(len(d.page_content) for d in docs)
    if total_chars > 80:
        return docs

    # 3. Try PyMuPDF (fitz) text extraction
    try:
        import fitz
        doc = fitz.open(str(file_path))
        fitz_docs = []
        for idx, page in enumerate(doc):
            t = page.get_text().strip()
            if t:
                fitz_docs.append(Document(
                    page_content=t,
                    metadata={"page": idx + 1, "source": str(file_path)}
                ))
        if sum(len(d.page_content) for d in fitz_docs) > 80:
            return fitz_docs
    except Exception as e:
        logger.warning(f"fitz text extraction failed: {e}")

    # 4. Multimodal Vision OCR Fallback (for image-based PDFs, slide decks, presentations)
    try:
        import fitz, base64, httpx, concurrent.futures
        from app.config import get_settings
        settings = get_settings()
        if settings.gemini_configured:
            doc = fitz.open(str(file_path))
            logger.info(f"PDF {file_path.name} contains no embedded text. Running multimodal Gemini Vision OCR on {len(doc)} pages...")
            
            def ocr_page(page_idx):
                for attempt in range(3):
                    try:
                        page = doc[page_idx]
                        pix = page.get_pixmap(dpi=85)
                        b64_img = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={settings.GEMINI_API_KEY}"
                        body = {
                            "contents": [{
                                "parts": [
                                    {"inlineData": {"mimeType": "image/jpeg", "data": b64_img}},
                                    {"text": "Transcribe all text from this presentation slide. Format headings with #, bullet points with -, and tables with Markdown syntax. Preserve all quotes, numbers, code snippets, and technical terms."}
                                ]
                            }]
                        }
                        r = httpx.post(url, json=body, timeout=25.0)
                        if r.status_code == 200:
                            content = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                            if content:
                                return Document(
                                    page_content=f"## Slide / Page {page_idx + 1}\n\n{content}",
                                    metadata={"page": page_idx + 1, "source": str(file_path), "format": ".pdf"}
                                )
                        elif r.status_code == 429:
                            import time
                            time.sleep(1.0 * (attempt + 1))
                    except Exception as page_err:
                        import time
                        time.sleep(0.8 * (attempt + 1))
                        logger.debug(f"Vision OCR retry {attempt+1} for page {page_idx + 1}: {page_err}")
                return None

            workers = min(max(len(doc), 1), 8)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                ocr_results = list(executor.map(ocr_page, range(len(doc))))

            valid_ocr_docs = [d for d in ocr_results if d is not None]
            if valid_ocr_docs:
                logger.info(f"Successfully transcribed {len(valid_ocr_docs)} slide pages from {file_path.name}")
                return valid_ocr_docs
    except Exception as vision_err:
        logger.warning(f"Multimodal vision OCR failed for {file_path}: {vision_err}")

    return docs if docs else [Document(page_content="", metadata={"source": str(file_path)})]

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
            return [Document(page_content=f.read(), metadata={"source": str(file_path)})]

def _load_csv_tsv(file_path: Path, delimiter: str = ",") -> List[Document]:
    """Loads CSV or TSV files, serializing each row into structured key-value representations."""
    docs = []
    with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for idx, row in enumerate(reader):
            row_str = " | ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
            if row_str.strip():
                docs.append(Document(
                    page_content=row_str,
                    metadata={"row": idx + 1, "source": str(file_path)}
                ))
    return docs if docs else [Document(page_content="Empty tabular dataset", metadata={"source": str(file_path)})]

def _load_json(file_path: Path) -> List[Document]:
    """Loads JSON documents, handling arrays, records, nested trees, and JSON Lines."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except Exception:
        # Fallback to JSON Lines or raw json blocks
        docs = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        docs.append(Document(page_content=json.dumps(obj, indent=2), metadata={"line": idx + 1, "source": str(file_path)}))
                    except Exception:
                        docs.append(Document(page_content=line, metadata={"line": idx + 1, "source": str(file_path)}))
        return docs or [Document(page_content="Empty JSON dataset", metadata={"source": str(file_path)})]

    if isinstance(data, list):
        docs = []
        for idx, item in enumerate(data):
            content = json.dumps(item, indent=2) if isinstance(item, (dict, list)) else str(item)
            docs.append(Document(page_content=content, metadata={"record_index": idx, "source": str(file_path)}))
        return docs
    elif isinstance(data, dict):
        chunks = []
        for k, v in data.items():
            chunks.append(f"## {k}\n{json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)}")
        return [Document(page_content="\n\n".join(chunks), metadata={"source": str(file_path)})]
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
            return [Document(page_content="\n".join(lines), metadata={"source": str(file_path)})]
    except Exception as e:
        logger.warning(f"BeautifulSoup fallback: {e}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [Document(page_content=f.read(), metadata={"source": str(file_path)})]

def _load_text_or_code(file_path: Path) -> List[Document]:
    """Loads plain text, markdown, yaml, xml, and code source files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": str(file_path)})]

def _load_image(file_path: Path) -> List[Document]:
    """Loads image and extracts structured multimodal visual knowledge."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        from app.tools.media_input import understand_media
        kind, text = understand_media(file_path.name, content)
        return [Document(
            page_content=f"# Image: {file_path.name}\n\n{text}",
            metadata={"source": str(file_path), "format": file_path.suffix.lower(), "title": file_path.name, "source_type": "image"}
        )]
    except Exception as e:
        logger.warning(f"Image load fallback for {file_path}: {e}")
        return [Document(
            page_content=f"[Image attachment: {file_path.name}]",
            metadata={"source": str(file_path), "format": file_path.suffix.lower(), "title": file_path.name, "source_type": "image"}
        )]

def _load_audio(file_path: Path) -> List[Document]:
    """Loads audio file and extracts transcription via multimodal understanding."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        from app.tools.media_input import understand_media
        kind, text = understand_media(file_path.name, content)
        return [Document(
            page_content=f"# Audio Transcription: {file_path.name}\n\n{text}",
            metadata={"source": str(file_path), "format": file_path.suffix.lower(), "title": file_path.name, "source_type": "audio"}
        )]
    except Exception as e:
        logger.warning(f"Audio load fallback for {file_path}: {e}")
        return [Document(
            page_content=f"[Audio recording: {file_path.name}]",
            metadata={"source": str(file_path), "format": file_path.suffix.lower(), "title": file_path.name, "source_type": "audio"}
        )]

def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Auto-detects format from file extension and loads multi-format documents.
    Supports 15+ formats including images, audio, JSON, tables, and documents.
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
        elif ext in [".txt", ".md", ".py", ".js", ".ts", ".sh", ".yaml", ".yml", ".xml"]:
            docs = _load_text_or_code(path_obj)
        elif ext in [".xlsx", ".xls"]:
            import pandas as pd
            df = pd.read_excel(str(path_obj))
            csv_str = df.to_csv(index=False)
            docs = [Document(page_content=csv_str, metadata={"source": str(file_path)})]
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"]:
            docs = _load_image(path_obj)
        elif ext in [".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"]:
            docs = _load_audio(path_obj)
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
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchAssistant/1.0"}
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for el in soup(["script", "style", "nav", "footer", "header"]):
                el.extract()
            text = soup.get_text(separator="\n")
            clean_lines = [l.strip() for l in text.splitlines() if l.strip()]
            doc = Document(
                page_content="\n".join(clean_lines[:500]), # reasonable cap
                metadata={"source_path": url, "format": "web_url", "title": soup.title.string if soup.title else url}
            )
            return [doc]
    except Exception as e:
        logger.warning(f"URL loading failed for {url}: {e}")
        return [Document(page_content=f"Snapshot from {url}", metadata={"source_path": url, "format": "web_url"})]

def fetch_arxiv_papers(query: str, max_results: int = 3) -> List[Document]:
    """Fetches academic paper abstracts from ArXiv API."""
    import httpx
    import xml.etree.ElementTree as ET
    logger.info(f"Fetching academic papers from ArXiv for query: '{query}'")
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        with httpx.Client(timeout=8.0) as client:
            resp = client.get("http://export.arxiv.org/api/query", params=params)
            if resp.status_code != 200:
                return []
            
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            docs = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                id_tag = entry.find("atom:id", ns)
                t_str = title.text.strip().replace("\n", " ") if title is not None else "ArXiv Paper"
                s_str = summary.text.strip().replace("\n", " ") if summary is not None else ""
                url_str = id_tag.text.strip() if id_tag is not None else "arxiv.org"
                docs.append(Document(
                    page_content=f"Title: {t_str}\n\nAbstract: {s_str}",
                    metadata={"source": url_str, "format": "arxiv", "title": t_str, "source_type": "academic"}
                ))
            return docs
    except Exception as e:
        logger.warning(f"ArXiv query failed: {e}")
        return []

def fetch_wikipedia_summary(query: str) -> List[Document]:
    """Fetches encyclopedia summary from Wikipedia API."""
    import httpx
    try:
        with httpx.Client(timeout=6.0) as client:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{httpx.URL(query).raw_path.decode()}"
            resp = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}")
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                if extract:
                    return [Document(
                        page_content=f"Wikipedia [{data.get('title')}]: {extract}",
                        metadata={
                            "source": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{query}"),
                            "format": "wikipedia",
                            "title": data.get("title", query),
                            "source_type": "encyclopedia"
                        }
                    )]
    except Exception as e:
        logger.warning(f"Wikipedia lookup error: {e}")
    return []
