"""
Multi-Hop Web Navigator & Cross-Site RAG Engine.
Enables agents to:
1. Extract user-supplied URLs from queries.
2. Fetch, scrape, and extract structured text from web pages (preserving tables and lists).
3. Discover outbound links (e.g. from problem aggregators like GrindMap to target platforms like LeetCode or GitHub).
4. Intelligently route to matched child links based on user query intent (multi-hop traversal).
5. Chunk and convert crawled content into LangChain Documents ready for FAISS & BM25 indexing.
"""
import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.rag.chunking import split_documents
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class MultiHopWebNavigator:
    """Autonomous cross-site crawler and multi-hop router for RAG pipelines."""

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extracts and sanitizes all HTTP/HTTPS URLs found in user text."""
        raw_urls = re.findall(r"https?://[^\s<>\"'{}|\\^`\[\]]+", text)
        cleaned = []
        for u in raw_urls:
            # Strip trailing punctuation often appended in conversation (e.g. '.', ',', ')')
            u_clean = re.sub(r"[.,;:!?)\]]+$", "", u)
            if u_clean and u_clean not in cleaned:
                cleaned.append(u_clean)
        return cleaned

    def fetch_leetcode_graphql(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Bypasses Cloudflare on leetcode.com/problems/<slug> by querying LeetCode's public GraphQL API.
        Extracts title, difficulty, full problem statement HTML/text, topic tags, and code hints.
        """
        graphql_url = "https://leetcode.com/graphql"
        query = """
        query getQuestionDetail($titleSlug: String!) {
          question(titleSlug: $titleSlug) {
            questionId
            title
            difficulty
            content
            topicTags {
              name
            }
            hints
          }
        }
        """
        try:
            req_headers = {
                "User-Agent": DEFAULT_USER_AGENT,
                "Content-Type": "application/json",
                "Referer": f"https://leetcode.com/problems/{slug}/",
            }
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    graphql_url,
                    json={"query": query, "variables": {"titleSlug": slug}},
                    headers=req_headers,
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", {}).get("question")
                    if data:
                        raw_html = data.get("content", "") or ""
                        soup = BeautifulSoup(raw_html, "html.parser")
                        clean_text = soup.get_text(separator="\n").strip()
                        tags = [t.get("name") for t in data.get("topicTags", []) if t.get("name")]
                        title = f"LeetCode #{data.get('questionId')}: {data.get('title')} ({data.get('difficulty')})"
                        
                        full_content = (
                            f"# {title}\n"
                            f"**URL:** https://leetcode.com/problems/{slug}/\n"
                            f"**Difficulty:** {data.get('difficulty')}\n"
                            f"**Topics:** {', '.join(tags)}\n\n"
                            f"## Problem Statement\n{clean_text}\n"
                        )
                        hints = data.get("hints", [])
                        if hints:
                            full_content += "\n## Hints\n" + "\n".join(f"- {h}" for h in hints)

                        return {
                            "title": title,
                            "content": full_content,
                            "url": f"https://leetcode.com/problems/{slug}/",
                            "source_type": "leetcode_problem",
                            "outbound_links": [],
                        }
        except Exception as e:
            logger.warning(f"LeetCode GraphQL fetch failed for slug '{slug}': {e}")
        return None

    def fetch_webpage(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetches and parses a webpage, returning readable markdown text and outbound links."""
        # 1. Specialized LeetCode Handler
        lc_match = re.search(r"leetcode\.com/problems/([^/?#]+)", url)
        if lc_match:
            slug = lc_match.group(1).lower().rstrip("/")
            lc_res = self.fetch_leetcode_graphql(slug)
            if lc_res:
                return lc_res

        # 2. General Web Handler via HTTP
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(url, headers=self.headers)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # Remove non-content elements
                for tag in soup(["script", "style", "noscript", "svg"]):
                    # Keep JSON-LD for dataset metadata if present
                    if tag.name == "script" and tag.get("type") == "application/ld+json":
                        continue
                    tag.extract()

                title = soup.title.string.strip() if (soup.title and soup.title.string) else url

                # Extract and format table rows if present (crucial for problem lists and tables)
                tables = soup.find_all("table")
                table_texts = []
                for idx, table in enumerate(tables):
                    rows = []
                    headers = [th.get_text(strip=True) for th in table.find_all("th")]
                    if headers:
                        rows.append(" | ".join(headers))
                        rows.append(" | ".join(["---"] * len(headers)))
                    for tr in table.find_all("tr"):
                        cols = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                        if cols:
                            rows.append(" | ".join(cols))
                    if rows:
                        table_texts.append("\n".join(rows))

                # Extract outbound links with contextual row/surrounding text
                outbound_links = []
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    full_href = urllib.parse.urljoin(url, href)
                    text = a.get_text(" ", strip=True)
                    parent_tr = a.find_parent("tr")
                    context = parent_tr.get_text(" ", strip=True) if parent_tr else ""
                    if full_href.startswith(("http://", "https://")) and full_href != url:
                        outbound_links.append({"url": full_href, "text": text, "context": context})

                # Extract general text
                text_content = soup.get_text(separator="\n")
                clean_lines = [l.strip() for l in text_content.splitlines() if l.strip()]
                body_text = "\n".join(clean_lines[:2500])  # generous ceiling

                if table_texts:
                    body_text += "\n\n### Extracted Structured Tables:\n" + "\n\n".join(table_texts)

                return {
                    "title": title,
                    "content": body_text,
                    "url": url,
                    "source_type": "web_page",
                    "outbound_links": outbound_links,
                }
        except Exception as e:
            logger.warning(f"Failed to fetch webpage '{url}': {e}")
            return None

    def match_child_routes(
        self,
        outbound_links: List[Dict[str, str]],
        query: str,
        max_children: int = 4
    ) -> List[str]:
        """
        Analyzes user query and outbound links from parent page to find target routing links.
        Matches question numbers (e.g. '99', 'question 99'), slugs (e.g. 'n-queens'),
        platform names ('leetcode'), or specific terms in query.
        """
        if not outbound_links:
            return []

        q_lower = query.lower()
        matched = []
        seen_urls = set()

        # Look for explicit problem numbers (e.g. '99', '#99', 'question 99')
        number_matches = re.findall(r"(?:question\s*|#\s*|problem\s*|^|\s)(\d{1,3})(?:\s|$|[.,?!])", q_lower)
        target_numbers = set(number_matches)

        # Look for LeetCode specific routing intent
        wants_leetcode = "leetcode" in q_lower or "problem" in q_lower or "code" in q_lower or bool(target_numbers)

        # 1. First pass: Match target question numbers if index exists in text/href/context
        for link in outbound_links:
            u = link["url"]
            t = link["text"].lower()
            ctx = link.get("context", "").lower()
            if u in seen_urls:
                continue

            # Check if link, text, or row context matches any target number
            for num in target_numbers:
                if (
                    f"/{num}-" in u
                    or f"/{num}/" in u
                    or t.startswith(f"{num}.")
                    or f"#{num}" in t
                    or f"question {num}" in t
                    or re.search(rf"\b{num}\b", ctx)
                ):
                    matched.append(u)
                    seen_urls.add(u)
                    break

        # 2. Second pass: Match keywords from query in anchor text or URL slug
        query_words = [w for w in re.findall(r"\w{4,}", q_lower) if w not in ("question", "problem", "explain", "answer", "stepwise", "microsoft", "grindmap", "company")]
        for link in outbound_links:
            if len(matched) >= max_children:
                break
            u = link["url"]
            t = link["text"].lower()
            if u in seen_urls:
                continue

            # Prioritize LeetCode problems if user mentioned LeetCode
            if wants_leetcode and "leetcode.com/problems/" in u:
                if any(w in u.lower() or w in t for w in query_words):
                    matched.append(u)
                    seen_urls.add(u)
                    continue

        # 3. Third pass: If user asked for LeetCode routing but no specific word matched, pick top high-value links
        if wants_leetcode and len(matched) < max_children:
            for link in outbound_links:
                if len(matched) >= max_children:
                    break
                u = link["url"]
                if u not in seen_urls and "leetcode.com/problems/" in u:
                    matched.append(u)
                    seen_urls.add(u)

        return matched[:max_children]

    def traverse_and_ingest(
        self,
        urls: List[str],
        query: str,
        max_hops: int = 1,
        max_children: int = 4
    ) -> List[Document]:
        """
        Executes multi-hop traversal across supplied root URLs and discovered child sites.
        Splits all retrieved text into chunked Documents with rich provenance metadata.
        """
        raw_docs: List[Document] = []
        visited_urls: Set[str] = set()

        for root_url in urls:
            if root_url in visited_urls:
                continue
            visited_urls.add(root_url)
            logger.info(f"Navigating root web URL: {root_url}")
            root_data = self.fetch_webpage(root_url)
            if not root_data:
                continue

            # Add root page document
            raw_docs.append(
                Document(
                    page_content=root_data["content"],
                    metadata={
                        "source": root_url,
                        "url": root_url,
                        "title": root_data["title"],
                        "hop_level": 0,
                        "source_type": root_data["source_type"],
                    },
                )
            )

            # Route to child pages if multi-hop is enabled
            if max_hops >= 1 and root_data.get("outbound_links"):
                child_urls = self.match_child_routes(
                    root_data["outbound_links"],
                    query,
                    max_children=max_children,
                )
                for c_url in child_urls:
                    if c_url in visited_urls:
                        continue
                    visited_urls.add(c_url)
                    logger.info(f"Traversing multi-hop link -> {c_url}")
                    child_data = self.fetch_webpage(c_url)
                    if child_data:
                        raw_docs.append(
                            Document(
                                page_content=child_data["content"],
                                metadata={
                                    "source": c_url,
                                    "url": c_url,
                                    "parent_url": root_url,
                                    "title": child_data["title"],
                                    "hop_level": 1,
                                    "source_type": child_data["source_type"],
                                },
                            )
                        )

        if not raw_docs:
            return []

        # Split documents into adaptive RAG chunks for FAISS & BM25
        chunked = split_documents(raw_docs, chunk_size=900, chunk_overlap=150, adaptive=True)
        logger.info(f"Web traversal complete: {len(raw_docs)} pages crawled, {len(chunked)} RAG chunks prepared.")
        return chunked
