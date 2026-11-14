"""Bounded public search with caching and thread-safe resilience."""
import concurrent.futures
import json
import re
import time
import threading
import warnings
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlparse
import httpx
from langchain_core.tools import tool
from app.config import get_settings
from app.rag.relevance import STOPWORDS, relevance

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

_SEARCH_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 300.0
_WIKI_HEADERS = {"User-Agent": "MultiAgentResearchAssistant/1.0"}
_QUERY_NOISE = {
    "explain", "describe", "compare", "comparison", "differ", "difference", "differences",
    "conventional", "versus", "vs", "between", "work", "works", "working", "used", "using",
    "advantages", "disadvantages", "tradeoffs", "trade-offs", "overview", "summary", "role",
}


def _extract_search_keywords(query: str, max_terms: int = 6) -> str:
    """Extracts core topical keywords from verbose natural-language questions."""
    cleaned = re.sub(r"[^\w\s-]", " ", query).strip()
    terms = [
        tok for tok in cleaned.split()
        if len(tok) > 1
        and tok.lower().strip("-") not in STOPWORDS
        and tok.lower().strip("-") not in _QUERY_NOISE
    ]
    return " ".join(terms[:max_terms]) if terms else cleaned


class WebSearchTool:
    def __init__(self):
        self.last_warning = None

    def _search_wikipedia(self, client: httpx.Client, query: str, max_results: int, budget: float) -> list[dict]:
        """Searches Wikipedia via MediaWiki Search API + REST summary extract with keyword fallback."""
        candidates = []
        kw6 = _extract_search_keywords(query, 6)
        kw3 = _extract_search_keywords(query, 3)
        for q_try in dict.fromkeys([q for q in (kw6, kw3, re.sub(r"[^\w\s]", " ", query).strip()) if q]):
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(q_try)}&utf8=&format=json"
            resp = client.get(wiki_url, headers=_WIKI_HEADERS, timeout=min(budget, 2.5))
            if resp.status_code == 200:
                items = resp.json().get("query", {}).get("search", [])
                if items:
                    candidates = items
                    break

        out = []
        for item in candidates[:max_results]:
            title = item.get("title", "Wikipedia")
            snippet_text = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
            if title and len(out) < 2:
                try:
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'))}"
                    s_resp = client.get(summary_url, headers=_WIKI_HEADERS, timeout=min(budget, 1.5), follow_redirects=True)
                    if s_resp.status_code == 200:
                        extract = (s_resp.json().get("extract") or "").strip()
                        if extract and len(extract) > len(snippet_text):
                            snippet_text = extract
                except Exception:
                    pass
            if snippet_text:
                out.append({
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    "snippet": snippet_text,
                    "source_type": "wikipedia",
                })
        return out

    def _search_ddgs(self, query: str, max_results: int, budget: float) -> list[dict]:
        """Bounded DuckDuckGo web search fallback using concise keywords."""
        if DDGS is None or budget < 1.2:
            return []
        kw_query = _extract_search_keywords(query, 7)

        def _fetch():
            return list(DDGS(timeout=max(2, int(min(budget, 4.0)))).text(kw_query, max_results=max_results))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_fetch)
            raw_items = fut.result(timeout=min(budget, 4.5))
        return [
            {
                "title": (r.get("title") or "Web Result").strip(),
                "url": r.get("href") or r.get("url") or "",
                "snippet": (r.get("body") or r.get("snippet") or "").strip(),
                "source_type": "web",
            }
            for r in raw_items
        ]

    def _search_arxiv(self, client: httpx.Client, query: str, max_results: int, budget: float) -> list[dict]:
        """Searches arXiv Atom API over HTTPS for scientific and systems queries."""
        kw = _extract_search_keywords(query, 5)
        if not kw:
            return []
        arxiv_url = f"https://export.arxiv.org/api/query?search_query=all:{quote(kw)}&max_results={max_results}"
        resp = client.get(arxiv_url, headers=_WIKI_HEADERS, timeout=min(budget, 3.0), follow_redirects=True)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        out = []
        for entry in root.findall("atom:entry", ns)[:max_results]:
            title = " ".join((entry.findtext("atom:title", "", ns) or "").split())
            summary = " ".join((entry.findtext("atom:summary", "", ns) or "").split())
            link = (entry.findtext("atom:id", "", ns) or "").strip()
            if title and summary and link:
                out.append({"title": title, "url": link, "snippet": summary, "source_type": "arxiv"})
        return out

    def search(self, query, max_results=5, timeout=None, wikipedia=True):
        from app.rag.privacy_guard import is_air_gapped
        from app.chains.llm import get_shared_client
        if is_air_gapped():
            self.last_warning = "Air-gapped privacy mode active: external web search is disabled."
            return []

        settings = get_settings()
        budget = timeout if timeout is not None else settings.SEARCH_TIMEOUT_SECONDS
        self.last_warning = None

        cache_key = f"{query.strip().lower()}:{max_results}"
        now = time.time()
        with _CACHE_LOCK:
            if cache_key in _SEARCH_CACHE:
                cached_time, cached_results = _SEARCH_CACHE[cache_key]
                if now - cached_time < _CACHE_TTL_SECONDS:
                    return [dict(r) for r in cached_results]

        results = []
        client = get_shared_client()
        started = time.monotonic()

        try:
            if settings.SEARCH_API_KEY:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": settings.SEARCH_API_KEY, "query": query, "max_results": max_results, "search_depth": "basic"},
                    timeout=min(budget, 3.0),
                )
                if response.status_code == 200:
                    results = [
                        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", ""), "source_type": "web"}
                        for r in response.json().get("results", [])
                    ]
        except Exception:
            results = []

        if len(results) < 2 and wikipedia and (budget - (time.monotonic() - started)) > 0.6:
            try:
                wiki_hits = self._search_wikipedia(client, query, max_results, budget - (time.monotonic() - started))
                filtered_wiki = [
                    r for r in wiki_hits
                    if relevance(query, r.get("snippet", ""), r.get("title", "")) >= 0.12
                ]
                results.extend(filtered_wiki or wiki_hits[:1])
            except Exception:
                pass

        rem = budget - (time.monotonic() - started)
        if len(results) < 2 and rem > 1.2:
            try:
                ddgs_hits = self._search_ddgs(query, max_results, rem)
                results.extend([
                    r for r in ddgs_hits
                    if relevance(query, r.get("snippet", ""), r.get("title", "")) >= 0.12
                ] or ddgs_hits[:2])
            except Exception:
                pass

        rem = budget - (time.monotonic() - started)
        if not results and rem > 0.8:
            try:
                results.extend(self._search_arxiv(client, query, max_results, rem))
            except Exception:
                pass

        if not results:
            self.last_warning = "Web search was unavailable or rate-limited. No search results were fabricated."

        seen_urls = set()
        cleaned = []
        for r in results:
            url = (r.get("url") or "").strip()
            if r.get("snippet", "").strip() and urlparse(url).scheme in {"http", "https"} and url not in seen_urls:
                seen_urls.add(url)
                cleaned.append(r)
                if len(cleaned) >= max_results:
                    break
        if cleaned:
            with _CACHE_LOCK:
                _SEARCH_CACHE[cache_key] = (now, cleaned)
        return cleaned


@tool
def web_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search public web sources for the supplied question."""
    return WebSearchTool().search(query, max_results)
