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


_WIKI_INDEX_RE = re.compile(
    r"^(?:outline of|list of|lists of|index of|timeline of|bibliography of|glossary of|category:|portal:)\b|\(disambiguation\)$",
    re.I,
)


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

    def _fetch_wiki_extract(self, client: httpx.Client, title: str, timeout_s: float) -> str:
        """Fetches plain-text Wikipedia article extract (intro + opening sections) via Action API."""
        try:
            resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": "1",
                    "exchars": "1800",
                    "redirects": "1",
                    "titles": title,
                    "format": "json",
                },
                headers=_WIKI_HEADERS,
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = (page.get("extract") or "").strip()
                    if extract:
                        return extract
        except Exception:
            pass
        return ""

    def _search_wikipedia(self, client: httpx.Client, query: str, max_results: int, budget: float) -> list[dict]:
        """Searches Wikipedia via MediaWiki Search API + batched plain-text extracts, demoting index pages."""
        import html as _html

        candidates = []
        kw8 = _extract_search_keywords(query, 8)
        kw_terms = kw8.split()
        # MediaWiki ANDs all space-separated words; fall back to core terms or OR query on multi-topic prompts
        queries_to_try = [kw8]
        if len(kw_terms) > 3:
            queries_to_try.append(" ".join(kw_terms[:3]))
            queries_to_try.append(" OR ".join(kw_terms[:6]))
        queries_to_try.append(re.sub(r"[^\w\s]", " ", query).strip())

        started_wiki = time.monotonic()
        srlimit = max(max_results * 2, 10)
        for q_try in dict.fromkeys(q for q in queries_to_try if q):
            rem_wiki = budget - (time.monotonic() - started_wiki)
            if rem_wiki <= 0.35:
                break
            try:
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srlimit={srlimit}&srsearch={quote(q_try)}&utf8=&format=json"
                resp = client.get(wiki_url, headers=_WIKI_HEADERS, timeout=min(rem_wiki, 3.5))
                if resp.status_code == 200:
                    items = resp.json().get("query", {}).get("search", [])
                    non_idx = [it for it in items if not _WIKI_INDEX_RE.search((it.get("title") or "").strip())]
                    if non_idx:
                        candidates = items
                        break
                    if items and not candidates:
                        candidates = items
            except (httpx.ConnectTimeout, httpx.ConnectError):
                break
            except Exception:
                continue

        if not candidates:
            return []

        real_items = [it for it in candidates if not _WIKI_INDEX_RE.search((it.get("title") or "").strip())]
        index_items = [it for it in candidates if _WIKI_INDEX_RE.search((it.get("title") or "").strip())]
        ordered_items = real_items + (index_items[:1] if len(real_items) < 2 else [])
        top_titles = [(it.get("title") or "Wikipedia").strip() for it in ordered_items[:max(max_results, 4)]]

        # One batched exintro=1 call populates full multi-paragraph intros cleanly without thread bursts
        extracts = {}
        rem_wiki = budget - (time.monotonic() - started_wiki)
        if top_titles and rem_wiki > 0.4:
            try:
                ex_resp = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "exintro": "1",
                        "explaintext": "1",
                        "exlimit": str(min(len(top_titles), 10)),
                        "redirects": "1",
                        "titles": "|".join(top_titles),
                        "format": "json",
                    },
                    headers=_WIKI_HEADERS,
                    timeout=min(rem_wiki, 2.5),
                )
                if ex_resp.status_code == 200:
                    for page in ex_resp.json().get("query", {}).get("pages", {}).values():
                        t = (page.get("title") or "").strip()
                        ext = (page.get("extract") or "").strip()
                        if t and ext:
                            extracts[t] = ext
            except Exception:
                pass

        scored = []
        for idx, item in enumerate(ordered_items[:max(max_results + 2, 6)]):
            title = (item.get("title") or "Wikipedia").strip()
            snippet_text = _html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", ""))).strip()
            body = extracts.get(title) or snippet_text
            if len(snippet_text) > len(body):
                body = snippet_text
            score = relevance(query, body, title) + max(0.0, (4 - idx) * 0.015)
            scored.append((score, title, body))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_s = scored[0][0] if scored else 0.0
        min_thresh = max(0.24, 0.50 * best_s) if best_s >= 0.30 else 0.18
        selected = ([row for row in scored if row[0] >= min_thresh] or scored[:1])[:max_results]

        # If the top hit has a brief intro (<1100 chars), fetch opening body sections too when budget allows
        rem_wiki = budget - (time.monotonic() - started_wiki)
        if selected and len(selected[0][2]) < 1100 and rem_wiki > 0.7:
            ext_full = self._fetch_wiki_extract(client, selected[0][1], min(rem_wiki, 1.4))
            if len(ext_full) > len(selected[0][2]):
                s0, t0, _ = selected[0]
                selected[0] = (s0, t0, ext_full)

        return [
            {
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                "snippet": body,
                "source_type": "wikipedia",
            }
            for _, title, body in selected if body
        ]

    def _search_ddgs(self, query: str, max_results: int, budget: float) -> list[dict]:
        """Bounded DuckDuckGo web search fallback using concise keywords."""
        if DDGS is None or budget < 1.2:
            return []
        kw_query = _extract_search_keywords(query, 7)

        def _fetch():
            return list(DDGS(timeout=max(2, int(min(budget, 4.0)))).text(kw_query, max_results=max_results))

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(_fetch)
            raw_items = fut.result(timeout=min(budget, 4.5))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
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
                    if relevance(query, r.get("snippet", ""), r.get("title", "")) >= 0.22
                ]
                results.extend(filtered_wiki or [
                    r for r in wiki_hits
                    if relevance(query, r.get("snippet", ""), r.get("title", "")) >= 0.14
                ][:2] or wiki_hits[:1])

                # Comparative queries often name two entities (e.g., Raft vs Paxos, or PostgreSQL vs MySQL)
                rem_cmp = budget - (time.monotonic() - started)
                cmp_sides = []
                m_two = re.search(
                    r"(?:\bcompare\b|\bdifference\s+between\b)\s+([^?.!,;]+?)\s+(?:and|vs\.?|versus|with)\s+([^?.!,;]+)",
                    query,
                    re.I,
                )
                if m_two:
                    cmp_sides = [m_two.group(1), m_two.group(2)]
                else:
                    m_one = re.search(r"(?:compared?\s+(?:to|with)|versus|vs\.?)\s+([^?.!,;]+)", query, re.I)
                    if m_one:
                        cmp_sides = [m_one.group(1)]
                base_terms = [t for t in _extract_search_keywords(query, 4).split() if len(t) >= 4]
                for side in cmp_sides:
                    rem_side = budget - (time.monotonic() - started)
                    if rem_side <= 0.9 or len(results) >= max_results:
                        break
                    side_kw = _extract_search_keywords(side, 3)
                    side_words = [w for w in side_kw.split() if len(w) >= 3]
                    if not side_words:
                        continue
                    titles_low = " ".join((r.get("title") or "").lower() for r in results)
                    if any(w.lower() in titles_low for w in side_words[:1]):
                        continue
                    # Add domain context word for short single-word targets (e.g. 'Paxos' -> 'Paxos consensus')
                    search_q = side_kw
                    if len(side_words) == 1:
                        ctx = next(
                            (
                                t for t in base_terms
                                if t.lower() not in side_kw.lower()
                                and t.lower() not in titles_low
                                and t.lower() not in ("compare", "difference")
                            ),
                            "",
                        )
                        if ctx:
                            search_q = f"{side_kw} {ctx}"
                    extra = self._search_wikipedia(client, search_q, 1, rem_side)
                    for r in extra:
                        if not _WIKI_INDEX_RE.search((r.get("title") or "").strip()) and relevance(query, r.get("snippet", ""), r.get("title", "")) >= 0.14:
                            results.append(r)
            except Exception:
                pass

        rem = budget - (time.monotonic() - started)
        has_rich_wiki = any(len(r.get("snippet", "")) >= 900 for r in results)
        if not has_rich_wiki and len(results) < 2 and rem > 1.5:
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
