"""Transparent lexical relevance diagnostics, not calibrated factual confidence."""
import hashlib
import math
import re
from collections import Counter

STOPWORDS = frozenset("a an the is are was were be been being do does did how what which who where when why can could should would will may might of for to in on at by with from and or as that this these those it its i me my we our you your please explain describe give tell about into than then also some any have has had using use used based discuss detailed detail example examples practical brief briefly current latest answer question research compare comparison versus vs differences difference between each their they them both include including provide information overview understand need want make more most less very".split())


def stem(word: str) -> str:
    word = word.lower().strip()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in ("ing", "tion", "tions", "ment", "ments", "ness", "able", "ible", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    raw = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
    return [stem(t) for t in raw if t not in STOPWORDS and len(t) > 1]


def fingerprint(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip()).encode("utf-8")).hexdigest()


def relevance(query: str, text: str, title: str = "") -> float:
    q = set(tokens(query))
    if not q:
        return 0.0
    body = set(tokens(text))
    heading = set(tokens(title))
    overlap = len(q & body) / len(q)
    title_overlap = len(q & heading) / len(q)
    clean_query = " ".join(tokens(query))
    phrase = float(bool(clean_query) and clean_query in " ".join(tokens(title + " " + text)))
    return min(1.0, 0.72 * overlap + 0.18 * title_overlap + 0.1 * phrase)


def grounded_sentences(query: str, text: str, limit: int = 4) -> list[str]:
    """Extract actual sentences. Never invent a summary when no evidence exists."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) >= 20]
    ranked = sorted(enumerate(sentences), key=lambda item: relevance(query, item[1]), reverse=True)
    selected = sorted((i, s) for i, s in ranked[:limit] if relevance(query, s) > 0)
    return [s for _, s in selected]


def evidence_diagnostics(report: str, sources: list[dict]) -> dict:
    cited = [int(i) for i in re.findall(r"\[(\d+)\]", report)]
    valid = set(range(1, len(sources) + 1))
    units = [s for s in report.splitlines() if s.strip() and not s.lstrip().startswith(("#", ">", "---"))]
    claims = [s for s in units if len(s) > 35]
    covered = sum(bool(re.search(r"\[\d+\]", s)) for s in claims)
    return {
        "citation_validity": round(sum(i in valid for i in cited) / len(cited), 4) if cited else None,
        "citation_coverage": round(covered / len(claims), 4) if claims else None,
        "evidence_count": len(sources),
        "invalid_citations": sorted(set(cited) - valid),
        "note": "Citation validity checks source IDs, not whether each claim is true. Coverage is a line-level heuristic; factual accuracy is not independently evaluated.",
    }
