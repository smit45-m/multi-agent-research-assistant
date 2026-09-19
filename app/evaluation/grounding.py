"""
Deterministic grounding and coverage measurement utilities.

These functions compute REAL metrics from actual text — no simulated or
calibrated constants. They are shared by the Verifier agent (per-claim
grounding of generated reports) and the benchmark runner (assertion
coverage of expected keyphrases).
"""

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_WORD_RE = re.compile(r"\b\w+\b")

_STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been
    before being below between both but by can did do does doing down during each
    few for from further had has have having he her here hers him his how i if in
    into is it its itself just me more most my no nor not of off on once only or
    other our out over own same she should so some such than that the their them
    then there these they this those through to too under until up very was we
    were what when where which while who whom why will with you your""".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer."""
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> List[str]:
    """Tokens with stopwords removed."""
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, dropping Markdown scaffolding."""
    cleaned = re.sub(r"[#*_`>\[\]|-]", " ", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [p.strip() for p in parts if len(p.strip()) >= 25]


# Report sections that carry pipeline metadata rather than factual claims
# about the research topic (methodology notes, citation lists, confidence
# bookkeeping). Grounding verification targets claim-bearing sections only.
_NON_CLAIM_SECTION_RE = re.compile(
    r"^#{1,6}\s*(methodology|sources?\b|citations?|confidence|references)",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(r"^#{1,6}\s")


def extract_claim_text(report: str) -> str:
    """
    Keep only claim-bearing body text from a Markdown report.

    Drops section headers themselves and the full content of metadata
    sections (Methodology, Sources & Citations, Confidence), which
    describe the pipeline rather than assert facts about the topic.
    """
    kept: List[str] = []
    skipping = False
    for line in report.splitlines():
        if _HEADER_RE.match(line.strip()):
            skipping = bool(_NON_CLAIM_SECTION_RE.match(line.strip()))
            continue  # headers are labels, not claims
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _cosine(a: Counter, b: Counter) -> float:
    """Cosine similarity between two token frequency vectors."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(
        sum(v * v for v in b.values())
    )
    return num / den if den else 0.0


def sentence_grounding(
    sentence: str, source_texts: Sequence[str], threshold: float = 0.18
) -> Tuple[bool, float]:
    """
    Check whether a sentence is grounded in any source text.

    A sentence is grounded when its content tokens either substantially
    overlap a source (token containment) or are distributionally similar
    to a source (cosine similarity of term frequencies).

    Args:
        sentence: The claim/sentence to verify.
        source_texts: Retrieved source passages.
        threshold: Minimum score to count as grounded.

    Returns:
        (grounded, best_score) tuple.
    """
    sent_tokens = content_tokens(sentence)
    if not sent_tokens:
        return True, 1.0  # nothing factual to verify

    sent_counter = Counter(sent_tokens)
    sent_set = set(sent_tokens)
    best = 0.0
    for src in source_texts:
        src_tokens = content_tokens(src)
        if not src_tokens:
            continue
        src_set = set(src_tokens)
        containment = len(sent_set & src_set) / len(sent_set)
        cos = _cosine(sent_counter, Counter(src_tokens))
        score = max(containment * 0.7 + cos * 0.3, cos)
        if score > best:
            best = score
        if best >= 0.95:
            break
    return best >= threshold, round(best, 4)


def measure_report_grounding(
    report: str, source_texts: Sequence[str]
) -> Dict[str, object]:
    """
    Measure what fraction of a report's factual sentences are grounded
    in the retrieved sources.

    Metadata sections (Methodology, Sources & Citations, Confidence) and
    Markdown headers are excluded: they describe the pipeline, not the
    research topic, so they are neither penalized nor credited.

    Returns:
        Dict with grounded_ratio, total_sentences, grounded_sentences,
        and the list of ungrounded sentences (up to 10 for reporting).
    """
    sentences = split_sentences(extract_claim_text(report))
    if not sentences:
        return {
            "grounded_ratio": 0.0,
            "total_sentences": 0,
            "grounded_sentences": 0,
            "ungrounded": [],
        }

    grounded = 0
    ungrounded: List[str] = []
    for sent in sentences:
        ok, _score = sentence_grounding(sent, source_texts)
        if ok:
            grounded += 1
        elif len(ungrounded) < 10:
            ungrounded.append(sent[:160])

    return {
        "grounded_ratio": round(grounded / len(sentences), 4),
        "total_sentences": len(sentences),
        "grounded_sentences": grounded,
        "ungrounded": ungrounded,
    }


def assertion_coverage(text: str, assertions: Sequence[str]) -> Dict[str, object]:
    """
    Measure how many expected assertions (keyphrases) appear in the text.

    An assertion counts as covered when at least 60% of its content tokens
    appear in the text (handles morphological variation via prefix matching).

    Returns:
        Dict with coverage ratio, matched and missing assertion lists.
    """
    if not assertions:
        return {"coverage": 1.0, "matched": [], "missing": []}

    text_tokens = set(tokenize(text))
    matched: List[str] = []
    missing: List[str] = []

    for assertion in assertions:
        a_tokens = content_tokens(assertion) or tokenize(assertion)
        if not a_tokens:
            continue
        hits = 0
        for tok in a_tokens:
            if tok in text_tokens:
                hits += 1
                continue
            # prefix match for morphological variants (quantize/quantization)
            stem = tok[: max(4, len(tok) - 3)]
            if any(t.startswith(stem) for t in text_tokens):
                hits += 1
        if hits / len(a_tokens) >= 0.6:
            matched.append(assertion)
        else:
            missing.append(assertion)

    total = len(matched) + len(missing)
    return {
        "coverage": round(len(matched) / total, 4) if total else 1.0,
        "matched": matched,
        "missing": missing,
    }
