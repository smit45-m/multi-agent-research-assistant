"""
Agent 3 - Data Analyzer.

Analyzes and cross-references retrieved multi-source context, detects
contradictions, and scores source reliability. Confidence is MEASURED from
retrieval relevance, source diversity, and content volume — never hardcoded.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.state import ResearchState
from app.chains.prompts import ANALYZER_SYSTEM_PROMPT
from app.utils.llm import build_chat_llm, extract_json_block
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_TRUSTED_TLDS = (".edu", ".gov", ".org")
_TRUSTED_HOSTS = ("arxiv.org", "pubmed.ncbi.nlm.nih.gov", "wikipedia.org")


def _source_reliability(source: str) -> float:
    """
    Heuristic reliability score for a source identifier based on domain
    reputation signals. Purely deterministic and explainable.
    """
    host = urlparse(source).netloc.lower() if "://" in source else source.lower()
    if any(t in host for t in _TRUSTED_HOSTS):
        return 0.9
    if any(host.endswith(t) for t in _TRUSTED_TLDS):
        return 0.85
    if host.startswith("http") or "." in host:
        return 0.7
    return 0.6  # local/internal documents


class AnalyzerAgent:
    """Autonomous agent for cross-source validation and synthesis."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm if llm is not None else build_chat_llm(temperature=0.15)

    @staticmethod
    def _measure_confidence(docs: List[Dict[str, Any]]) -> float:
        """
        Measure analysis confidence from real retrieval signals:
        - mean retrieval relevance of the used passages (60%)
        - source diversity: distinct sources vs. passages used (20%)
        - content volume: whether passages carry substantive text (20%)
        """
        if not docs:
            return 0.0

        relevances = [float(d.get("relevance_score", 0.0)) for d in docs]
        mean_rel = sum(relevances) / len(relevances)

        distinct = len({d.get("source", "") for d in docs})
        diversity = min(distinct / 3.0, 1.0)

        substantive = sum(1 for d in docs if len(str(d.get("content", ""))) >= 200)
        volume = substantive / len(docs)

        return round(0.6 * mean_rel + 0.2 * diversity + 0.2 * volume, 4)

    def _heuristic_synthesis(
        self, docs: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        """Deterministic synthesis used when no LLM is available."""
        grouped: Dict[str, List[str]] = {}
        for d in docs:
            src = d.get("source", "knowledge_base")
            grouped.setdefault(src, []).append(str(d.get("content", "")))

        key_findings: List[str] = []
        reliability: Dict[str, float] = {}
        for src, contents in list(grouped.items())[:8]:
            reliability[src] = _source_reliability(src)
            merged = " ".join(" ".join(contents).split())
            sample = merged[:400]
            if len(merged) > 400 and " " in sample:
                sample = sample.rsplit(" ", 1)[0]  # cut at a word boundary
            label = os.path.basename(str(src)) or str(src)
            if len(sample) > 30:
                key_findings.append(f"Evidence from {label}: {sample}")

        return {
            "key_findings": key_findings
            or [f"No substantive evidence retrieved for '{query}'."],
            # Contradiction analysis needs an LLM; report none rather than
            # emitting an unverifiable meta-sentence into the report body.
            "contradictions": [],
            "source_reliability": reliability,
            "themes": sorted(grouped.keys())[:5],
        }

    def analyze(self, state: ResearchState) -> ResearchState:
        """Analyze retrieved documents and record measured telemetry."""
        start_t = time.perf_counter()
        docs = state.get("retrieved_documents", [])
        query = state.get("query", "")
        logger.info("[AnalyzerAgent] Synthesizing %d retrieved passages...", len(docs))

        if not docs:
            state["analysis"] = {
                "key_findings": [],
                "contradictions": [],
                "source_reliability": {},
                "themes": [],
            }
            state["confidence_score"] = 0.0
            state["status"] = "analyzed"
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            state["agent_telemetry"]["analyzer_time_ms"] = round(elapsed_ms, 2)
            return state

        top_docs = docs[:8]
        analysis_dict: Optional[Dict[str, Any]] = None

        if self.llm is not None:
            context_str = json.dumps(
                [
                    {
                        "source": d.get("source"),
                        "title": d.get("title"),
                        "snippet": str(d.get("content", ""))[:350],
                    }
                    for d in top_docs
                ],
                indent=2,
            )
            instruction = (
                "Cross-reference the provided documents for the query. "
                "Identify key findings, cross-source contradictions, source "
                "reliability, and themes. Respond strictly with JSON: "
                '{"key_findings": [...], "contradictions": [...], '
                '"source_reliability": {"source": 0.9}, "themes": [...]}'
            )
            messages = [
                SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
                SystemMessage(content=instruction),
                HumanMessage(
                    content=f"Query: {query}\n\nRetrieved Passages:\n{context_str}"
                ),
            ]
            try:
                response = self.llm.invoke(messages)
                analysis_dict = json.loads(extract_json_block(str(response.content)))
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning(
                    "[AnalyzerAgent] LLM analysis failed (%s); using "
                    "heuristic synthesis.",
                    exc,
                )

        if analysis_dict is None:
            analysis_dict = self._heuristic_synthesis(top_docs, query)

        # Confidence is measured from retrieval signals, not asserted.
        confidence = self._measure_confidence(top_docs)

        state["analysis"] = analysis_dict
        state["confidence_score"] = confidence
        state["status"] = "analyzed"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        state["agent_telemetry"]["analyzer_time_ms"] = round(elapsed_ms, 2)
        logger.info(
            "[AnalyzerAgent] Synthesis completed in %.1fms (measured confidence: %.2f)",
            elapsed_ms,
            confidence,
        )
        return state
