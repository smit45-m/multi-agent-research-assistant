"""
Agent 5 - Fact Verifier.

Verifies the written report against the actually retrieved sources by
measuring per-sentence grounding. The response_accuracy_score it produces
is a REAL measurement (fraction of report sentences traceable to retrieved
evidence, blended with citation integrity) — never a hardcoded constant.
"""

import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.agents.state import ResearchState
from app.evaluation.grounding import measure_report_grounding
from app.utils.llm import build_chat_llm
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class VerifierAgent:
    """Autonomous agent that fact-checks reports against retrieved evidence."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm if llm is not None else build_chat_llm(temperature=0.0)

    @staticmethod
    def _citation_integrity(
        report: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Measure citation integrity: what fraction of cited sources actually
        exist in the retrieval set, and whether the report cites at all.
        """
        if not sources:
            return {"cited_ratio": 0.0, "citations_found": 0}

        citations_found = 0
        for src in sources:
            title = str(src.get("title", ""))[:60]
            url = str(src.get("source", src.get("url_or_path", "")))
            if (title and title in report) or (url and url in report):
                citations_found += 1

        return {
            "cited_ratio": round(citations_found / len(sources), 4),
            "citations_found": citations_found,
        }

    def verify(self, state: ResearchState) -> ResearchState:
        """
        Verify the report and compute a measured accuracy score.

        accuracy = 0.75 * grounded_sentence_ratio + 0.25 * citation_integrity
        """
        start_t = time.perf_counter()
        report = state.get("final_report", "")
        docs = state.get("retrieved_documents", [])
        sources = state.get("sources_cited", [])

        source_texts = [d.get("content", "") for d in docs if d.get("content")]

        grounding = measure_report_grounding(report, source_texts)
        citation = self._citation_integrity(report, sources)

        grounded_ratio = float(str(grounding["grounded_ratio"]))
        cited_ratio = float(citation["cited_ratio"])
        accuracy = round(0.75 * grounded_ratio + 0.25 * cited_ratio, 4)

        state["verification"] = {
            "grounded_ratio": grounded_ratio,
            "total_sentences": grounding["total_sentences"],
            "grounded_sentences": grounding["grounded_sentences"],
            "ungrounded_sentences": grounding["ungrounded"],
            "citation_integrity": cited_ratio,
            "citations_found": citation["citations_found"],
            "measured_accuracy": accuracy,
            "method": (
                "0.75 * grounded_sentence_ratio + 0.25 * citation_integrity; "
                "grounding = token containment/cosine vs retrieved passages"
            ),
        }
        state["response_accuracy_score"] = accuracy
        state["status"] = "verified"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        state["agent_telemetry"]["verifier_time_ms"] = round(elapsed_ms, 2)
        logger.info(
            "[VerifierAgent] Measured accuracy %.1f%% "
            "(grounded %s/%s sentences, citation integrity %.0f%%) in %.1fms",
            accuracy * 100.0,
            grounding["grounded_sentences"],
            grounding["total_sentences"],
            cited_ratio * 100.0,
            elapsed_ms,
        )
        return state
