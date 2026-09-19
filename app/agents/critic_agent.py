"""
Agent 6 - Quality Critic.

Reviews the verified report for structural completeness, evidence coverage,
and grounding quality. Produces actionable revision notes; when quality is
below threshold and the revision budget allows, the graph loops back to the
Writer with the critique attached.
"""

import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.agents.state import ResearchState
from app.utils.llm import build_chat_llm
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

REQUIRED_SECTIONS = [
    "# Executive Summary",
    "# Detailed Findings",
    "# Sources & Citations",
]

MAX_REVISIONS = 1
QUALITY_THRESHOLD = 0.70


class CriticAgent:
    """Autonomous agent that reviews report quality and requests revisions."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm if llm is not None else build_chat_llm(temperature=0.1)

    @staticmethod
    def _structure_score(report: str) -> Dict[str, Any]:
        """Fraction of required sections present in the report."""
        present = [s for s in REQUIRED_SECTIONS if s in report]
        missing = [s for s in REQUIRED_SECTIONS if s not in report]
        score = len(present) / len(REQUIRED_SECTIONS)
        return {"score": round(score, 4), "missing_sections": missing}

    @staticmethod
    def _evidence_score(state: ResearchState) -> float:
        """Score how well the report is backed by retrieved evidence."""
        docs = state.get("retrieved_documents", [])
        sources = state.get("sources_cited", [])
        doc_component = min(len(docs) / 4.0, 1.0)
        source_component = min(len(sources) / 3.0, 1.0)
        return round(0.5 * doc_component + 0.5 * source_component, 4)

    def critique(self, state: ResearchState) -> ResearchState:
        """
        Compute a composite quality score and revision notes.

        quality = 0.4 * grounding + 0.3 * structure + 0.3 * evidence
        """
        start_t = time.perf_counter()
        report = state.get("final_report", "")
        verification = state.get("verification", {})

        structure = self._structure_score(report)
        evidence = self._evidence_score(state)
        grounding = float(verification.get("grounded_ratio", 0.0))

        quality = round(0.4 * grounding + 0.3 * structure["score"] + 0.3 * evidence, 4)

        notes: List[str] = []
        if structure["missing_sections"]:
            notes.append(
                "Add missing sections: " + ", ".join(structure["missing_sections"])
            )
        if grounding < 0.6:
            ungrounded = verification.get("ungrounded_sentences", [])
            if ungrounded:
                notes.append(
                    "Remove or cite these ungrounded claims: "
                    + " | ".join(str(u) for u in ungrounded[:3])
                )
            else:
                notes.append(
                    "Tie claims more tightly to the retrieved source passages."
                )
        if evidence < 0.5:
            notes.append(
                "Evidence base is thin; reference more of the retrieved sources."
            )

        needs_revision = (
            quality < QUALITY_THRESHOLD
            and state.get("revision_count", 0) < MAX_REVISIONS
            and bool(notes)
        )

        state["critique"] = {
            "quality_score": quality,
            "grounding_component": grounding,
            "structure_component": structure["score"],
            "evidence_component": evidence,
            "revision_notes": notes,
            "needs_revision": needs_revision,
            "method": "0.4*grounding + 0.3*structure + 0.3*evidence (all measured)",
        }
        state["status"] = "critiqued"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        state["agent_telemetry"]["critic_time_ms"] = round(elapsed_ms, 2)
        logger.info(
            "[CriticAgent] Quality %.2f (grounding %.2f, structure %.2f, "
            "evidence %.2f) revision=%s in %.1fms",
            quality,
            grounding,
            structure["score"],
            evidence,
            needs_revision,
            elapsed_ms,
        )
        return state
