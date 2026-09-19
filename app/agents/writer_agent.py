"""
Agent 4 - Report Writer.

Synthesizes findings into structured Markdown research reports with inline
citations. Accuracy scoring is NOT done here — the Verifier agent measures
grounding after writing. When a Critic revision is requested, the writer
incorporates the revision notes.
"""

import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.state import ResearchState
from app.chains.prompts import WRITER_SYSTEM_PROMPT
from app.utils.llm import build_chat_llm
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class WriterAgent:
    """Autonomous agent responsible for report synthesis."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm if llm is not None else build_chat_llm(temperature=0.25)

    @staticmethod
    def _compile_sources(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate retrieved documents into a citation list."""
        unique_sources: List[Dict[str, Any]] = []
        seen: set = set()
        for doc in docs:
            src = doc.get("source", "knowledge_base")
            if src not in seen:
                seen.add(src)
                unique_sources.append(
                    {
                        "title": doc.get("title", src),
                        "source": src,
                        "url_or_path": src,
                        "relevance_score": float(doc.get("relevance_score", 0.0)),
                        "source_type": doc.get("source_type", "document"),
                        "snippet": str(doc.get("content", ""))[:200],
                    }
                )
        return unique_sources

    @staticmethod
    def _template_report(
        query: str,
        analysis: Dict[str, Any],
        sources: List[Dict[str, Any]],
        confidence: float,
    ) -> str:
        """
        Deterministic evidence-quoting report used when no LLM is available.

        Quotes retrieved passages directly (keeping the report grounded in
        actual evidence) and clearly labels itself as extractive.
        """
        key_findings = analysis.get("key_findings", [])
        contradictions = analysis.get("contradictions", [])
        themes = analysis.get("themes", [])

        findings_md = (
            "\n".join(f"- {kf}" for kf in key_findings)
            if key_findings
            else "- No substantive findings could be extracted from the "
            "retrieved evidence."
        )
        sources_md = "\n".join(
            f"- **[{s.get('title', 'Reference')}]({s.get('source')})** "
            f"({s.get('source_type', 'document')}, retrieval relevance: "
            f"{s.get('relevance_score', 0.0):.2f})\n  > {s.get('snippet', '')}"
            for s in sources[:8]
        )
        themes_md = "\n".join(f"- {t}" for t in themes) if themes else "- Not derived."
        contradictions_md = (
            "\n".join(f"- {c}" for c in contradictions)
            if contradictions
            else "- Not evaluated."
        )

        return f"""# Executive Summary
Extractive research summary for the query: **"{query}"**.

# Detailed Findings
{findings_md}

## Themes
{themes_md}

# Multi-Source Cross-Verification & Contradictions
{contradictions_md}

# Methodology & Retrieval Architecture
- Generated in extractive mode (no LLM configured): findings quote
  retrieved source passages directly rather than paraphrasing them.
- Multi-agent pipeline: Planner, Retriever, Analyzer, Writer, Verifier, Critic.
- Hybrid retrieval: dense FAISS embeddings + sparse BM25, fused with
  Reciprocal Rank Fusion (RRF, k=60), with optional multi-query expansion.
- Measured retrieval confidence for this run: {confidence:.2f}.

# Sources & Citations
{sources_md if sources_md else "- No external sources were retrieved."}

# Confidence & Accuracy Assessment
- Retrieval confidence (measured): {confidence:.2f}
- Factual accuracy: measured post-hoc by the Verifier agent; see the
  `verification` metadata attached to this task's response.
"""

    def write(self, state: ResearchState) -> ResearchState:
        """Synthesize the research report and compile structured citations."""
        start_t = time.perf_counter()
        query = state.get("query", "")
        analysis = state.get("analysis", {})
        docs = state.get("retrieved_documents", [])
        confidence = state.get("confidence_score", 0.0)
        critique = state.get("critique", {})

        is_revision = bool(critique.get("needs_revision"))
        if is_revision:
            state["revision_count"] = state.get("revision_count", 0) + 1
            logger.info(
                "[WriterAgent] Revising report (revision %d) with critic notes",
                state["revision_count"],
            )
        else:
            logger.info(
                "[WriterAgent] Drafting research report for: '%s...'",
                query[:50],
            )

        unique_sources = self._compile_sources(docs)
        report_text: Optional[str] = None

        if self.llm is not None:
            instruction = (
                "Write an authoritative research report in Markdown with "
                "these section headers:\n"
                "# Executive Summary\n# Detailed Findings\n"
                "# Multi-Source Cross-Verification & Contradictions\n"
                "# Methodology & Retrieval Architecture\n"
                "# Sources & Citations\n"
                "# Confidence & Accuracy Assessment\n\n"
                "Ground EVERY claim in the provided sources with inline "
                "citations [Source: ...]. Never invent facts or citations."
            )
            revision_msg = ""
            if is_revision and critique.get("revision_notes"):
                revision_msg = (
                    "\nREVISION REQUEST - address these reviewer notes:\n"
                    + "\n".join(f"- {n}" for n in critique["revision_notes"])
                )
            context_msg = (
                f"Query: {query}\n"
                f"Measured retrieval confidence: {confidence:.2f}\n"
                f"Analysis: {analysis}\n"
                f"Sources: {unique_sources[:8]}"
                f"{revision_msg}"
            )
            messages = [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                SystemMessage(content=instruction),
                HumanMessage(content=context_msg),
            ]
            try:
                response = self.llm.invoke(messages)
                report_text = str(response.content).strip()
                if "# Executive Summary" not in report_text:
                    report_text = f"# Executive Summary\n{report_text}"
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning(
                    "[WriterAgent] LLM writing failed (%s); using extractive template.",
                    exc,
                )

        if report_text is None:
            report_text = self._template_report(
                query=query,
                analysis=analysis,
                sources=unique_sources,
                confidence=confidence,
            )

        state["final_report"] = report_text
        state["sources_cited"] = unique_sources
        state["status"] = "written"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        tel = state["agent_telemetry"]
        tel["writer_time_ms"] = round(tel.get("writer_time_ms", 0.0) + elapsed_ms, 2)
        logger.info("[WriterAgent] Report completed in %.1fms", elapsed_ms)
        return state
