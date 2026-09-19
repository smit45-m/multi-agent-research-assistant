"""
CrewAI multi-agent orchestration (optional execution engine).

CrewAI is an OPTIONAL dependency (install via requirements-crewai.txt).
When available, the crew drives generation; the resulting report is then
passed through the same Verifier agent used by the LangGraph engine so
accuracy is measured, not asserted. When CrewAI is not installed, execution
falls back transparently to the LangGraph 6-agent engine.
"""

import time
from typing import Any, Dict, Optional

from app.config import get_settings
from app.rag.vector_store import VectorStoreManager
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class ResearchCrew:
    """Manages CrewAI setup, agent coordination, and verified execution."""

    def __init__(self, vector_store: Optional[VectorStoreManager]):
        self.vector_store = vector_store
        self.settings = get_settings()
        self._initialized = False

    def _setup_crew(self) -> None:
        """Initialize CrewAI agents if the optional dependency is present."""
        if self._initialized:
            return
        if not self.settings.llm_available:
            logger.info("CrewAI requires a configured LLM; using LangGraph engine.")
            return
        try:
            from crewai import Agent  # type: ignore[import-not-found]
            from langchain_openai import ChatOpenAI

            from app.chains.prompts import (
                ANALYZER_SYSTEM_PROMPT,
                PLANNER_SYSTEM_PROMPT,
                RETRIEVER_SYSTEM_PROMPT,
                WRITER_SYSTEM_PROMPT,
            )
            from app.tools.retrieval_tool import create_retrieval_tool
            from app.tools.search_tool import web_search_tool

            kwargs: dict = {
                "model": self.settings.OPENAI_MODEL_NAME,
                "api_key": self.settings.OPENAI_API_KEY,
            }
            if self.settings.OPENAI_BASE_URL:
                kwargs["base_url"] = self.settings.OPENAI_BASE_URL
            self.llm = ChatOpenAI(**kwargs)

            if self.vector_store is None:
                raise RuntimeError("CrewAI engine requires a vector store")
            self.retrieval_tool = create_retrieval_tool(self.vector_store)

            self.planner = Agent(
                role="Research Planner",
                goal=(
                    "Decompose complex queries into actionable research "
                    "plans across multi-format sources."
                ),
                backstory=PLANNER_SYSTEM_PROMPT,
                verbose=False,
                allow_delegation=False,
                llm=self.llm,
            )
            self.retriever = Agent(
                role="Information Retriever",
                goal=(
                    "Retrieve and fuse context using Hybrid RAG (Dense "
                    "FAISS + Sparse BM25 + Reciprocal Rank Fusion)."
                ),
                backstory=RETRIEVER_SYSTEM_PROMPT,
                tools=[self.retrieval_tool, web_search_tool],
                verbose=False,
                allow_delegation=False,
                llm=self.llm,
            )
            self.analyzer = Agent(
                role="Data Analyzer",
                goal=("Cross-verify findings and flag contradictions across sources."),
                backstory=ANALYZER_SYSTEM_PROMPT,
                verbose=False,
                allow_delegation=False,
                llm=self.llm,
            )
            self.writer = Agent(
                role="Report Writer",
                goal=(
                    "Produce a structured research report with inline "
                    "citations grounded in retrieved evidence."
                ),
                backstory=WRITER_SYSTEM_PROMPT,
                verbose=False,
                allow_delegation=False,
                llm=self.llm,
            )
            self._initialized = True
        except ImportError:
            logger.info(
                "CrewAI not installed (optional); using LangGraph engine. "
                "Install with: pip install -r requirements-crewai.txt"
            )
        except Exception as exc:  # noqa: BLE001 - fall back gracefully
            logger.warning("CrewAI initialization failed: %s", exc)

    def _run_crewai(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute the CrewAI process and verify the result. May return None."""
        try:
            from crewai import (  # type: ignore[import-not-found]
                Crew,
                Process,
                Task,
            )

            plan_task = Task(
                description=f"Create a research plan for the query: {query}",
                expected_output="Structured sub-questions and target sources.",
                agent=self.planner,
            )
            retrieve_task = Task(
                description="Retrieve relevant snippets using hybrid search.",
                expected_output="Grounded snippets and sources.",
                agent=self.retriever,
            )
            analyze_task = Task(
                description="Analyze retrieved information; identify themes.",
                expected_output="Synthesized themes with evidence.",
                agent=self.analyzer,
            )
            write_task = Task(
                description=(
                    "Produce the final Markdown report with Executive "
                    "Summary and Citations."
                ),
                expected_output="Complete Markdown report.",
                agent=self.writer,
            )
            crew = Crew(
                agents=[
                    self.planner,
                    self.retriever,
                    self.analyzer,
                    self.writer,
                ],
                tasks=[plan_task, retrieve_task, analyze_task, write_task],
                process=Process.sequential,
                verbose=False,
            )
            start_t = time.perf_counter()
            result = crew.kickoff(inputs={"query": query})
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            report = str(result)

            # Measure accuracy with the same Verifier used by LangGraph.
            from app.agents.state import create_initial_state
            from app.agents.verifier_agent import VerifierAgent

            v_state = create_initial_state(query, orchestrator="crewai")
            v_state["final_report"] = report
            v_state = VerifierAgent().verify(v_state)

            return {
                "final_report": report,
                "sources": [],
                "confidence_score": 0.0,  # CrewAI path exposes no retrieval scores
                "response_accuracy_score": v_state.get("response_accuracy_score", 0.0),
                "synthesis_speedup_ratio": 0.0,
                "verification": v_state.get("verification", {}),
                "agent_telemetry": {
                    "total_latency_ms": round(elapsed_ms, 2),
                    "verifier_time_ms": v_state["agent_telemetry"].get(
                        "verifier_time_ms", 0.0
                    ),
                },
                "metadata": {"process": "CrewAI", "agents_used": 5},
            }
        except Exception as exc:  # noqa: BLE001 - fall back gracefully
            logger.warning(
                "CrewAI execution error, switching to LangGraph engine: %s",
                exc,
            )
            return None

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the workflow, preferring CrewAI when installed and falling
        back to the LangGraph 6-agent engine otherwise.
        """
        self._setup_crew()

        if self._initialized:
            result = self._run_crewai(query)
            if result is not None:
                return result

        # Transparent fallback: LangGraph 6-agent engine.
        from app.agents.graph import ResearchGraph

        graph = ResearchGraph(self.vector_store)
        state = graph.run(query)
        return {
            "final_report": state.get("final_report", ""),
            "sources": state.get("sources_cited", []),
            "confidence_score": state.get("confidence_score", 0.0),
            "response_accuracy_score": state.get("response_accuracy_score", 0.0),
            "synthesis_speedup_ratio": state.get("synthesis_speedup_ratio", 0.0),
            "verification": state.get("verification", {}),
            "critique": state.get("critique", {}),
            "agent_telemetry": state.get("agent_telemetry", {}),
            "metadata": {
                "process": "LangGraph (CrewAI fallback)",
                "agents_used": 6,
            },
        }
