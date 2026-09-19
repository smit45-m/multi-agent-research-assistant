"""
LangGraph state graph orchestrator for the 6 autonomous research agents.

Pipeline:
    plan -> retrieve -> analyze -> (re-retrieve loop) -> write
         -> verify -> critique -> (revise loop back to write) -> END

Implements conditional re-retrieval cycles, a verification/critique quality
loop, and measured telemetry tracking.
"""

import time
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.critic_agent import CriticAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.state import ResearchState, create_initial_state
from app.agents.verifier_agent import VerifierAgent
from app.agents.writer_agent import WriterAgent
from app.rag.vector_store import VectorStoreManager
from app.tools.search_tool import WebSearchTool
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class ResearchGraph:
    """Manages the LangGraph multi-agent execution flow with quality loops."""

    def __init__(self, vector_store: Optional[VectorStoreManager]):
        self.vector_store = vector_store
        self.search_tool = WebSearchTool()

        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(self.vector_store, self.search_tool)
        self.analyzer = AnalyzerAgent()
        self.writer = WriterAgent()
        self.verifier = VerifierAgent()
        self.critic = CriticAgent()

        self._compiled: Optional[Any] = None

    def should_continue(self, state: ResearchState) -> str:
        """
        Conditional edge between analysis and writing.
        If confidence < 0.4 and cycle count < 2, loop back to retrieval.
        """
        conf = state.get("confidence_score", 0.0)
        iters = state.get("iteration_count", 1)

        if conf < 0.4 and iters < 2:
            logger.info(
                "[ResearchGraph] Low confidence (%.2f), re-retrieval loop "
                "(iteration %d)",
                conf,
                iters,
            )
            return "retrieve"
        return "write"

    def should_revise(self, state: ResearchState) -> str:
        """
        Conditional edge after critique: revise the report when the critic
        requests it (bounded by the critic's revision budget).
        """
        if state.get("critique", {}).get("needs_revision"):
            logger.info("[ResearchGraph] Critic requested a revision")
            return "write"
        return "end"

    def build_graph(self) -> Any:
        """Compile the LangGraph StateGraph with conditional edges."""
        if self._compiled is not None:
            return self._compiled

        workflow = StateGraph(ResearchState)

        # 6 autonomous agent nodes
        workflow.add_node("plan", self.planner.plan)
        workflow.add_node("retrieve", self.retriever.retrieve)
        workflow.add_node("analyze", self.analyzer.analyze)
        workflow.add_node("write", self.writer.write)
        workflow.add_node("verify", self.verifier.verify)
        workflow.add_node("critique", self.critic.critique)

        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "analyze")

        # Quality loop 1: confidence-gated re-retrieval
        workflow.add_conditional_edges(
            "analyze",
            self.should_continue,
            {"retrieve": "retrieve", "write": "write"},
        )

        workflow.add_edge("write", "verify")
        workflow.add_edge("verify", "critique")

        # Quality loop 2: critic-gated revision
        workflow.add_conditional_edges(
            "critique",
            self.should_revise,
            {"write": "write", "end": END},
        )

        self._compiled = workflow.compile()
        return self._compiled

    def run(self, query: str, rag_mode: str = "hybrid") -> ResearchState:
        """
        Execute the compiled multi-agent graph.

        Args:
            query: The research query.
            rag_mode: Retrieval mode ('hybrid', 'vector', 'bm25', 'rrf',
                'multi_query').

        Returns:
            Final state containing report, citations, measured accuracy,
            verification details, critique, and telemetry.
        """
        total_start = time.perf_counter()
        initial_state = create_initial_state(
            query=query, rag_mode=rag_mode, orchestrator="langgraph"
        )
        graph = self.build_graph()

        final_state: ResearchState = graph.invoke(initial_state)
        final_state["status"] = "completed"

        total_latency_ms = (time.perf_counter() - total_start) * 1000.0
        final_state["agent_telemetry"]["total_latency_ms"] = round(total_latency_ms, 2)

        logger.info(
            "[ResearchGraph] Execution complete in %.1fms "
            "(measured accuracy: %.1f%%, quality: %.2f, speedup: %.0f%%)",
            total_latency_ms,
            final_state.get("response_accuracy_score", 0.0) * 100.0,
            final_state.get("critique", {}).get("quality_score", 0.0),
            final_state.get("synthesis_speedup_ratio", 0.0) * 100.0,
        )
        return final_state
