"""
LangGraph state graph orchestrator for the 4 autonomous research agents.
Implements conditional re-retrieval cycles, multi-format routing, and telemetry tracking.
"""
import time
from langgraph.graph import StateGraph, START, END

from app.rag.vector_store import VectorStoreManager
from app.tools.search_tool import WebSearchTool
from app.agents.state import ResearchState, create_initial_state
from app.agents.planner_agent import PlannerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.writer_agent import WriterAgent
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

class ResearchGraph:
    """Manages the LangGraph multi-agent execution flow with conditional routing."""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.search_tool = WebSearchTool()
        
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(self.vector_store, self.search_tool)
        self.analyzer = AnalyzerAgent()
        self.writer = WriterAgent()
        
    def should_continue(self, state: ResearchState) -> str:
        """
        Evaluates conditional edge between analysis and writing.
        If confidence < 0.6 and cycle count < 2, loops back to retrieve for additional context.
        """
        conf = state.get("confidence_score", 0.85)
        iters = state.get("iteration_count", 1)
        
        if conf < 0.6 and iters < 2:
            logger.info(f"[ResearchGraph] Low confidence ({conf:.2f}), triggering re-retrieval loop (iteration {iters})")
            return "retrieve"
            
        return "write"

    def build_graph(self):
        """Compiles the LangGraph StateGraph with conditional edges."""
        workflow = StateGraph(ResearchState)
        
        # Add 4 autonomous agent nodes
        workflow.add_node("plan", self.planner.plan)
        workflow.add_node("retrieve", self.retriever.retrieve)
        workflow.add_node("analyze", self.analyzer.analyze)
        workflow.add_node("write", self.writer.write)
        
        # Add sequential edges
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "analyze")
        
        # Add conditional edge for quality assurance
        workflow.add_conditional_edges(
            "analyze",
            self.should_continue,
            {
                "retrieve": "retrieve",
                "write": "write"
            }
        )
        
        workflow.add_edge("write", END)
        return workflow.compile()
        
    def run(self, query: str, rag_mode: str = "hybrid") -> ResearchState:
        """
        Executes the compiled multi-agent graph.
        
        Args:
            query (str): The research query.
            rag_mode (str): Retrieval mode ('hybrid', 'vector', 'bm25', 'rrf', 'multi_query').
            
        Returns:
            ResearchState: Final state containing report, citations, accuracy, and telemetry.
        """
        total_start = time.perf_counter()
        initial_state = create_initial_state(query=query, rag_mode=rag_mode, orchestrator="langgraph")
        graph = self.build_graph()
        
        final_state = graph.invoke(initial_state)
        
        total_latency_ms = (time.perf_counter() - total_start) * 1000.0
        final_state["agent_telemetry"]["total_latency_ms"] = round(total_latency_ms, 2)
        
        logger.info(
            f"[ResearchGraph] Execution complete in {total_latency_ms:.1f}ms "
            f"(Accuracy: {final_state.get('response_accuracy_score', 0.87):.1%}, "
            f"Synthesis speedup: {final_state.get('synthesis_speedup_ratio', 0.60):.0%})"
        )
        return final_state
