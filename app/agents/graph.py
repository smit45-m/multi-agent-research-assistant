"""
LangGraph state graph workflow for the research process.
"""
from langgraph.graph import StateGraph, START, END

from app.rag.vector_store import VectorStoreManager
from app.tools.search_tool import WebSearchTool
from app.agents.state import ResearchState, create_initial_state
from app.agents.planner_agent import PlannerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.writer_agent import WriterAgent

class ResearchGraph:
    """Manages the LangGraph execution flow."""
    
    def __init__(self, vector_store: VectorStoreManager):
        """
        Initialize the ResearchGraph and its agents.
        
        Args:
            vector_store (VectorStoreManager): Vector store for document retrieval.
        """
        self.vector_store = vector_store
        self.search_tool = WebSearchTool()
        
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(self.vector_store, self.search_tool)
        self.analyzer = AnalyzerAgent()
        self.writer = WriterAgent()
        
    def should_continue(self, state: ResearchState) -> str:
        """
        Routing function for conditional edges.
        Determines whether to write the report or retrieve more information.
        
        Args:
            state (ResearchState): Current research state.
            
        Returns:
            str: Name of the next node.
        """
        # If we have low confidence and haven't exceeded iteration limits
        if state.get("confidence_score", 1.0) < 0.6 and state.get("iteration_count", 0) < 3:
            return "retrieve"
            
        return "write"

    def build_graph(self):
        """
        Build and compile the LangGraph StateGraph.
        
        Returns:
            CompiledGraph: The compiled runnable graph.
        """
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("plan", self.planner.plan)
        workflow.add_node("retrieve", self.retriever.retrieve)
        workflow.add_node("analyze", self.analyzer.analyze)
        workflow.add_node("write", self.writer.write)
        
        # Add edges
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "analyze")
        
        # Add conditional edge
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
        
    def run(self, query: str) -> ResearchState:
        """
        Invoke the compiled graph for a query.
        
        Args:
            query (str): The research query.
            
        Returns:
            ResearchState: Final state containing the report and metadata.
        """
        initial_state = create_initial_state(query)
        graph = self.build_graph()
        
        # Run graph
        final_state = graph.invoke(initial_state)
        return final_state
