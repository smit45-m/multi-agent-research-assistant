"""
Shared state schema for LangGraph.
"""
import operator
from typing import TypedDict, Annotated, List, Dict, Any

class ResearchState(TypedDict):
    """Represents the shared state of the research workflow."""
    query: str                              # Original user query
    research_plan: Dict[str, Any]           # Planner's decomposed plan
    sub_questions: List[str]                # Decomposed sub-questions
    retrieved_documents: List[Dict[str, Any]] # Retrieved docs with metadata
    analysis: Dict[str, Any]                # Analyzer's synthesis
    final_report: str                       # Writer's final output
    sources_cited: List[Dict[str, Any]]     # Sources used in report
    confidence_score: float                 # Overall confidence 0-1
    errors: Annotated[List[str], operator.add]  # Accumulated errors
    status: str                             # Current status
    iteration_count: int                    # For cycle detection

def create_initial_state(query: str) -> ResearchState:
    """
    Helper function to create the initial research state.
    
    Args:
        query (str): The original user query.
        
    Returns:
        ResearchState: Initialized state dictionary.
    """
    return {
        "query": query,
        "research_plan": {},
        "sub_questions": [],
        "retrieved_documents": [],
        "analysis": {},
        "final_report": "",
        "sources_cited": [],
        "confidence_score": 0.0,
        "errors": [],
        "status": "initialized",
        "iteration_count": 0
    }
