"""
Agent 3 - Analyzer.
"""
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.chains.prompts import ANALYZER_SYSTEM_PROMPT
from app.agents.state import ResearchState

class AnalyzerAgent:
    """Agent responsible for analyzing retrieved information."""
    
    def __init__(self, llm=None):
        """
        Initialize the AnalyzerAgent.
        
        Args:
            llm (Optional[ChatOpenAI]): Language model to use.
        """
        if llm is None:
            settings = get_settings()
            kwargs = {
                "model": settings.OPENAI_MODEL_NAME,
                "api_key": settings.OPENAI_API_KEY,
                "temperature": 0.2
            }
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            self.llm = ChatOpenAI(**kwargs)
        else:
            self.llm = llm

    def analyze(self, state: ResearchState) -> ResearchState:
        """
        Cross-references retrieved docs, identifies contradictions, synthesizes findings.
        
        Args:
            state (ResearchState): Current research state.
            
        Returns:
            ResearchState: Updated state with analysis dict.
        """
        docs = state.get("retrieved_documents", [])
        
        if not docs:
            state["analysis"] = {
                "key_findings": [],
                "contradictions": [],
                "source_reliability": {},
                "themes": [],
                "confidence_score": 0.0
            }
            state["status"] = "analyzed"
            return state
            
        # Prepare context
        context_str = json.dumps(docs[:10]) # Limit to top 10 for context window
        
        instruction = """
        Analyze the provided documents. Respond with a valid JSON object containing exactly these keys:
        - key_findings: list of strings representing the main points
        - contradictions: list of strings noting any conflicting information between sources
        - source_reliability: dict mapping source URLs/names to a score 0-10
        - themes: list of strings (overarching topics)
        - confidence_score: float (0.0 to 1.0) indicating how confident you are that the query can be answered with this data
        """
        
        messages = [
            SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
            SystemMessage(content=instruction),
            HumanMessage(content=f"Query: {state['query']}\\nDocuments: {context_str}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
                
            analysis_dict = json.loads(content)
            
            state["analysis"] = analysis_dict
            state["confidence_score"] = float(analysis_dict.get("confidence_score", 0.5))
            state["status"] = "analyzed"
            
        except Exception as e:
            state["errors"].append(f"Analyzer error: {str(e)}")
            state["status"] = "error"
            
        return state
