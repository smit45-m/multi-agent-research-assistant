"""
Agent 4 - Report Writer.
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.chains.prompts import WRITER_SYSTEM_PROMPT
from app.agents.state import ResearchState

class WriterAgent:
    """Agent responsible for writing the final report."""
    
    def __init__(self, llm=None):
        """
        Initialize the WriterAgent.
        
        Args:
            llm (Optional[ChatOpenAI]): Language model.
        """
        if llm is None:
            settings = get_settings()
            kwargs = {
                "model": settings.OPENAI_MODEL_NAME,
                "api_key": settings.OPENAI_API_KEY,
                "temperature": 0.4
            }
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            self.llm = ChatOpenAI(**kwargs)
        else:
            self.llm = llm

    def write(self, state: ResearchState) -> ResearchState:
        """
        Generates structured research report based on analysis and retrieved docs.
        
        Args:
            state (ResearchState): Current research state.
            
        Returns:
            ResearchState: Updated state with final_report and sources_cited.
        """
        query = state.get("query", "")
        analysis = state.get("analysis", {})
        docs = state.get("retrieved_documents", [])
        
        # Extract unique sources
        sources = {doc["source"] for doc in docs}
        sources_list = [{"url": s, "type": "reference"} for s in sources]
        
        instruction = """
        Write a comprehensive markdown report for the user's query based on the analysis.
        The report MUST include the following sections:
        - # Executive Summary
        - # Detailed Findings
        - # Methodology
        - # Sources Cited
        - # Confidence Assessment
        """
        
        context_msg = f"""
        Query: {query}
        
        Analysis:
        {analysis}
        
        Available Sources:
        {sources}
        """
        
        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            SystemMessage(content=instruction),
            HumanMessage(content=context_msg)
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["final_report"] = response.content
            state["sources_cited"] = sources_list
            state["status"] = "completed"
        except Exception as e:
            state["errors"].append(f"Writer error: {str(e)}")
            state["status"] = "error"
            
        return state
