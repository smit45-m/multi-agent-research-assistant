"""
Agent 1 - Research Planner.
"""
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.chains.prompts import PLANNER_SYSTEM_PROMPT
from app.agents.state import ResearchState

class PlannerAgent:
    """Agent responsible for planning the research process."""
    
    def __init__(self, llm=None):
        """
        Initialize the PlannerAgent.
        
        Args:
            llm (Optional[ChatOpenAI]): Language model to use. Defaults to a standard ChatOpenAI instance.
        """
        if llm is None:
            settings = get_settings()
            kwargs = {
                "model": settings.OPENAI_MODEL_NAME,
                "api_key": settings.OPENAI_API_KEY,
                "temperature": 0.3
            }
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            self.llm = ChatOpenAI(**kwargs)
        else:
            self.llm = llm
            
    def plan(self, state: ResearchState) -> ResearchState:
        """
        Decomposes query into sub-questions, identifies source types needed, and creates a retrieval strategy.
        
        Args:
            state (ResearchState): Current research state.
            
        Returns:
            ResearchState: Updated research state with plan and sub-questions.
        """
        query = state["query"]
        
        instruction = """
        Based on the user query, create a research plan.
        Respond with a valid JSON object containing exactly these keys:
        - objective: string explaining the main goal
        - sub_questions: list of string questions to research
        - source_types: list of strings (e.g., 'academic', 'news', 'docs')
        - priority_order: list of strings indicating what to research first
        """
        
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            SystemMessage(content=instruction),
            HumanMessage(content=f"Query: {query}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
                
            plan_dict = json.loads(content)
            
            state["research_plan"] = plan_dict
            state["sub_questions"] = plan_dict.get("sub_questions", [])
            state["status"] = "planned"
            
        except Exception as e:
            state["errors"].append(f"Planner error: {str(e)}")
            state["status"] = "error"
            
        return state
