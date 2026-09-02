"""
Agent 1 - Research Planner.
Decomposes complex queries, selects multi-format target sources, and devises
the retrieval strategy via multi-step LLM routing.
"""
import json
import time
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.chains.prompts import PLANNER_SYSTEM_PROMPT
from app.chains.router import ResearchRouter
from app.agents.state import ResearchState
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

class PlannerAgent:
    """Autonomous agent responsible for multi-step query planning and source routing."""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.router = ResearchRouter()
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

    def plan(self, state: ResearchState) -> ResearchState:
        """
        Decomposes query into atomic sub-questions and assigns multi-format retrieval strategies.
        Tracks agent telemetry and routing decisions.
        """
        start_t = time.perf_counter()
        query = state["query"]
        logger.info(f"[PlannerAgent] Formulating multi-step research plan for: '{query[:60]}...'")

        # Step 1: Execute multi-step LLM routing workflow
        routing_decision = self.router.route(
            query=query,
            depth="standard"
        )
        state["routing_metadata"] = {
            "domain": routing_decision.domain,
            "complexity": routing_decision.complexity,
            "recommended_rag_mode": routing_decision.recommended_rag_mode,
            "target_source_formats": routing_decision.target_source_formats,
            "requires_multi_query_expansion": routing_decision.requires_multi_query_expansion
        }
        state["rag_mode"] = state.get("rag_mode") or routing_decision.recommended_rag_mode

        # Step 2: Formulate decomposed sub-questions
        instruction = f"""
        Analyze the query and create a structured research plan.
        Target domain: {routing_decision.domain}. Target formats: {', '.join(routing_decision.target_source_formats)}.
        Respond strictly with a JSON object:
        {{
            "objective": "Clear research objective",
            "sub_questions": ["Atomic sub-question 1", "Atomic sub-question 2", "Atomic sub-question 3"],
            "source_types": {json.dumps(routing_decision.target_source_formats)},
            "priority_order": ["High priority question", "Follow-up question"]
        }}
        """

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            SystemMessage(content=instruction),
            HumanMessage(content=f"Query: {query}")
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            plan_dict = json.loads(content)
            state["research_plan"] = plan_dict
            state["sub_questions"] = plan_dict.get("sub_questions", [query])
            state["status"] = "planned"
        except Exception as e:
            logger.warning(f"[PlannerAgent] Fallback to heuristic decomposition: {e}")
            # Robust deterministic fallback
            sub_qs = [
                f"What are the foundational principles and technical architecture of {query}?",
                f"What are the state-of-the-art benchmarks, performance metrics, and advancements in {query}?",
                f"What are the current limitations, tradeoffs, and future outlook for {query}?"
            ]
            state["research_plan"] = {
                "objective": f"Comprehensive investigation of {query}",
                "sub_questions": sub_qs,
                "source_types": routing_decision.target_source_formats,
                "priority_order": sub_qs
            }
            state["sub_questions"] = sub_qs
            state["status"] = "planned"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        state["agent_telemetry"]["planner_time_ms"] = round(elapsed_ms, 2)
        logger.info(f"[PlannerAgent] Planned {len(state['sub_questions'])} sub-questions in {elapsed_ms:.1f}ms")
        return state
