"""
Agent 1 - Research Planner.

Decomposes complex queries, selects multi-format target sources, and devises
the retrieval strategy. Uses the LLM when configured; otherwise falls back to
deterministic decomposition (clearly reported as such in the plan metadata).
"""

import json
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.state import ResearchState
from app.chains.prompts import PLANNER_SYSTEM_PROMPT
from app.chains.router import ResearchRouter
from app.utils.llm import build_chat_llm, extract_json_block
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class PlannerAgent:
    """Autonomous agent for multi-step query planning and source routing."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.router = ResearchRouter()
        self.llm = llm if llm is not None else build_chat_llm(temperature=0.2)

    def plan(self, state: ResearchState) -> ResearchState:
        """
        Decompose query into sub-questions and assign retrieval strategies.
        Tracks agent telemetry and routing decisions.
        """
        start_t = time.perf_counter()
        query = state["query"]
        logger.info("[PlannerAgent] Formulating research plan for: '%s...'", query[:60])

        routing_decision = self.router.route(query=query, depth="standard")
        state["routing_metadata"] = {
            "domain": routing_decision.domain,
            "complexity": routing_decision.complexity,
            "recommended_rag_mode": routing_decision.recommended_rag_mode,
            "target_source_formats": routing_decision.target_source_formats,
            "requires_multi_query_expansion": (
                routing_decision.requires_multi_query_expansion
            ),
        }
        state["rag_mode"] = (
            state.get("rag_mode") or routing_decision.recommended_rag_mode
        )

        plan_dict = None
        if self.llm is not None:
            instruction = (
                "Analyze the query and create a structured research plan. "
                f"Target domain: {routing_decision.domain}. Target formats: "
                f"{', '.join(routing_decision.target_source_formats)}. "
                "Respond strictly with a JSON object: "
                '{"objective": "...", "sub_questions": ["...", "...", "..."], '
                f'"source_types": '
                f"{json.dumps(routing_decision.target_source_formats)}, "
                '"priority_order": ["...", "..."]}'
            )
            messages = [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                SystemMessage(content=instruction),
                HumanMessage(content=f"Query: {query}"),
            ]
            try:
                response = self.llm.invoke(messages)
                plan_dict = json.loads(extract_json_block(str(response.content)))
                plan_dict["planning_mode"] = "llm"
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning(
                    "[PlannerAgent] LLM planning failed (%s); using "
                    "heuristic decomposition.",
                    exc,
                )

        if plan_dict is None:
            sub_qs = [
                f"What are the foundational principles and technical "
                f"architecture of {query}?",
                f"What are the state-of-the-art benchmarks, performance "
                f"metrics, and advancements in {query}?",
                f"What are the current limitations, tradeoffs, and future "
                f"outlook for {query}?",
            ]
            plan_dict = {
                "objective": f"Comprehensive investigation of {query}",
                "sub_questions": sub_qs,
                "source_types": routing_decision.target_source_formats,
                "priority_order": sub_qs,
                "planning_mode": "heuristic",
            }

        state["research_plan"] = plan_dict
        state["sub_questions"] = plan_dict.get("sub_questions", [query])
        state["status"] = "planned"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        state["agent_telemetry"]["planner_time_ms"] = round(elapsed_ms, 2)
        logger.info(
            "[PlannerAgent] Planned %d sub-questions in %.1fms (%s mode)",
            len(state["sub_questions"]),
            elapsed_ms,
            plan_dict.get("planning_mode", "heuristic"),
        )
        return state
