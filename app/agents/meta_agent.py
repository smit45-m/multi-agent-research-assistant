"""
MetaAgent / OrchestrationManager:
Oversees the multi-agent lifecycle, dynamically manages and balances agents,
and automatically optimizes configurations between:
- Mode 1: Quality Answer + Low Latency (<5s SLA)
- Mode 2: Detailed Researched Answer (Deep Literature Review & Verification)

Selects vector store strategies, chunk sizes (450 vs 1200), chunk overlaps (60 vs 250),
retrieval algorithms (Hybrid, Agentic, Vectorless, Hierarchical, Multi-Query, Dense, BM25),
and engines (LangGraph vs CrewAI).
"""
import re
from typing import Dict, Any, Tuple
from app.config import get_settings
from app.agents.jev_engine import JEVDecisionEngine


class MetaAgent:
    """
    Central Orchestration Manager that balances and governs the other agents,
    dynamically tuning chunking, vector stores, RAG modes, and model backends
    using the JEV (Joint Expected Value) Decision Framework.
    """

    def __init__(self):
        self.settings = get_settings()
        self.jev_engine = JEVDecisionEngine()

    def analyze_and_configure(
        self,
        query: str,
        mode: str = "auto",
        depth: str = None,
        rag_mode: str = "auto",
        orchestrator: str = "auto",
        corpus_size: int = 0,
        has_attachments: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates the complete adaptive execution profile for the query
        grounded in TypeSafe AI's Jev (System One AI Decision Model).
        """
        # Execute Jev System One decision optimization across candidates
        jev_plan = self.jev_engine.optimize_pipeline_config(
            query=query,
            mode=mode,
            depth=depth,
            rag_mode=rag_mode,
            orchestrator=orchestrator,
            corpus_size=corpus_size,
            has_multimodal=has_attachments
        )

        resolved_mode = jev_plan["target_mode"]
        chunk_size = jev_plan["chunk_size"]
        chunk_overlap = jev_plan["chunk_overlap"]
        selected_rag = jev_plan["selected_rag_mode"]
        selected_engine = jev_plan["selected_orchestrator"]
        latency_sla = jev_plan["sla_ceiling_s"]
        llm_model = self.settings.model_for(resolved_mode)

        return {
            "resolved_mode": resolved_mode,
            "target_latency_sla_s": latency_sla,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "selected_rag_mode": selected_rag,
            "selected_orchestrator": selected_engine,
            "recommended_vector_store": jev_plan["recommended_vector_store"],
            "llm_model": llm_model,
            "jev_score": jev_plan.get("jev_score", 0.95),
            "jev_decision": jev_plan.get("jev_decision", "jev_sys1_decision"),
            "expected_quality": jev_plan.get("expected_quality", 0.88),
            "estimated_latency_s": jev_plan.get("estimated_latency_s", 2.0),
            "model_source": jev_plan.get("model_source", "jev_system_one"),
            "candidate_evaluations": jev_plan.get("candidate_evaluations", []),
            "enable_fact_checker": (resolved_mode == "research" or has_attachments),
            "enable_supervisor": True,
            "summary_rationale": jev_plan.get("rationale", "")
        }
