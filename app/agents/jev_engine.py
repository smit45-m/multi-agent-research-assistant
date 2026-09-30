"""
JEV (Joint Expected Value) Decision Engine.

Provides mathematically grounded multi-criteria decision optimization across
multi-agent research layers:
- Dynamic RAG retrieval mode selection (Hybrid, BM25, Multi-Query, Hierarchical, Agentic, Vectorless)
- Vector database / store selection (FAISS in-memory, BM25 inverted index, Knowledge Graph, Hierarchical)
- Chunk size and overlap tuning (Compact 400/50 for <5s vs 1200/250 for Deep Research vs 600/100 for Privacy)
- Model backend routing (gemini-flash-lite-latest vs gemini-3.5-flash vs local air-gapped)
- Orchestration engine selection (Direct single-pass vs LangGraph DAG)
- Supervisory arbitration (Immediate deterministic polish vs full LLM elevation pass)

Mathematical formulation:
    JEV(a) = w_q * Quality(a) + w_l * LatencyUtility(a) + w_g * Grounding(a) + w_p * Privacy(a) - Cost(a)
"""
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class JEVWeights(BaseModel):
    weight_quality: float = 0.35
    weight_latency: float = 0.35
    weight_grounding: float = 0.20
    weight_privacy: float = 0.10
    cost_penalty: float = 0.05


class JEVDecisionResult(BaseModel):
    selected_option: str
    target_mode: str
    jev_score: float
    expected_quality: float
    expected_latency_s: float
    expected_grounding: float
    privacy_compliance: float
    utility_breakdown: Dict[str, float]
    all_evaluated_options: List[Dict[str, Any]] = Field(default_factory=list)
    decision_rationale: str


class JEVDecisionEngine:
    """
    Joint Expected Value Decision Engine for Multi-Agent RAG Orchestration.
    """

    def __init__(self, weights: Optional[JEVWeights] = None):
        self.settings = get_settings()
        self.weights = weights or JEVWeights()

    def _get_mode_weights(self, mode: str) -> JEVWeights:
        """Adapts multi-criteria weights dynamically according to operational mode."""
        if mode in ("fast", "quick"):
            return JEVWeights(
                weight_quality=0.25,
                weight_latency=0.50,
                weight_grounding=0.20,
                weight_privacy=0.05,
                cost_penalty=0.02
            )
        elif mode in ("research", "deep"):
            return JEVWeights(
                weight_quality=0.45,
                weight_latency=0.15,
                weight_grounding=0.30,
                weight_privacy=0.10,
                cost_penalty=0.05
            )
        elif mode == "privacy":
            return JEVWeights(
                weight_quality=0.25,
                weight_latency=0.25,
                weight_grounding=0.20,
                weight_privacy=0.30,
                cost_penalty=0.01
            )
        return self.weights

    def optimize_pipeline_config(
        self,
        query: str,
        mode: str = "auto",
        depth: Optional[str] = None,
        rag_mode: str = "auto",
        orchestrator: str = "auto",
        corpus_size: int = 0,
        has_multimodal: bool = False
    ) -> Dict[str, Any]:
        """
        Uses JEV to determine the globally optimal pipeline configuration:
        - Mode resolution (fast vs research vs privacy)
        - Vector store / Index architecture
        - Chunk size & overlap
        - RAG retrieval strategy
        - Orchestration engine
        - Primary & fallback LLM models
        """
        q_lower = query.lower()
        words = query.split()

        # Step 1: Resolve mode
        if mode == "privacy" or depth == "privacy":
            target_mode = "privacy"
        elif mode in ("fast", "quick") or depth == "quick":
            target_mode = "fast"
        elif mode in ("research", "deep") or depth == "deep":
            target_mode = "research"
        else:
            # Auto mode: classify by cognitive complexity
            cognitive_verbs = len(re.findall(r"\b(compare|versus|analy[sz]e|evaluate|justify|trade-offs?|limitations?)\b", q_lower))
            is_deep = len(words) > 15 or cognitive_verbs >= 1 or any(
                term in q_lower for term in ["comprehensive", "detailed", "literature review", "in-depth", "architecture"]
            )
            target_mode = "research" if is_deep else "fast"

        weights = self._get_mode_weights(target_mode)

        # Step 2: Define candidate pipeline configurations
        candidates = [
            {
                "id": "fast_direct_hybrid",
                "name": "Fast Direct Hybrid (FAISS L2 + BM25Okapi + Gemini Flash Lite)",
                "target_mode": "fast",
                "vector_store": "faiss_inmemory_normalized + bm25_inverted",
                "chunk_size": 400,
                "chunk_overlap": 50,
                "rag_mode": "hybrid",
                "orchestrator": "direct",
                "model": "gemini-flash-lite-latest",
                "est_latency_s": 2.2,
                "base_quality": 0.88,
                "grounding": 0.88,
                "privacy": 0.50 if target_mode != "privacy" else 0.0,
                "cost": 0.02
            },
            {
                "id": "fast_direct_bm25",
                "name": "Fast Direct BM25 (Exact Token Inverted Index)",
                "target_mode": "fast",
                "vector_store": "bm25_inverted_index",
                "chunk_size": 350,
                "chunk_overlap": 40,
                "rag_mode": "bm25",
                "orchestrator": "direct",
                "model": "gemini-flash-lite-latest",
                "est_latency_s": 1.9,
                "base_quality": 0.84,
                "grounding": 0.89,
                "privacy": 0.50 if target_mode != "privacy" else 0.0,
                "cost": 0.01
            },
            {
                "id": "research_langgraph_multiquery",
                "name": "Deep Research LangGraph Multi-Query (Decomposed Parallel RAG)",
                "target_mode": "research",
                "vector_store": "faiss_dense + hierarchical_context + bm25",
                "chunk_size": 1200,
                "chunk_overlap": 250,
                "rag_mode": "multi_query",
                "orchestrator": "langgraph",
                "model": "gemini-3.5-flash",
                "est_latency_s": 18.0,
                "base_quality": 0.96,
                "grounding": 0.94,
                "privacy": 0.50 if target_mode != "privacy" else 0.0,
                "cost": 0.08
            },
            {
                "id": "research_langgraph_hierarchical",
                "name": "Deep Research LangGraph Hierarchical (Parent-Document Tree)",
                "target_mode": "research",
                "vector_store": "hierarchical_parent_child_store",
                "chunk_size": 1400,
                "chunk_overlap": 300,
                "rag_mode": "hierarchical",
                "orchestrator": "langgraph",
                "model": "gemini-3.5-flash",
                "est_latency_s": 16.5,
                "base_quality": 0.94,
                "grounding": 0.95,
                "privacy": 0.50 if target_mode != "privacy" else 0.0,
                "cost": 0.07
            },
            {
                "id": "research_langgraph_agentic",
                "name": "Deep Research Agentic Corrective RAG (CRAG Self-Evaluation)",
                "target_mode": "research",
                "vector_store": "faiss_dense + vectorless_graph",
                "chunk_size": 1100,
                "chunk_overlap": 200,
                "rag_mode": "agentic",
                "orchestrator": "langgraph",
                "model": "gemini-3.5-flash",
                "est_latency_s": 19.5,
                "base_quality": 0.95,
                "grounding": 0.96,
                "privacy": 0.50 if target_mode != "privacy" else 0.0,
                "cost": 0.09
            },
            {
                "id": "privacy_airgapped_local",
                "name": "Air-Gapped Privacy Guardian (Local In-Memory RAG + Zero Cloud Egress)",
                "target_mode": "privacy",
                "vector_store": "local_sanitized_memory_store + local_bm25",
                "chunk_size": 600,
                "chunk_overlap": 100,
                "rag_mode": "hybrid",
                "orchestrator": "direct",
                "model": "llama3.2_local_or_inprocess_airgapped",
                "est_latency_s": 2.1,
                "base_quality": 0.88,
                "grounding": 0.96,
                "privacy": 1.00,
                "cost": 0.00
            }
        ]

        # Explicit user overrides take priority if specified
        sla_ceiling = 5.0 if target_mode == "fast" else 30.0 if target_mode == "research" else 4.0

        evaluated = []
        for c in candidates:
            # Latency utility score: normalized relative to SLA ceiling
            if c["target_mode"] != target_mode:
                # Mode mismatch penalty
                mode_penalty = 0.50
            else:
                mode_penalty = 0.0

            # Latency utility
            if c["est_latency_s"] <= sla_ceiling:
                lat_util = max(0.0, 1.0 - (c["est_latency_s"] / (sla_ceiling * 1.5)))
            else:
                lat_util = max(0.0, 0.5 * (sla_ceiling / c["est_latency_s"]))

            # Quality adjustment based on query features
            quality = c["base_quality"]
            if c["rag_mode"] == "bm25" and re.search(r'"[^\"]+"|\b[A-Z]{2,}[-_]\d+|\b\w+_\w+\b', query):
                quality += 0.05
            if c["rag_mode"] in ("multi_query", "agentic") and ("compare" in q_lower or "vs" in q_lower):
                quality += 0.06
            if c["rag_mode"] == "hierarchical" and (corpus_size > 1000 or has_multimodal):
                quality += 0.05
            quality = min(1.0, quality)

            # Compute JEV
            jev = (
                weights.weight_quality * quality +
                weights.weight_latency * lat_util +
                weights.weight_grounding * c["grounding"] +
                weights.weight_privacy * c["privacy"] -
                weights.cost_penalty * c["cost"] -
                mode_penalty
            )
            c_eval = dict(c)
            c_eval["jev_score"] = round(jev, 4)
            c_eval["latency_utility"] = round(lat_util, 3)
            c_eval["computed_quality"] = round(quality, 3)
            evaluated.append(c_eval)

        evaluated.sort(key=lambda x: x["jev_score"], reverse=True)
        winner = evaluated[0]

        # Respect explicit overrides from caller if not 'auto'
        final_rag = rag_mode if rag_mode != "auto" else winner["rag_mode"]
        final_engine = orchestrator if orchestrator != "auto" else winner["orchestrator"]

        return {
            "jev_decision": winner["name"],
            "jev_score": winner["jev_score"],
            "target_mode": target_mode,
            "selected_rag_mode": final_rag,
            "selected_orchestrator": final_engine,
            "chunk_size": winner["chunk_size"],
            "chunk_overlap": winner["chunk_overlap"],
            "recommended_vector_store": winner["vector_store"],
            "primary_model": winner["model"],
            "estimated_latency_s": winner["est_latency_s"],
            "expected_quality": winner["computed_quality"],
            "sla_ceiling_s": sla_ceiling,
            "weights_used": weights.model_dump(),
            "candidate_evaluations": evaluated[:4],
            "rationale": (
                f"JEV Engine selected {winner['name']} with JEV score {winner['jev_score']:.3f}. "
                f"Balances expected quality ({winner['computed_quality']:.2f}) and latency "
                f"({winner['est_latency_s']}s) under {target_mode} SLA ({sla_ceiling}s)."
            )
        }

    def evaluate_supervisor_action(
        self,
        draft: str,
        sources: List[Dict[str, Any]],
        mode: str,
        remaining_budget_s: float,
        has_table: bool,
        has_emojis: bool
    ) -> Dict[str, Any]:
        """
        Arbitrates whether Supervisor should:
        - ACTION A: accept_and_verify (deterministic polish in <1ms)
        - ACTION B: elevate_llm_pass (second full LLM synthesis in 4-8s)
        """
        is_fast = mode in ("fast", "quick")
        is_privacy = mode == "privacy"

        if is_privacy:
            return {
                "action": "accept_and_verify",
                "jev_score": 1.0,
                "reason": "Air-Gapped Privacy Mode requires zero cloud LLM egress."
            }

        # Candidate A: Accept & verify deterministically
        # Expected quality: high if table and content are present
        qual_a = 0.90 if has_table else 0.85
        lat_a = 1.0  # 0.5ms execution has maximum latency utility
        jev_a = 0.40 * qual_a + 0.50 * lat_a + 0.10 * 0.95

        # Candidate B: Invoke secondary LLM pass
        if remaining_budget_s < 8.0 or is_fast:
            lat_b = 0.10  # Severe SLA violation penalty
            qual_b = 0.93
        else:
            lat_b = max(0.0, 1.0 - (6.0 / max(remaining_budget_s, 1.0)))
            qual_b = 0.95
        jev_b = 0.55 * qual_b + 0.35 * lat_b + 0.10 * 0.95 - 0.05

        action = "elevate_llm_pass" if (mode == "research" and jev_b > jev_a and remaining_budget_s >= 14.0) else "accept_and_verify"

        return {
            "action": action,
            "jev_score_a_verify": round(jev_a, 4),
            "jev_score_b_elevate": round(jev_b, 4),
            "winner_action": action,
            "reason": (
                f"JEV favored '{action}' (JEV_A: {jev_a:.3f} vs JEV_B: {jev_b:.3f}) "
                f"with {remaining_budget_s:.1f}s remaining budget in {mode} mode."
            )
        }
