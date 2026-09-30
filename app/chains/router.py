"""
Multi-step LLM routing workflows and prompt optimization in LangChain.
Dynamically routes queries across 15+ multi-format sources and configures
the multi-agent retrieval pipeline for maximum accuracy and efficiency.
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

class RoutingDecision(BaseModel):
    """Structured decision produced by the multi-step routing workflow."""
    domain: str = Field(description="Detected domain: 'academic', 'biomedical', 'financial', 'technical', 'news', 'general'")
    complexity: str = Field(description="Query complexity level: 'simple', 'moderate', 'complex'")
    recommended_rag_mode: str = Field(description="Optimal RAG strategy: 'hybrid', 'rrf', 'multi_query', 'vector', 'bm25'")
    target_source_formats: List[str] = Field(description="List of target source formats to query from the 15+ supported sources")
    optimized_planner_prompt: str = Field(description="Optimized prompt for the Research Planner agent")
    requires_multi_query_expansion: bool = Field(default=True)
    estimated_synthesis_time_s: float = Field(default=4.5)

class ResearchRouter:
    """
    Multi-step LLM routing workflow.
    Analyzes queries, decomposes intent, and selects the optimal multi-source
    retrieval pipeline with customized, prompt-optimized parameters.
    """
    
    # 15+ supported source categories
    AVAILABLE_SOURCES = [
        "pdf", "docx", "txt", "csv", "json", "html", "md", "xlsx",
        "arxiv", "wikipedia", "pubmed", "news", "web", "code", "configs"
    ]

    def __init__(self, llm=None):
        self.llm = llm

    def execution_plan(self, query: str, mode="auto", rag_mode="auto", orchestrator="auto", depth=None, max_sources=10, corpus_size=0):
        """Deterministic budget routing; explicit user choices take priority over heuristics."""
        import re
        from app.config import get_settings
        settings = get_settings()
        words = query.split()
        lower = query.lower()
        compare = bool(re.search(r"\b(compare|versus|trade.?offs?|differences?|evaluate)\b|\bvs\.?\b", lower))
        deep = bool(re.search(r"\b(comprehensive|systematic|in.depth|deep research|literature review|cross.reference|multi.hop)\b", lower))
        fresh = bool(re.search(r"\b(today|latest|current|recent|now|this year|this month)\b|\b20[2-9]\d\b", lower))
        private = bool(re.search(r"\b(our|my|internal|attached|uploaded|this document|this report|company policy)\b", lower))
        exact = bool(re.search(r'"[^\"]+"|\b[A-Z]{2,}[-_]\d+|\b\w+_\w+\b|\b\w+\.(?:py|js|ts|json)\b', query))
        cognitive = len(re.findall(r"\b(compare|analy[sz]e|evaluate|justify|design|limitations|evidence|causes|implications)\b", lower))
        complexity = "complex" if deep or len(words) > 65 or cognitive >= 3 or query.count("?") >= 3 else "moderate" if compare or len(words) > 12 or cognitive else "simple"
        selected_mode = {"quick": "fast", "standard": "balanced", "deep": "research"}.get(depth) if mode == "auto" and depth else mode
        reason = []
        if selected_mode == "auto":
            selected_mode = {"simple": "fast", "moderate": "balanced", "complex": "research"}[complexity]
            reason.append(f"{len(words)} words and {complexity} evidence demand selected {selected_mode} mode")
        else:
            reason.append(f"User selected {selected_mode} mode")
        selected_rag = rag_mode
        if rag_mode == "auto":
            if exact:
                selected_rag = "bm25"
                reason.append("Exact identifier or quoted phrase favors keyword retrieval")
            elif len(words) > 250 or (corpus_size > 1500 and "summar" in lower):
                selected_rag = "hierarchical"
                reason.append("Large context favors passages with neighboring context")
            elif selected_mode == "research" and (compare or complexity == "complex"):
                selected_rag = "multi_query"
                reason.append("Multiple evidence needs favor diversified query retrieval")
            else:
                selected_rag = "hybrid"
                reason.append("Hybrid is the general-purpose retrieval default")
        else:
            reason.append(f"User selected {rag_mode} retrieval")
        engine = ("direct" if selected_mode == "fast" else "langgraph") if orchestrator == "auto" else orchestrator
        warnings = []
        if engine == "crewai":
            engine = "langgraph"
            warnings.append("CrewAI is not used by the bounded evidence pipeline; executed LangGraph instead.")
        if selected_mode == "privacy":
            private = True
            fresh = False
            engine = "direct" if orchestrator == "auto" else orchestrator
        budgets = {
            "fast": settings.FAST_BUDGET_SECONDS,
            "balanced": settings.BALANCED_BUDGET_SECONDS,
            "research": settings.RESEARCH_BUDGET_SECONDS,
            "privacy": getattr(settings, "PRIVACY_BUDGET_SECONDS", 10.0),
        }
        caps = {
            "fast": (3, 1, 1, 750),
            "balanced": (8, 2, 1, 3500),
            "research": (14, 4, 2, 4200),
            "privacy": (12, 2, 1, 3500),
        }
        sources, queries, rounds, chars = caps.get(selected_mode, caps["balanced"])
        budget_sec = budgets.get(selected_mode, budgets["balanced"])
        from app.agents.jev_engine import JEVDecisionEngine
        jev_plan = JEVDecisionEngine().optimize_pipeline_config(
            query=query, mode=selected_mode, depth=depth,
            rag_mode=selected_rag, orchestrator=engine, corpus_size=corpus_size
        )
        return {"complexity": complexity, "selected_mode": selected_mode, "selected_rag_mode": selected_rag,
                "selected_orchestrator": engine, "requested_orchestrator": orchestrator,
                "reason": "; ".join(reason), "budget_seconds": budget_sec,
                "max_sources": min(max_sources, sources), "max_queries": queries,
                "max_retrieval_rounds": rounds, "per_source_chars": chars,
                "requires_fresh_evidence": fresh, "private_context": private,
                "jev_score": jev_plan.get("jev_score", 0.90),
                "jev_decision": jev_plan.get("jev_decision", ""),
                "jev_model_source": jev_plan.get("model_source", "typesafe_ai_jev"),
                "jev_probabilities": jev_plan.get("decision_probabilities", {}),
                "warnings": warnings, "stages": [], "actual_retrieval": []}

    def route(self, query: str, depth: str = "standard", user_sources: List[str] = None) -> RoutingDecision:
        """
        Executes the multi-step routing workflow:
        Step 1: Domain & Intent Classification
        Step 2: Source Format Mapping (across 15+ formats)
        Step 3: RAG Retrieval Strategy Optimization (Dense, Sparse, RRF, Multi-Query)
        Step 4: Prompt Optimization Calibration
        """
        q_lower = query.lower()
        
        # Step 1: Intent & Domain Classification
        if any(term in q_lower for term in ["paper", "study", "arxiv", "theorem", "quantum", "llm", "transformer", "neural"]):
            domain = "academic"
            target_sources = ["arxiv", "pdf", "wikipedia", "web"]
            rag_mode = "rrf"
        elif any(term in q_lower for term in ["clinical", "gene", "disease", "drug", "fda", "medical", "patient", "protein"]):
            domain = "biomedical"
            target_sources = ["pubmed", "arxiv", "pdf", "web"]
            rag_mode = "hybrid"
        elif any(term in q_lower for term in ["revenue", "stock", "margin", "ebitda", "inflation", "market", "q1", "q2", "q3", "q4", "finance", "gdp"]):
            domain = "financial"
            target_sources = ["csv", "xlsx", "json", "pdf", "web"]
            rag_mode = "hybrid"
        elif any(term in q_lower for term in ["code", "api", "function", "architecture", "docker", "fastapi", "python", "kubernetes", "bug"]):
            domain = "technical"
            target_sources = ["code", "configs", "md", "txt", "web"]
            rag_mode = "multi_query"
        elif any(term in q_lower for term in ["today", "latest", "recent", "2025", "2026", "news", "announcement"]):
            domain = "news"
            target_sources = ["news", "web", "html"]
            rag_mode = "hybrid"
        else:
            domain = "general"
            target_sources = ["pdf", "wikipedia", "web", "txt", "md"]
            rag_mode = "hybrid"

        # Apply user overrides if specified
        if user_sources:
            target_sources = [s for s in user_sources if s in self.AVAILABLE_SOURCES] or target_sources

        # Step 2: Complexity Evaluation
        word_count = len(query.split())
        if word_count > 15 or depth == "deep":
            complexity = "complex"
            requires_multi_query = True
            est_time = 6.2
        elif word_count > 7 or depth == "standard":
            complexity = "moderate"
            requires_multi_query = (rag_mode in ["rrf", "multi_query"])
            est_time = 4.5
        else:
            complexity = "simple"
            requires_multi_query = False
            est_time = 2.8

        # Step 3: Prompt Optimization Calibration
        optimized_prompt = (
            f"Role: Expert {domain.capitalize()} Research Planner. "
            f"Query Complexity: {complexity}. "
            f"Prioritize verifiable factual cross-referencing across target source formats: {', '.join(target_sources)}. "
            f"Enforce explicit atomic sub-questions, elimination of speculative claims, and strict attribution."
        )

        decision = RoutingDecision(
            domain=domain,
            complexity=complexity,
            recommended_rag_mode=rag_mode,
            target_source_formats=target_sources,
            optimized_planner_prompt=optimized_prompt,
            requires_multi_query_expansion=requires_multi_query,
            estimated_synthesis_time_s=est_time
        )
        logger.info(f"Routing workflow completed: Domain={domain}, RAG_Mode={rag_mode}, Sources={len(target_sources)}")
        return decision
