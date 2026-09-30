"""Adaptive direct/graph execution with bounded evidence refinement and review."""
import time
from langgraph.graph import StateGraph, START, END
from app.agents.state import ResearchState, create_initial_state
from app.agents.planner_agent import PlannerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.writer_agent import WriterAgent
from app.agents.fact_checker_agent import FactCheckerAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.meta_agent import MetaAgent
from app.chains.router import ResearchRouter
from app.chains.llm import BoundedLLM
from app.tools.search_tool import WebSearchTool


class ResearchGraph:
    def __init__(self, vector_store, llm=None, search_tool=None, offline=False, min_relevance=None):
        self.vector_store = vector_store
        self.llm = llm
        self.search_tool = search_tool
        self.offline = offline
        self.min_relevance = min_relevance

    def should_continue(self, state):
        route = state.get("routing_metadata", {})
        if state.get("mode") != "research" or state.get("iteration_count", 0) >= route.get("max_retrieval_rounds", 1):
            return "write"
        if time.monotonic() > state.get("deadline", 0) - 15:
            return "write"
        docs = state.get("retrieved_documents", [])
        if len(docs) >= 4:
            return "write"
        history = state.get("evidence_history", [])
        if len(history) > 1 and history[-1] <= history[-2]:
            return "write"
        return "retrieve" if state.get("analysis", {}).get("gaps") else "write"

    def build_graph(self, planner=None, retriever=None, analyzer=None, writer=None, fact_checker=None, supervisor=None):
        planner = planner or PlannerAgent(self.llm)
        retriever = retriever or RetrieverAgent(self.vector_store, self.search_tool)
        analyzer = analyzer or AnalyzerAgent(self.llm)
        writer = writer or WriterAgent(self.llm)
        fact_checker = fact_checker or FactCheckerAgent(self.llm)
        supervisor = supervisor or SupervisorAgent(self.llm)
        workflow = StateGraph(ResearchState)
        def tracked(name, action):
            def run(state):
                state["routing_metadata"]["stages"].append(name)
                return action(state)
            return run
        workflow.add_node("plan", tracked("plan", planner.plan))
        workflow.add_node("retrieve", tracked("retrieve", retriever.retrieve))
        workflow.add_node("analyze", tracked("analyze", analyzer.analyze))
        workflow.add_node("write", tracked("write", writer.write))
        workflow.add_node("fact_checker", tracked("fact_checker", fact_checker.verify))
        workflow.add_node("supervise", tracked("supervise", supervisor.supervise))
        workflow.add_node("review", tracked("review", writer.review))
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_conditional_edges("analyze", self.should_continue, {"retrieve": "retrieve", "write": "write"})
        workflow.add_edge("write", "fact_checker")
        workflow.add_edge("fact_checker", "supervise")
        workflow.add_conditional_edges("supervise", lambda s: "review" if s.get("mode") == "research" and not s.get("reviewed") else "end", {"review": "review", "end": END})
        workflow.add_edge("review", END)
        return workflow.compile()

    def run(self, query, rag_mode="auto", orchestrator="auto", **options):
        started = time.perf_counter()
        meta_config = MetaAgent().analyze_and_configure(
            query, mode=options.get("mode", "auto"), depth=options.get("depth"),
            rag_mode=rag_mode, orchestrator=orchestrator,
            has_attachments=bool(options.get("attachments") or options.get("attachment_ids"))
        )
        effective_rag = rag_mode if rag_mode != "auto" else meta_config["selected_rag_mode"]
        effective_mode = meta_config["resolved_mode"]
        if effective_mode in ("fast", "quick") or options.get("mode") in ("fast", "quick"):
            effective_engine = "direct"
        elif effective_mode == "privacy" or options.get("privacy_mode"):
            effective_engine = "direct" if orchestrator == "auto" else orchestrator
        elif orchestrator != "auto":
            effective_engine = orchestrator
        else:
            effective_engine = meta_config["selected_orchestrator"]

        route = ResearchRouter().execution_plan(
            query, mode=effective_mode, rag_mode=effective_rag,
            orchestrator=effective_engine, depth=options.get("depth"),
            max_sources=options.get("max_sources", 10), corpus_size=self.vector_store.get_document_count())
        state = create_initial_state(query, route["selected_rag_mode"], route["selected_orchestrator"], **options)
        attachment_ids = set(options.get("attachment_ids", []))
        if attachment_ids:
            all_docs = self.vector_store.get_all_documents()
            for d in all_docs:
                doc_id = str(d.metadata.get("document_id", ""))
                if doc_id in attachment_ids:
                    state.setdefault("attachments", []).append({
                        "id": doc_id,
                        "filename": d.metadata.get("filename", "attachment"),
                        "text": d.page_content,
                        "kind": str(d.metadata.get("format", "document")).lstrip(".")
                    })
        state["mode"] = route["selected_mode"]
        state["orchestrator"] = route["selected_orchestrator"]
        state["privacy_mode"] = (state["mode"] == "privacy" or options.get("privacy_mode", False))
        state["offline"] = self.offline or options.get("offline", False)
        state["deadline"] = time.monotonic() + (15.0 if state["mode"] in ("fast", "quick") else max(90.0, route.get("budget_seconds", 120.0)))
        state["routing_metadata"] = route
        state["meta_orchestration"] = meta_config
        state["warnings"].extend(route["warnings"])
        store_warnings = getattr(self.vector_store, "warnings", [])
        if isinstance(store_warnings, list):
            state["warnings"].extend(store_warnings)
        client = BoundedLLM(state["mode"], state["deadline"], backend=self.llm, disabled=state["offline"], privacy_mode=state["privacy_mode"])
        planner, analyzer, writer = PlannerAgent(client), AnalyzerAgent(client), WriterAgent(client)
        fact_checker = FactCheckerAgent(client)
        supervisor = SupervisorAgent(client)
        retriever = RetrieverAgent(self.vector_store, self.search_tool or WebSearchTool(), min_relevance=self.min_relevance)
        if state["orchestrator"] == "direct":
            state["sub_questions"] = [query]
            state["research_plan"] = {"objective": query, "sub_questions": [query], "method": "direct"}
            state["routing_metadata"]["stages"].append("retrieve")
            state = retriever.retrieve(state)
            state = analyzer.analyze(state)
            state["routing_metadata"]["stages"].append("write")
            state = writer.write(state)
            state["routing_metadata"]["stages"].append("fact_checker")
            state = fact_checker.verify(state)
            state["routing_metadata"]["stages"].append("supervise")
            state = supervisor.supervise(state)
            if state["mode"] == "research" and not state.get("reviewed"):
                state["routing_metadata"]["stages"].append("review")
                state = writer.review(state)
        else:
            state = self.build_graph(planner, retriever, analyzer, writer, fact_checker, supervisor).invoke(state, {"recursion_limit": 20})
        state["agent_telemetry"].update({"total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                                         "llm_calls": client.calls, "input_tokens": client.input_tokens,
                                         "output_tokens": client.output_tokens, "retrieval_rounds": state["iteration_count"]})
        state["warnings"] = list(dict.fromkeys(state["warnings"]))
        state["model"] = client.model if client.calls else None
        state["quality"]["research_review_completed"] = state.get("reviewed", False)
        return state
