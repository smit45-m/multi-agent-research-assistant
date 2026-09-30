"""Retrieve relevant evidence, preserve provenance, reject unrelated results."""
import time
from app.config import get_settings
from app.rag.hybrid_retriever import HybridRetriever, source_allowed
from app.rag.relevance import fingerprint, relevance, tokens
from app.tools.search_tool import WebSearchTool


class RetrieverAgent:
    def __init__(self, vector_store, search_tool=None, llm=None, min_relevance=None):
        self.vector_store = vector_store
        self.search_tool = search_tool or WebSearchTool()
        self.minimum = min_relevance if min_relevance is not None else get_settings().RETRIEVAL_MIN_RELEVANCE
        self.hybrid_retriever = HybridRetriever(vector_store, min_relevance=self.minimum)

    def retrieve(self, state):
        start = time.perf_counter()
        query = state["query"]
        route = state.get("routing_metadata", {})
        options = state.get("options", {})
        top_k = route.get("max_sources", options.get("max_sources", 8))
        filters = options.get("source_filters")
        questions = state.get("sub_questions") or [query]
        if state.get("iteration_count", 0):
            gaps = state.get("analysis", {}).get("gaps", [])
            questions = [g for g in gaps if isinstance(g, str) and set(tokens(g)) & set(tokens(query))][:2] or [query]
        questions = questions[:route.get("max_queries", 1)]
        candidates = list(state.get("retrieved_documents", []))
        actual = list(route.get("actual_retrieval", []))
        # User-supplied attachments are explicit context; preserve their complete extracted text.
        for attachment in state.get("attachments", []):
            if filters and attachment["kind"] not in filters:
                continue
            candidates.append({"title": attachment["filename"], "content": attachment["text"],
                               "source": "attachment:" + attachment["id"], "source_type": attachment["kind"],
                               "relevance_score": 1.0, "user_supplied": True})
        for question in questions:
            if time.monotonic() > state.get("deadline", float("inf")) - 2:
                state["warnings"].append("Retrieval stopped at the request time budget.")
                break
            try:
                documents = self.hybrid_retriever.retrieve(question, state.get("rag_mode", "hybrid"), top_k, filters)
                diagnostics = self.hybrid_retriever.last_diagnostics
                state["warnings"].extend(diagnostics.get("warnings", []))
                actual.append(diagnostics.get("actual", "bm25"))
                for doc in documents:
                    meta = doc.metadata
                    doc_filename = str(meta.get("filename") or "")
                    doc_title = str(meta.get("filename") or meta.get("title") or meta.get("source", "Document"))
                    candidates.append({"title": doc_title,
                                       "filename": doc_filename,
                                       "content": doc.page_content,
                                       "source": str(meta.get("source", meta.get("source_path", "document"))),
                                       "source_type": str(meta.get("format", "document")).lstrip("."),
                                       "relevance_score": float(meta.get("relevance_score", 0)),
                                       "dense_score": float(meta.get("dense_score", 0))})
            except Exception as exc:
                state["warnings"].append(f"Local retrieval failed: {exc}")
        relevant_local = [d for d in candidates if d.get("user_supplied") or relevance(query, d["content"], d["title"]) >= self.minimum or d.get("dense_score", 0) >= 0.5]
        permitted_web = options.get("web_search", True) and get_settings().WEB_SEARCH_ENABLED and not state.get("offline")
        # Private/library questions never leak into public search automatically.
        if state.get("mode") == "privacy" or options.get("privacy_mode"):
            permitted_web = False
        else:
            permitted_web = permitted_web and not route.get("private_context", False)
            permitted_web = permitted_web and (not filters or "web" in filters or "wikipedia" in filters or "arxiv" in filters)
        # TypeSafe AI Jev System One decision: decide if web search is needed
        from app.agents.jev_engine import JEVDecisionEngine
        jev = JEVDecisionEngine()
        max_local_score = max([d.get("relevance_score", 0) for d in relevant_local], default=0.0)
        jev_search_decision = jev.decide_web_search(
            query,
            local_doc_count=len(relevant_local),
            max_local_relevance=max_local_score,
            mode=state.get("mode", "balanced"),
            privacy_mode=state.get("privacy_mode", False)
        )
        state["jev_retrieval_decision"] = {
            "need_web_search": jev_search_decision.value,
            "search_probability": jev_search_decision.probability,
            "confidence": jev_search_decision.confidence,
            "latency_ms": jev_search_decision.latency_ms
        }

        should_search = permitted_web and jev_search_decision.value
        if should_search:
            search_questions = questions[:2 if state.get("mode") == "research" else 1]
            remaining = state.get("deadline", float("inf")) - time.monotonic() - 3
            if remaining > 1.0 and search_questions:
                import concurrent.futures
                fast_cap = getattr(get_settings(), "FAST_MODE_SEARCH_TIMEOUT_SECONDS", 1.2)
                search_timeout = min(remaining, fast_cap if state.get("mode") in ("fast", "quick") else get_settings().SEARCH_TIMEOUT_SECONDS)
                actual.append("web_search")
                
                def _do_search(q):
                    try:
                        return self.search_tool.search(q, max_results=top_k, timeout=search_timeout)
                    except Exception:
                        return []

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(search_questions), 2)) as executor:
                    futures = [executor.submit(_do_search, q) for q in search_questions]
                    for fut in concurrent.futures.as_completed(futures):
                        try:
                            results = fut.result()
                            for result in results:
                                category = result.get("source_type", "web")
                                if filters and category not in filters:
                                    continue
                                score = relevance(query, result.get("snippet", ""), result.get("title", ""))
                                if score >= self.minimum:
                                    relevant_local.append({"title": result.get("title", "Web source"), "content": result.get("snippet", ""),
                                                           "source": result.get("url", ""), "source_type": category, "relevance_score": score})
                        except Exception:
                            pass
        # Reject low-relevance snippets even if their search rank was high. Never synthesize a source.
        unique = {}
        for document in relevant_local:
            if document["content"].strip():
                key = fingerprint(document["content"])
                if key not in unique or document["relevance_score"] > unique[key]["relevance_score"]:
                    unique[key] = document
        ordered = sorted(unique.values(), key=lambda d: d["relevance_score"], reverse=True)
        chosen, counts = [], {}
        for document in ordered:
            source = document["source"]
            if counts.get(source, 0) >= 3:
                continue
            chosen.append(document)
            counts[source] = counts.get(source, 0) + 1
            if len(chosen) >= top_k:
                break
        state["retrieved_documents"] = chosen
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["evidence_history"].append(len(chosen))
        route["actual_retrieval"] = list(dict.fromkeys(actual))
        state["routing_metadata"] = route
        state["status"] = "retrieved"
        state["agent_telemetry"]["retriever_time_ms"] += round((time.perf_counter() - start) * 1000, 3)
        return state
