"""
Agent 2 - RAG Retriever.

Retrieves and fuses context from multi-format sources using:
- Hybrid Dense FAISS + Sparse BM25 with Reciprocal Rank Fusion (RRF)
- Multi-Query Expansion
- ArXiv, Wikipedia, and Live Web search tools

Sub-question retrieval fans out over a thread pool. The synthesis speedup
reported in telemetry is MEASURED: estimated sequential time (sum of
per-task durations) vs. actual parallel wall-clock time.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from app.agents.state import ResearchState
from app.rag.document_loader import fetch_arxiv_papers, fetch_wikipedia_summary
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import VectorStoreManager
from app.tools.search_tool import WebSearchTool
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_MAX_PARALLEL_WORKERS = 6


class RetrieverAgent:
    """Autonomous agent responsible for hybrid multi-source retrieval."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager],
        search_tool: WebSearchTool,
        llm: Optional[ChatOpenAI] = None,
    ):
        self.vector_store = vector_store
        self.search_tool = search_tool
        self.hybrid_retriever = (
            HybridRetriever(vector_store) if vector_store is not None else None
        )
        self.llm = llm  # reserved for LLM-guided query refinement

    # ------------------------------------------------------------------
    # Individual retrieval tasks (each returns (duration_s, results))
    # ------------------------------------------------------------------

    def _task_hybrid(
        self, sub_q: str, rag_mode: str
    ) -> Tuple[float, List[Dict[str, Any]]]:
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        if self.hybrid_retriever is None:
            return 0.0, results
        try:
            docs = self.hybrid_retriever.retrieve(query=sub_q, mode=rag_mode, top_k=4)
            for doc in docs:
                score = doc.metadata.get(
                    "rrf_score", doc.metadata.get("relevance_score", 0.5)
                )
                results.append(
                    {
                        "title": doc.metadata.get(
                            "title",
                            doc.metadata.get("source_path", "Knowledge Base"),
                        ),
                        "content": doc.page_content,
                        "source": doc.metadata.get(
                            "source_path",
                            doc.metadata.get("source", "internal_kb"),
                        ),
                        "relevance_score": float(score),
                        "source_type": doc.metadata.get("format", "hybrid_store"),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
            logger.warning(
                "[RetrieverAgent] Hybrid search failed for '%s': %s",
                sub_q[:50],
                exc,
            )
        return time.perf_counter() - t0, results

    def _task_arxiv(self, sub_q: str) -> Tuple[float, List[Dict[str, Any]]]:
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        try:
            for adoc in fetch_arxiv_papers(sub_q, max_results=2):
                results.append(
                    {
                        "title": adoc.metadata.get("title", "ArXiv Paper"),
                        "content": adoc.page_content,
                        "source": adoc.metadata.get("source", "https://arxiv.org"),
                        "relevance_score": 0.8,
                        "source_type": "arxiv_academic",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("ArXiv retrieval skipped: %s", exc)
        return time.perf_counter() - t0, results

    def _task_wikipedia(self, sub_q: str) -> Tuple[float, List[Dict[str, Any]]]:
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        try:
            clean_term = (
                sub_q.split("?")[0]
                .replace("What are", "")
                .replace("How does", "")
                .strip()
            )
            for wdoc in fetch_wikipedia_summary(clean_term[:40]):
                results.append(
                    {
                        "title": wdoc.metadata.get("title", "Wikipedia"),
                        "content": wdoc.page_content,
                        "source": wdoc.metadata.get("source", "https://wikipedia.org"),
                        "relevance_score": 0.7,
                        "source_type": "wikipedia",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Wikipedia retrieval skipped: %s", exc)
        return time.perf_counter() - t0, results

    def _task_web(self, sub_q: str) -> Tuple[float, List[Dict[str, Any]]]:
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        try:
            for r in self.search_tool.search(sub_q, max_results=2):
                snippet = r.get("snippet", "")
                if not snippet:
                    continue
                results.append(
                    {
                        "title": r.get("title", r.get("url", "Web Source")),
                        "content": snippet,
                        "source": r.get("url", ""),
                        "relevance_score": 0.6,
                        "source_type": "web_search",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Web search skipped: %s", exc)
        return time.perf_counter() - t0, results

    # ------------------------------------------------------------------

    def _build_tasks(
        self, sub_questions: List[str], rag_mode: str, routing: Dict[str, Any]
    ) -> List[Tuple[Any, tuple]]:
        """Build the list of (callable, args) retrieval tasks to run."""
        target_formats = routing.get("target_source_formats", [])
        domain = routing.get("domain", "")
        web_available = self.search_tool.is_available()

        tasks: List[Tuple[Any, tuple]] = []
        for sq in sub_questions:
            tasks.append((self._task_hybrid, (sq, rag_mode)))
            if "arxiv" in target_formats or domain in ("academic", "technical"):
                if web_available:
                    tasks.append((self._task_arxiv, (sq,)))
            if "wikipedia" in target_formats and web_available:
                tasks.append((self._task_wikipedia, (sq,)))
            if web_available:
                tasks.append((self._task_web, (sq,)))
        return tasks

    def retrieve(self, state: ResearchState) -> ResearchState:
        """
        Execute hybrid retrieval across all sub-questions in parallel and
        record measured telemetry (including the real parallel speedup).
        """
        start_t = time.perf_counter()
        sub_questions = state.get("sub_questions", []) or [state["query"]]
        rag_mode = state.get("rag_mode", "hybrid")
        routing = state.get("routing_metadata", {})

        logger.info(
            "[RetrieverAgent] Running '%s' retrieval across %d sub-questions",
            rag_mode,
            len(sub_questions),
        )

        tasks = self._build_tasks(sub_questions, rag_mode, routing)

        all_retrieved: List[Dict[str, Any]] = []
        seen_hashes: set = set()
        task_durations: List[float] = []

        def _collect(items: List[Dict[str, Any]]) -> None:
            for item in items:
                c_hash = hash(str(item.get("content", "")).strip()[:150])
                if c_hash not in seen_hashes and item.get("content"):
                    seen_hashes.add(c_hash)
                    all_retrieved.append(item)

        if len(tasks) <= 1:
            for fn, args in tasks:
                duration, items = fn(*args)
                task_durations.append(duration)
                _collect(items)
        else:
            with ThreadPoolExecutor(
                max_workers=min(_MAX_PARALLEL_WORKERS, len(tasks))
            ) as pool:
                futures = [pool.submit(fn, *args) for fn, args in tasks]
                for fut in as_completed(futures):
                    duration, items = fut.result()
                    task_durations.append(duration)
                    _collect(items)

        # Sort fused results by relevance for downstream consumers.
        all_retrieved.sort(
            key=lambda d: float(d.get("relevance_score", 0.0)), reverse=True
        )

        state["retrieved_documents"] = all_retrieved
        state["status"] = "retrieved"
        state["iteration_count"] = state.get("iteration_count", 0) + 1

        # --- MEASURED parallel speedup -------------------------------
        elapsed_s = time.perf_counter() - start_t
        sequential_estimate_s = sum(task_durations)
        if sequential_estimate_s > 0 and elapsed_s > 0:
            speedup_ratio = max(
                0.0, 1.0 - (elapsed_s / max(sequential_estimate_s, 1e-9))
            )
        else:
            speedup_ratio = 0.0

        state["synthesis_speedup_ratio"] = round(speedup_ratio, 4)
        tel = state["agent_telemetry"]
        tel["retriever_time_ms"] = round(elapsed_s * 1000.0, 2)
        tel["sequential_retrieval_baseline_ms"] = round(
            sequential_estimate_s * 1000.0, 2
        )
        tel["parallel_retrieval_actual_ms"] = round(elapsed_s * 1000.0, 2)
        tel["synthesis_reduction_pct"] = round(speedup_ratio * 100.0, 1)

        logger.info(
            "[RetrieverAgent] Retrieved %d passages in %.0fms "
            "(sequential estimate %.0fms -> measured speedup %.0f%%)",
            len(all_retrieved),
            elapsed_s * 1000.0,
            sequential_estimate_s * 1000.0,
            speedup_ratio * 100.0,
        )
        return state
