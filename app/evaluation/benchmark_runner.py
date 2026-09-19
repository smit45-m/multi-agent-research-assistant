"""
Automated benchmark runner for the multi-agent research assistant.

Every metric reported here is MEASURED by actually executing the full
6-agent pipeline for each test case:

- accuracy      = 0.6 * assertion coverage of the generated report/evidence
                  + 0.4 * verifier-measured grounding of the report
- latency       = real wall-clock seconds for the end-to-end pipeline run
- speedup       = retriever-measured parallel-vs-sequential retrieval ratio

There are no simulated latencies, calibrated constants, or self-matching
assertions. A case passes only if its measured accuracy and latency meet
the thresholds declared in the case definition. Honest failures are
reported as failures.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.evaluation.grounding import assertion_coverage
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

DEFAULT_PARALLELISM = 4


class BenchmarkResult(BaseModel):
    """Result of running an individual test case through the real pipeline."""

    test_id: str
    category: str
    query: str
    target_source_format: str
    accuracy_score: float  # measured, 0-100
    assertion_coverage: float  # measured, 0-100
    grounding_score: float  # measured, 0-100
    latency_seconds: float  # measured wall-clock
    synthesis_speedup: float  # measured, 0-100
    sources_retrieved: int
    passed: bool
    details: str


class BenchmarkSummary(BaseModel):
    """Aggregate measured results across the benchmark suite."""

    total_test_cases: int
    passed_test_cases: int
    pass_rate_percentage: float
    average_accuracy_percentage: float
    target_accuracy_percentage: float = 85.0
    average_latency_seconds: float
    p95_latency_seconds: float
    target_latency_seconds: float = 8.0
    average_synthesis_speedup_percentage: float
    target_synthesis_speedup_percentage: float = 60.0
    categories_evaluated: List[str]
    llm_mode: str  # 'llm' or 'extractive-fallback'
    measurement_note: str
    sample_results: List[BenchmarkResult]


class BenchmarkEvaluator:
    """
    Evaluates the multi-agent pipeline across curated test cases by
    actually running each case end-to-end and measuring the outcome.
    """

    def __init__(self, benchmark_path: Optional[str] = None):
        base_dir = Path(__file__).parent.parent.parent
        self.benchmark_file = (
            Path(benchmark_path)
            if benchmark_path
            else base_dir / "data" / "benchmarks" / "eval_200_cases.json"
        )
        self._graph: Optional[Any] = None

    def _get_graph(self) -> Any:
        """Lazily build one shared ResearchGraph for all cases."""
        if self._graph is None:
            from app.agents.graph import ResearchGraph
            from app.rag.vector_store import VectorStoreManager

            vstore = VectorStoreManager()
            try:
                vstore.initialize()
                from app.rag.corpus_loader import ensure_corpus_indexed

                ensure_corpus_indexed(vstore)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vector store unavailable for benchmark: %s", exc)
            graph = ResearchGraph(vstore)
            graph.build_graph()
            self._graph = graph
        return self._graph

    def load_cases(self) -> List[Dict[str, Any]]:
        """Load test cases, generating the dataset file if missing."""
        if not self.benchmark_file.exists():
            logger.info("Benchmark cases file not found, generating...")
            from app.evaluation.generate_benchmark import main as gen_main

            gen_main()

        with open(self.benchmark_file, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
        return data

    def evaluate_case(self, case: Dict[str, Any]) -> BenchmarkResult:
        """
        Run one test case through the REAL pipeline and measure the outcome.
        """
        query = case["query"]
        expected_assertions = case.get("expected_assertions", [])
        graph = self._get_graph()

        start_t = time.perf_counter()
        try:
            state = graph.run(query, rag_mode="hybrid")
            error_detail = ""
        except Exception as exc:  # noqa: BLE001 - a crash is a failed case
            elapsed = time.perf_counter() - start_t
            return BenchmarkResult(
                test_id=case["id"],
                category=case["category"],
                query=query,
                target_source_format=case.get("target_source_format", "pdf"),
                accuracy_score=0.0,
                assertion_coverage=0.0,
                grounding_score=0.0,
                latency_seconds=round(elapsed, 2),
                synthesis_speedup=0.0,
                sources_retrieved=0,
                passed=False,
                details=f"Pipeline error: {exc}",
            )
        elapsed = time.perf_counter() - start_t

        report = state.get("final_report", "")
        docs = state.get("retrieved_documents", [])
        evidence_text = " ".join(str(d.get("content", "")) for d in docs)

        # Assertion coverage: expected keyphrases found in report OR evidence.
        coverage = assertion_coverage(f"{report}\n{evidence_text}", expected_assertions)
        coverage_ratio = float(str(coverage["coverage"]))

        # Grounding: measured by the Verifier agent during the run.
        grounding_ratio = float(
            state.get("verification", {}).get("grounded_ratio", 0.0)
        )

        accuracy = round(0.6 * coverage_ratio + 0.4 * grounding_ratio, 4)
        speedup = float(state.get("synthesis_speedup_ratio", 0.0))

        min_acc = float(case.get("min_accuracy_threshold", 0.85))
        max_lat = float(case.get("target_latency_s", 8.0))
        passed = accuracy >= min_acc and elapsed < max_lat

        missing_obj = coverage.get("missing", [])
        missing = missing_obj if isinstance(missing_obj, list) else []
        detail_parts = [
            f"coverage {coverage_ratio:.0%}",
            f"grounding {grounding_ratio:.0%}",
            f"{len(docs)} passages",
        ]
        if missing:
            detail_parts.append(
                "missing assertions: " + ", ".join(str(m) for m in missing[:3])
            )
        if error_detail:
            detail_parts.append(error_detail)

        return BenchmarkResult(
            test_id=case["id"],
            category=case["category"],
            query=query,
            target_source_format=case.get("target_source_format", "pdf"),
            accuracy_score=round(accuracy * 100.0, 1),
            assertion_coverage=round(coverage_ratio * 100.0, 1),
            grounding_score=round(grounding_ratio * 100.0, 1),
            latency_seconds=round(elapsed, 2),
            synthesis_speedup=round(speedup * 100.0, 1),
            sources_retrieved=len(docs),
            passed=passed,
            details=" | ".join(detail_parts),
        )

    def run_benchmark(
        self,
        max_cases: Optional[int] = None,
        parallelism: int = DEFAULT_PARALLELISM,
    ) -> BenchmarkSummary:
        """
        Run the benchmark by executing every case through the real pipeline.

        Args:
            max_cases: Optional cap on the number of cases.
            parallelism: Concurrent case executions (the pipeline itself is
                thread-safe; retrieval fan-out is bounded per run).
        """
        from app.config import get_settings

        cases = self.load_cases()
        if max_cases:
            cases = cases[:max_cases]

        logger.info(
            "Running benchmark on %d cases (parallelism=%d)...",
            len(cases),
            parallelism,
        )
        # Build the graph once, up front (not inside worker threads).
        self._get_graph()

        results: List[BenchmarkResult] = []
        if parallelism <= 1:
            for case in cases:
                results.append(self.evaluate_case(case))
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as pool:
                futures = {pool.submit(self.evaluate_case, c): c["id"] for c in cases}
                for fut in as_completed(futures):
                    results.append(fut.result())

        results.sort(key=lambda r: r.test_id)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_acc = sum(r.accuracy_score for r in results) / max(total, 1)
        latencies = sorted(r.latency_seconds for r in results)
        avg_lat = sum(latencies) / max(total, 1)
        p95_lat = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        avg_speedup = sum(r.synthesis_speedup for r in results) / max(total, 1)
        categories = sorted({r.category for r in results})

        settings = get_settings()
        llm_mode = "llm" if settings.llm_available else "extractive-fallback"

        summary = BenchmarkSummary(
            total_test_cases=total,
            passed_test_cases=passed,
            pass_rate_percentage=round((passed / max(total, 1)) * 100.0, 1),
            average_accuracy_percentage=round(avg_acc, 1),
            average_latency_seconds=round(avg_lat, 2),
            p95_latency_seconds=round(p95_lat, 2),
            average_synthesis_speedup_percentage=round(avg_speedup, 1),
            categories_evaluated=categories,
            llm_mode=llm_mode,
            measurement_note=(
                "All metrics measured from real end-to-end pipeline "
                "executions: accuracy = 0.6*assertion_coverage + "
                "0.4*verifier_grounding; latency = wall-clock; speedup = "
                "measured parallel retrieval gain."
            ),
            sample_results=results[:25],
        )
        logger.info(
            "Benchmark complete: %d/%d passed (accuracy %.1f%%, avg latency "
            "%.2fs, p95 %.2fs, speedup %.1f%%, mode=%s)",
            passed,
            total,
            summary.average_accuracy_percentage,
            summary.average_latency_seconds,
            summary.p95_latency_seconds,
            summary.average_synthesis_speedup_percentage,
            llm_mode,
        )
        return summary


def main() -> None:
    """CLI entry point for the benchmark."""
    parser = argparse.ArgumentParser(description="Run the evaluation benchmark")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Cap the number of cases (default: all)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=DEFAULT_PARALLELISM,
        help="Concurrent case executions",
    )
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator()
    summary = evaluator.run_benchmark(
        max_cases=args.max_cases, parallelism=args.parallelism
    )

    print("\n" + "=" * 64)
    print("=== MULTI-AGENT RESEARCH ASSISTANT BENCHMARK (MEASURED) ===")
    print(f"Execution mode:             {summary.llm_mode}")
    print(f"Total Test Cases Evaluated: {summary.total_test_cases}")
    print(
        f"Passed Test Cases:          {summary.passed_test_cases} "
        f"({summary.pass_rate_percentage}%)"
    )
    print(
        f"Average Measured Accuracy:  "
        f"{summary.average_accuracy_percentage}% "
        f"(Target: >={summary.target_accuracy_percentage}%)"
    )
    print(
        f"Average Latency:            {summary.average_latency_seconds}s "
        f"(p95: {summary.p95_latency_seconds}s, "
        f"Target: <{summary.target_latency_seconds}s)"
    )
    print(
        f"Measured Retrieval Speedup: {summary.average_synthesis_speedup_percentage}%"
    )
    print(f"Note: {summary.measurement_note}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
