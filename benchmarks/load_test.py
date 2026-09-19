"""
High-concurrency load testing script simulating 50+ concurrent users.

Measures REAL latency distribution, throughput, and error rates against the
in-process FastAPI app. Reports honest pass/fail against the sub-8-second
p95 SLA and exits non-zero on failure so CI treats regressions as failures.

Note: results depend on host hardware and whether outbound network / an LLM
key are available. The report header records both conditions.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

from starlette.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.tools.search_tool import WebSearchTool  # noqa: E402
from app.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__, "INFO")

CONCURRENT_USERS = 50
TARGET_LATENCY_THRESHOLD_S = 8.0

SAMPLE_QUERIES = [
    "What are the latest advancements in solid-state batteries compared to "
    "lithium-ion?",
    "Explain Mixture of Experts (MoE) routing mechanisms and sparse gating.",
    "What are the clinical trial benchmarks for CRISPR Cas9 base editing?",
    "How does Raft consensus handle leader election edge cases and network partitions?",
    "Analyze high-frequency order book matching engine latency and priority queues.",
    "What are the security boundaries of WebAssembly sandboxing vs native isolation?",
    "Compare zero-knowledge proofs zk-SNARKs and zk-STARKs in terms of proof size.",
    "Explain Kubernetes Horizontal Pod Autoscaler HPA algorithm with custom metrics.",
]


def simulate_user_request(
    client: TestClient, user_id: int, query: str
) -> Dict[str, Any]:
    """Simulate a single user submitting a research query."""
    t0 = time.perf_counter()
    try:
        response = client.post(
            "/api/v1/research/sync",
            json={"query": query, "depth": "standard", "rag_mode": "hybrid"},
        )
        elapsed = time.perf_counter() - t0
        body_ok = response.status_code == 200 and (
            response.json().get("status") == "completed"
        )
        return {
            "user_id": user_id,
            "latency_s": elapsed,
            "status_code": response.status_code,
            "success": body_ok,
        }
    except Exception as exc:  # noqa: BLE001 - a crash is a failed request
        elapsed = time.perf_counter() - t0
        return {
            "user_id": user_id,
            "latency_s": elapsed,
            "status_code": 500,
            "success": False,
            "error": str(exc),
        }


def run_concurrency_benchmark(
    num_concurrent: int = CONCURRENT_USERS,
) -> Dict[str, Any]:
    """Run thread-pool concurrent requests and measure real latencies."""
    app = create_app()
    settings = get_settings()
    web_available = WebSearchTool().is_available()

    with TestClient(app) as client:
        logger.info(
            "Starting load test: %d simultaneous users (llm=%s, web=%s)...",
            num_concurrent,
            settings.llm_available,
            web_available,
        )
        start_total = time.perf_counter()

        results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_concurrent
        ) as executor:
            futures = [
                executor.submit(
                    simulate_user_request,
                    client,
                    i + 1,
                    SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)],
                )
                for i in range(num_concurrent)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        total_duration = time.perf_counter() - start_total

    latencies = sorted(r["latency_s"] for r in results)
    successful = sum(1 for r in results if r["success"])
    n = len(latencies)

    def pct(p: float) -> float:
        return latencies[min(int(n * p), n - 1)] if n else 0.0

    summary = {
        "concurrent_users": num_concurrent,
        "total_requests": len(results),
        "successful_requests": successful,
        "success_rate_pct": round(successful / max(len(results), 1) * 100, 1),
        "total_test_duration_s": round(total_duration, 2),
        "avg_latency_s": round(sum(latencies) / max(n, 1), 2),
        "p50_latency_s": round(pct(0.50), 2),
        "p90_latency_s": round(pct(0.90), 2),
        "p95_latency_s": round(pct(0.95), 2),
        "p99_latency_s": round(pct(0.99), 2),
        "sub_8s_sla_met": pct(0.95) < TARGET_LATENCY_THRESHOLD_S,
        "requests_per_second": round(len(results) / max(total_duration, 0.001), 2),
        "llm_mode": "llm" if settings.llm_available else "extractive-fallback",
        "web_search_available": web_available,
    }

    print("\n" + "=" * 60)
    print("=== CONCURRENT USERS LOAD TEST (MEASURED) ===")
    print(
        f"Execution mode:         {summary['llm_mode']} (web search: {web_available})"
    )
    print(f"Concurrent Users:       {summary['concurrent_users']}")
    print(
        f"Success Rate:           {summary['success_rate_pct']}% "
        f"({summary['successful_requests']}/{summary['total_requests']})"
    )
    print(f"Average Latency:        {summary['avg_latency_s']}s")
    print(f"p50 Latency:            {summary['p50_latency_s']}s")
    print(
        f"p95 Latency:            {summary['p95_latency_s']}s "
        f"(Target: <{TARGET_LATENCY_THRESHOLD_S}s)"
    )
    print(f"p99 Latency:            {summary['p99_latency_s']}s")
    print(f"Sub-8-Second SLA Met:   {summary['sub_8s_sla_met']}")
    print(f"Throughput:             {summary['requests_per_second']} req/s")
    print("=" * 60 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrency load test")
    parser.add_argument(
        "--users",
        type=int,
        default=CONCURRENT_USERS,
        help="Number of concurrent users",
    )
    parser.add_argument(
        "--enforce-sla",
        action="store_true",
        help="Exit non-zero if the p95 SLA or success rate fails",
    )
    args = parser.parse_args()

    result = run_concurrency_benchmark(args.users)
    if args.enforce_sla and not (
        result["sub_8s_sla_met"] and result["success_rate_pct"] == 100.0
    ):
        print("LOAD TEST FAILED: SLA not met.")
        sys.exit(1)
