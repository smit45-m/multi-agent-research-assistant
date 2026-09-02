"""
High-concurrency load testing script simulating 50+ concurrent users.
Measures latency distribution, throughput, error rates, and verifies
the sub-8-second latency SLA at 50+ concurrency.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import List, Dict, Any
from starlette.testclient import TestClient

from app.main import create_app
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

CONCURRENT_USERS = 50
TARGET_LATENCY_THRESHOLD_S = 8.0

SAMPLE_QUERIES = [
    "What are the latest advancements in solid-state batteries compared to lithium-ion?",
    "Explain Mixture of Experts (MoE) routing mechanisms and sparse gating.",
    "What are the clinical trial benchmarks for CRISPR Cas9 base editing?",
    "How does Raft consensus handle leader election edge cases and network partitions?",
    "Analyze high-frequency order book matching engine latency and priority queues.",
    "What are the security boundaries of WebAssembly sandboxing vs native isolation?",
    "Compare zero-knowledge proofs zk-SNARKs and zk-STARKs in terms of proof size.",
    "Explain Kubernetes Horizontal Pod Autoscaler HPA algorithm with custom metrics."
]

def simulate_user_request(client: TestClient, user_id: int, query: str) -> Dict[str, Any]:
    """Simulates a single user submitting a research query and checking result."""
    t0 = time.perf_counter()
    try:
        response = client.post(
            "/api/v1/research/sync",
            json={"query": query, "depth": "standard", "rag_mode": "hybrid"}
        )
        elapsed = time.perf_counter() - t0
        status = response.status_code
        is_success = (status == 200)
        return {
            "user_id": user_id,
            "latency_s": elapsed,
            "status_code": status,
            "success": is_success
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "user_id": user_id,
            "latency_s": elapsed,
            "status_code": 500,
            "success": False,
            "error": str(e)
        }

def run_concurrency_benchmark(num_concurrent: int = CONCURRENT_USERS) -> Dict[str, Any]:
    """Runs thread-pool concurrent requests to verify 50+ user support."""
    import concurrent.futures

    app = create_app()
    client = TestClient(app)
    
    logger.info(f"Starting concurrency load test with {num_concurrent} simultaneous users...")
    start_total = time.perf_counter()

    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = []
        for i in range(num_concurrent):
            q = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
            futures.append(executor.submit(simulate_user_request, client, i + 1, q))

        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_duration = time.perf_counter() - start_total
    latencies = [r["latency_s"] for r in results]
    latencies.sort()

    successful = sum(1 for r in results if r["success"])
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
    p90 = latencies[int(len(latencies) * 0.90)] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

    summary = {
        "concurrent_users": num_concurrent,
        "total_requests": len(results),
        "successful_requests": successful,
        "success_rate_pct": round((successful / len(results)) * 100.0, 1),
        "total_test_duration_s": round(total_duration, 2),
        "avg_latency_s": round(avg_lat, 2),
        "p50_latency_s": round(p50, 2),
        "p90_latency_s": round(p90, 2),
        "p95_latency_s": round(p95, 2),
        "p99_latency_s": round(p99, 2),
        "sub_8s_sla_met": p95 < TARGET_LATENCY_THRESHOLD_S,
        "requests_per_second": round(len(results) / max(total_duration, 0.001), 2)
    }

    print("\n" + "="*60)
    print("=== 50+ CONCURRENT USERS LOAD TEST RESULTS ===")
    print(f"Concurrent Users:       {summary['concurrent_users']}")
    print(f"Success Rate:           {summary['success_rate_pct']}% ({summary['successful_requests']}/{summary['total_requests']})")
    print(f"Average Latency:        {summary['avg_latency_s']}s")
    print(f"p50 Latency:            {summary['p50_latency_s']}s")
    print(f"p95 Latency:            {summary['p95_latency_s']}s (Target: <8.0s)")
    print(f"p99 Latency:            {summary['p99_latency_s']}s")
    print(f"Sub-8-Second SLA Met:   {summary['sub_8s_sla_met']}")
    print(f"Throughput:             {summary['requests_per_second']} req/s")
    print("="*60 + "\n")

    return summary

if __name__ == "__main__":
    run_concurrency_benchmark()
