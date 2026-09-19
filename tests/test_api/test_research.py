"""
Integration tests for the research API endpoints.
"""


def test_submit_research_query(test_client):
    response = test_client.post(
        "/api/v1/research/sync",
        json={
            "query": "What are autonomous agents?",
            "depth": "standard",
            "max_sources": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert data["status"] == "completed"
    # Metrics are MEASURED: they must exist and be valid ratios, but we do
    # not assert flattering values — honest scores can be low.
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert 0.0 <= data["response_accuracy_score"] <= 1.0
    # The Verifier and Critic must have run and documented their method.
    assert data["verification"] is not None
    assert data["verification"]["method"]
    assert data["critique"] is not None
    assert data["critique"]["method"]
    # Telemetry must reflect real measured stage timings.
    assert data["telemetry"]["total_latency_ms"] > 0.0


def test_submit_research_query_async(test_client):
    response = test_client.post(
        "/api/v1/research/",
        json={
            "query": "What are autonomous agents in AI?",
            "depth": "standard",
            "max_sources": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] in ["pending", "completed"]


def test_submit_research_query_invalid(test_client):
    response = test_client.post(
        "/api/v1/research/",
        json={"query": "", "depth": "standard", "max_sources": 5},
    )
    assert response.status_code == 422


def test_submit_research_query_short(test_client):
    response = test_client.post(
        "/api/v1/research/",
        json={"query": "ab", "depth": "standard", "max_sources": 5},
    )
    assert response.status_code == 422


def test_get_research_result_not_found(test_client):
    response = test_client.get("/api/v1/research/nonexistent-id")
    assert response.status_code == 404


def test_get_benchmark_endpoint(test_client):
    """The benchmark endpoint must run real cases and report honestly."""
    response = test_client.get("/api/v1/research/benchmark?max_cases=3")
    assert response.status_code == 200
    data = response.json()
    assert data["total_test_cases"] == 3
    # Honest bounds: pass rate can be anything from 0 to 100.
    assert 0.0 <= data["pass_rate_percentage"] <= 100.0
    assert 0.0 <= data["average_accuracy_percentage"] <= 100.0
    # Latency must be real (non-zero wall clock for a full pipeline run).
    assert data["average_latency_seconds"] > 0.0
    # The response must disclose measurement mode and methodology.
    assert data["llm_mode"] in ("llm", "extractive-fallback")
    assert "measured" in data["measurement_note"].lower()
    # Sample results carry per-case measured details.
    sample = data["sample_results"][0]
    assert "assertion_coverage" in sample
    assert "grounding_score" in sample
    assert "sources_retrieved" in sample
