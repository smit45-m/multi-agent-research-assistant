"""
Meta-tests guaranteeing the benchmark measures the real system.

These tests exist to prevent regression to fabricated metrics: the
benchmark must invoke the actual pipeline, its latency must be real
wall-clock time, and its scores must respond to actual output quality.
"""

import time
from unittest.mock import patch

from app.evaluation.benchmark_runner import BenchmarkEvaluator

_CASE = {
    "id": "TC-TEST",
    "category": "Test",
    "query": "solid state battery electrolytes",
    "target_source_format": "pdf",
    "expected_assertions": ["ceramic electrolyte", "energy density"],
    "min_accuracy_threshold": 0.85,
    "target_latency_s": 8.0,
}


def _fake_state(report: str, docs: list, grounded_ratio: float) -> dict:
    return {
        "final_report": report,
        "retrieved_documents": docs,
        "verification": {"grounded_ratio": grounded_ratio},
        "synthesis_speedup_ratio": 0.5,
    }


def test_benchmark_invokes_real_pipeline():
    """evaluate_case must call graph.run with the case query."""
    evaluator = BenchmarkEvaluator()
    with patch.object(evaluator, "_get_graph") as mock_get:
        mock_get.return_value.run.return_value = _fake_state(
            "ceramic electrolyte improves energy density",
            [{"content": "ceramic electrolyte energy density study"}],
            0.9,
        )
        result = evaluator.evaluate_case(_CASE)
        mock_get.return_value.run.assert_called_once()
        assert mock_get.return_value.run.call_args[0][0] == _CASE["query"]
    assert result.accuracy_score > 0.0


def test_benchmark_latency_is_wall_clock():
    """Latency must reflect real elapsed time, not a simulated constant."""
    evaluator = BenchmarkEvaluator()

    def slow_run(query, rag_mode="hybrid"):
        time.sleep(0.25)
        return _fake_state("report text", [], 0.0)

    with patch.object(evaluator, "_get_graph") as mock_get:
        mock_get.return_value.run.side_effect = slow_run
        result = evaluator.evaluate_case(_CASE)

    assert 0.2 <= result.latency_seconds < 2.0


def test_benchmark_accuracy_responds_to_output_quality():
    """Good output must score higher than bad output — no calibration."""
    evaluator = BenchmarkEvaluator()

    good = _fake_state(
        "The ceramic electrolyte design achieved high energy density.",
        [{"content": "ceramic electrolyte energy density measurements"}],
        0.95,
    )
    bad = _fake_state(
        "Nothing relevant here.",
        [{"content": "unrelated cooking recipes"}],
        0.0,
    )

    with patch.object(evaluator, "_get_graph") as mock_get:
        mock_get.return_value.run.return_value = good
        good_result = evaluator.evaluate_case(_CASE)
        mock_get.return_value.run.return_value = bad
        bad_result = evaluator.evaluate_case(_CASE)

    assert good_result.accuracy_score > bad_result.accuracy_score
    assert bad_result.accuracy_score < 50.0
    assert not bad_result.passed


def test_benchmark_pipeline_crash_is_a_failure():
    """A pipeline crash must be reported as a failed case, not hidden."""
    evaluator = BenchmarkEvaluator()
    with patch.object(evaluator, "_get_graph") as mock_get:
        mock_get.return_value.run.side_effect = RuntimeError("boom")
        result = evaluator.evaluate_case(_CASE)
    assert not result.passed
    assert result.accuracy_score == 0.0
    assert "boom" in result.details
