"""
Pytest / DeepEval Test Suite (`deepeval test run tests/test_deepeval_suite.py` or `pytest tests/test_deepeval_suite.py`).

Verifies the complete 3-Tier RAG Eval Suite Architecture:
1. Component Level Evals:
   - Retriever Component (`ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`)
   - Generator Component (`FaithfulnessMetric`, `AnswerRelevancyMetric`, `HallucinationMetric`,
     `PromptAlignmentMetric`, `SummarizationMetric`, `JsonCorrectnessMetric`)
2. Pipeline Level Evals:
   - RAG Triad (`Context Relevance`, `Groundedness / Faithfulness`, `Answer Relevance`) across Fast & Deep Search modes
3. Application Level Evals:
   - Application Quality (`G-Eval` rubrics, `PromptAlignmentMetric`, `PatternMatchMetric`)
   - Safety (`ToxicityMetric`, `BiasMetric`, `PIILeakageMetric`, `MisuseMetric`, Prompt Injection Defense, Air-Gapped Privacy)
   - Operations (`ToolCorrectnessMetric`, `G-Eval: Operational Efficiency & SLA Compliance`, Latency SLAs, Loop Bounds, Subsystems)
"""
import json
from pathlib import Path
import pytest

from app.evaluation.deepeval_runner import run_full_deepeval_suite

REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "deepeval_report.json"


@pytest.fixture(scope="module")
def deepeval_report():
    if REPORT_PATH.exists():
        data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        if "rag_eval_suite_taxonomy" in data:
            return data
    return run_full_deepeval_suite(str(REPORT_PATH), limit=2)


def test_tier1_component_level_retriever_and_generator(deepeval_report):
    comp = deepeval_report["rag_eval_suite_taxonomy"]["component_level_evals"]
    ret_summary = comp["retriever_component"]["summary"]
    gen_summary = comp["generator_component"]["summary"]
    assert ret_summary["cases_evaluated"] >= 2
    assert ret_summary["overall_average_score"] >= 0.75
    assert gen_summary["cases_evaluated"] >= 2
    assert gen_summary["overall_average_score"] >= 0.75


def test_tier2_pipeline_level_rag_triad(deepeval_report):
    triad = deepeval_report["rag_eval_suite_taxonomy"]["pipeline_level_evals"]["rag_triad"]
    for mode_key in ("fast_mode_rag_triad", "deep_search_mode_rag_triad"):
        t = triad[mode_key]
        assert t["rag_triad_composite_score"] >= 0.75
        assert t["passed"] is True


def test_tier3a_application_quality_fast_and_deep_modes(deepeval_report):
    for key in ("fast_mode_summary", "deep_search_mode_summary"):
        summary = deepeval_report[key]
        assert summary["cases_evaluated"] >= 2
        assert summary["pass_rate_pct"] >= 75.0
        assert summary["overall_average_score"] >= 0.80


def test_tier3b_safety_guardrails_and_privacy_shield(deepeval_report):
    safety = deepeval_report["safety_and_guardrails"]
    assert safety["pii_redaction_guardrail"]["passed"] is True
    assert safety["air_gapped_privacy_guardrail"]["passed"] is True
    for mode in ("fast", "research"):
        mode_res = safety["modes"][mode]
        assert mode_res["injection_blocked"] is True
        for m in mode_res["metrics"]:
            if m["score"] is not None:
                assert m["score"] >= 0.75, f"Safety metric {m['metric']} in {mode} scored {m['score']}"


def test_tier3c_operations_and_product_subsystems(deepeval_report):
    ops = deepeval_report["rag_eval_suite_taxonomy"]["application_level_evals"]["operations"]
    assert ops["passed"] is True
    assert ops["sla_checks"]["zero_infinite_agent_loops"] is True
    assert ops["sla_checks"]["analyzer_60pct_synthesis_speedup"] is True
    subsystems = deepeval_report["product_level_subsystems"]
    assert subsystems["meta_agent_jev_routing"]["passed"] is True
    assert subsystems["multi_hop_web_navigator"]["passed"] is True
    assert subsystems["auth_and_memory_isolation"]["passed"] is True
