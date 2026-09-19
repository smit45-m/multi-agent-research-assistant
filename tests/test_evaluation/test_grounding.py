"""
Tests for the deterministic grounding/coverage measurement utilities.
"""

from app.evaluation.grounding import (
    assertion_coverage,
    extract_claim_text,
    measure_report_grounding,
    sentence_grounding,
)


def test_grounded_sentence_scores_high():
    sources = [
        "FlashAttention reduces memory bandwidth usage by tiling the "
        "attention computation into SRAM-sized blocks, avoiding redundant "
        "reads and writes to HBM."
    ]
    grounded, score = sentence_grounding(
        "FlashAttention tiles the attention computation into SRAM-sized "
        "blocks to reduce memory bandwidth.",
        sources,
    )
    assert grounded
    assert score > 0.5


def test_fabricated_sentence_scores_low():
    sources = ["FlashAttention reduces memory bandwidth usage by tiling attention."]
    grounded, score = sentence_grounding(
        "Unicorn-based quantum teleportation enables intergalactic pizza "
        "delivery through wormhole logistics networks.",
        sources,
    )
    assert not grounded
    assert score < 0.18


def test_report_grounding_mixed():
    sources = [
        "Solid-state batteries with ceramic electrolytes reached 500 Wh/kg "
        "in laboratory prototypes."
    ]
    report = (
        "Solid-state batteries with ceramic electrolytes reached 500 Wh/kg "
        "in laboratory prototypes during recent testing. "
        "Meanwhile, dragons authenticated the blockchain using interpretive "
        "dance and telepathic consensus mechanisms across seven dimensions."
    )
    result = measure_report_grounding(report, sources)
    assert result["total_sentences"] == 2
    assert result["grounded_sentences"] == 1
    assert result["grounded_ratio"] == 0.5
    assert len(result["ungrounded"]) == 1


def test_empty_report_grounding():
    result = measure_report_grounding("", ["some source"])
    assert result["grounded_ratio"] == 0.0
    assert result["total_sentences"] == 0


def test_assertion_coverage_matches_present_terms():
    text = (
        "The router uses sparse top-k gating with expert capacity limits "
        "and load balancing losses for token routing."
    )
    result = assertion_coverage(
        text,
        ["sparse top-k", "expert capacity", "load balancing", "quantum leap"],
    )
    assert "sparse top-k" in result["matched"]
    assert "expert capacity" in result["matched"]
    assert "load balancing" in result["matched"]
    assert "quantum leap" in result["missing"]
    assert result["coverage"] == 0.75


def test_assertion_coverage_is_not_self_referential():
    """
    Regression test for the original fabricated benchmark: coverage must be
    measured against the TEXT, never against the assertions themselves.
    """
    result = assertion_coverage("completely unrelated text", ["kernel tiling"])
    assert result["coverage"] == 0.0

    empty = assertion_coverage("", ["kernel tiling", "SRAM"])
    assert empty["coverage"] == 0.0


def test_assertion_coverage_handles_morphology():
    text = "The model applies quantization with per-channel scaling factors."
    result = assertion_coverage(text, ["quantized scaling"])
    assert result["coverage"] == 1.0


def test_extract_claim_text_drops_metadata_sections():
    report = (
        "# Executive Summary\n"
        "The system uses kernel tiling for attention.\n"
        "# Detailed Findings\n"
        "SRAM blocks avoid HBM round trips.\n"
        "# Methodology & Retrieval Architecture\n"
        "- Multi-agent pipeline with six agents.\n"
        "# Sources & Citations\n"
        "- some/local/path.md\n"
        "# Confidence & Accuracy Assessment\n"
        "- Retrieval confidence: 0.42\n"
    )
    claims = extract_claim_text(report)
    assert "kernel tiling" in claims
    assert "SRAM blocks" in claims
    # Metadata sections describe the pipeline, not the topic.
    assert "Multi-agent pipeline" not in claims
    assert "some/local/path.md" not in claims
    assert "Retrieval confidence" not in claims
    # Headers themselves are labels, not claims.
    assert "Executive Summary" not in claims


def test_grounding_ignores_methodology_and_sources_sections():
    sources = ["Kernel tiling keeps attention blocks resident in SRAM."]
    report = (
        "# Detailed Findings\n"
        "Kernel tiling keeps attention blocks resident in SRAM.\n"
        "# Methodology & Retrieval Architecture\n"
        "- This sentence describes the pipeline and matches no source text "
        "at all whatsoever.\n"
    )
    result = measure_report_grounding(report, sources)
    assert result["grounded_ratio"] == 1.0
