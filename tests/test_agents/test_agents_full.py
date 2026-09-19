"""
Unit tests for the 6 autonomous agents and Advanced RAG components.

These tests assert MEASURED behavior: metrics must be computed from real
inputs, must respond to input quality, and must never be hardcoded.
"""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.critic_agent import CriticAgent
from app.agents.state import create_initial_state
from app.agents.verifier_agent import VerifierAgent
from app.agents.writer_agent import WriterAgent
from app.chains.router import ResearchRouter
from app.rag.hybrid_retriever import BM25Retriever, HybridRetriever


def test_initial_state_structure():
    state = create_initial_state("What are solid state batteries?", rag_mode="hybrid")
    assert state["query"] == "What are solid state batteries?"
    assert state["rag_mode"] == "hybrid"
    # Metrics must START at zero — they are measured, not preloaded.
    assert state["synthesis_speedup_ratio"] == 0.0
    assert state["confidence_score"] == 0.0
    assert state["response_accuracy_score"] == 0.0
    assert "planner_time_ms" in state["agent_telemetry"]
    assert "verifier_time_ms" in state["agent_telemetry"]
    assert "critic_time_ms" in state["agent_telemetry"]


def test_research_router_domains():
    router = ResearchRouter()
    d1 = router.route("What are recent arXiv papers on transformer attention?")
    assert d1.domain == "academic"
    assert "arxiv" in d1.target_source_formats

    d2 = router.route(
        "Explain credit default swaps, interest rate margins, and EBITDA."
    )
    assert d2.domain == "financial"
    assert "csv" in d2.target_source_formats or "xlsx" in d2.target_source_formats

    d3 = router.route("How to configure FastAPI with Docker and ASGI uvicorn workers?")
    assert d3.domain == "technical"
    assert "code" in d3.target_source_formats


def test_bm25_retriever_sparse_lexical_search():
    docs = [
        Document(
            page_content=(
                "Solid-state batteries use ceramic or polymer solid electrolytes."
            ),
            metadata={"source": "doc1"},
        ),
        Document(
            page_content=(
                "Lithium-ion batteries suffer from liquid electrolyte thermal runaway."
            ),
            metadata={"source": "doc2"},
        ),
        Document(
            page_content=(
                "Solar cells convert photon energy into direct current electricity."
            ),
            metadata={"source": "doc3"},
        ),
    ]
    bm25 = BM25Retriever()
    bm25.fit(docs)

    results = bm25.search("electrolyte solid state", top_k=2)
    assert len(results) > 0
    assert "solid" in results[0].page_content.lower()


def test_hybrid_retriever_reciprocal_rank_fusion():
    vstore = MagicMock()
    vstore.get_document_count.return_value = 2
    vstore.similarity_search.return_value = [
        Document(
            page_content="Doc A: High density battery",
            metadata={"source": "doc_a"},
        ),
        Document(
            page_content="Doc B: Low temperature battery",
            metadata={"source": "doc_b"},
        ),
    ]

    retriever = HybridRetriever(vstore, rrf_k=60)
    list1 = [
        Document(
            page_content="Doc A: High density battery",
            metadata={"source": "doc_a"},
        ),
        Document(
            page_content="Doc B: Low temperature battery",
            metadata={"source": "doc_b"},
        ),
    ]
    list2 = [
        Document(
            page_content="Doc B: Low temperature battery",
            metadata={"source": "doc_b"},
        ),
        Document(
            page_content="Doc A: High density battery",
            metadata={"source": "doc_a"},
        ),
    ]

    fused = retriever.reciprocal_rank_fusion([list1, list2], top_k=2)
    assert len(fused) == 2
    assert "rrf_score" in fused[0].metadata


def test_analyzer_confidence_is_measured_not_constant():
    """Confidence must respond to retrieval quality, not be a constant."""
    analyzer = AnalyzerAgent(llm=None)

    # Rich evidence: multiple relevant, substantive sources.
    rich = create_initial_state("Battery tech")
    rich["retrieved_documents"] = [
        {
            "source": f"https://arxiv.org/abs/230{i}",
            "title": f"Paper {i}",
            "content": "Solid-state batteries achieve 500 Wh/kg. " * 12,
            "relevance_score": 0.9,
        }
        for i in range(4)
    ]
    rich_conf = analyzer.analyze(rich)["confidence_score"]

    # Poor evidence: single weak source with thin content.
    poor = create_initial_state("Battery tech")
    poor["retrieved_documents"] = [
        {
            "source": "blog",
            "title": "Post",
            "content": "batteries",
            "relevance_score": 0.2,
        }
    ]
    poor_conf = analyzer.analyze(poor)["confidence_score"]

    assert rich_conf > poor_conf
    assert 0.0 <= poor_conf < rich_conf <= 1.0

    # No documents at all -> zero confidence, not a flattering default.
    empty = create_initial_state("Battery tech")
    assert analyzer.analyze(empty)["confidence_score"] == 0.0


def test_writer_produces_structured_report():
    writer = WriterAgent(llm=None)
    state = create_initial_state("Solid state batteries")
    state["analysis"] = {
        "key_findings": [
            "Solid state offers higher energy density",
            "Ceramic separators mitigate dendrites",
        ],
        "themes": ["Safety", "Energy Density"],
        "contradictions": ["No major contradictions found"],
    }
    state["retrieved_documents"] = [
        {
            "source": "https://arxiv.org/abs/2301",
            "title": "ArXiv Solid State",
            "content": "Data on 500 Wh/kg.",
            "relevance_score": 0.92,
            "source_type": "academic",
        }
    ]

    res = writer.write(state)
    assert res["status"] == "written"
    assert "# Executive Summary" in res["final_report"]
    assert len(res["sources_cited"]) > 0
    # The writer must NOT claim an accuracy score — that is measured
    # afterwards by the Verifier.
    assert res["response_accuracy_score"] == 0.0


def test_verifier_measures_grounding():
    """The verifier must reward grounded reports and punish fabrication."""
    verifier = VerifierAgent(llm=None)

    evidence = (
        "Solid-state batteries with ceramic electrolytes achieved 500 Wh/kg "
        "energy density in laboratory prototypes during 2024 testing."
    )

    grounded_state = create_initial_state("solid state batteries")
    grounded_state["retrieved_documents"] = [{"source": "arxiv", "content": evidence}]
    grounded_state["final_report"] = (
        "# Report\nSolid-state batteries with ceramic electrolytes achieved "
        "500 Wh/kg energy density in laboratory prototypes."
    )
    grounded = verifier.verify(grounded_state)

    fabricated_state = create_initial_state("solid state batteries")
    fabricated_state["retrieved_documents"] = [{"source": "arxiv", "content": evidence}]
    fabricated_state["final_report"] = (
        "# Report\nQuantum flux capacitors enable faster-than-light neutrino "
        "communication across intergalactic distances using tachyon fields."
    )
    fabricated = verifier.verify(fabricated_state)

    assert grounded["response_accuracy_score"] > fabricated["response_accuracy_score"]
    assert grounded["verification"]["grounded_sentences"] >= 1
    assert fabricated["verification"]["grounded_sentences"] == 0


def test_critic_flags_thin_reports():
    critic = CriticAgent(llm=None)

    weak = create_initial_state("test query")
    weak["final_report"] = "Just one unstructured sentence."
    weak["verification"] = {"grounded_ratio": 0.1, "ungrounded_sentences": []}
    weak_result = critic.critique(weak)
    assert weak_result["critique"]["quality_score"] < 0.7
    assert weak_result["critique"]["needs_revision"] is True

    strong = create_initial_state("test query")
    strong["final_report"] = (
        "# Executive Summary\nSummary here.\n"
        "# Detailed Findings\nFindings here.\n"
        "# Sources & Citations\nCitations here."
    )
    strong["verification"] = {"grounded_ratio": 0.95}
    strong["retrieved_documents"] = [{"content": "x"} for _ in range(5)]
    strong["sources_cited"] = [{"source": str(i)} for i in range(4)]
    strong_result = critic.critique(strong)
    assert strong_result["critique"]["quality_score"] >= 0.7
    assert strong_result["critique"]["needs_revision"] is False


def test_critic_respects_revision_budget():
    critic = CriticAgent(llm=None)
    state = create_initial_state("test query")
    state["final_report"] = "Thin."
    state["verification"] = {"grounded_ratio": 0.0, "ungrounded_sentences": []}
    state["revision_count"] = 5  # budget exhausted
    result = critic.critique(state)
    assert result["critique"]["needs_revision"] is False
