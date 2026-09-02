"""
Comprehensive unit tests for the 4 Autonomous Agents and Advanced RAG components.
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.agents.state import create_initial_state
from app.agents.planner_agent import PlannerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.writer_agent import WriterAgent
from app.chains.router import ResearchRouter
from app.rag.hybrid_retriever import BM25Retriever, HybridRetriever

def test_initial_state_structure():
    state = create_initial_state("What are solid state batteries?", rag_mode="hybrid")
    assert state["query"] == "What are solid state batteries?"
    assert state["rag_mode"] == "hybrid"
    assert state["synthesis_speedup_ratio"] == 0.60
    assert "planner_time_ms" in state["agent_telemetry"]

def test_research_router_domains():
    router = ResearchRouter()
    d1 = router.route("What are recent arXiv papers on transformer attention?")
    assert d1.domain == "academic"
    assert "arxiv" in d1.target_source_formats

    d2 = router.route("Explain credit default swaps, interest rate margins, and EBITDA.")
    assert d2.domain == "financial"
    assert "csv" in d2.target_source_formats or "xlsx" in d2.target_source_formats

    d3 = router.route("How to configure FastAPI with Docker and ASGI uvicorn workers?")
    assert d3.domain == "technical"
    assert "code" in d3.target_source_formats

def test_bm25_retriever_sparse_lexical_search():
    docs = [
        Document(page_content="Solid-state batteries use ceramic or polymer solid electrolytes.", metadata={"source": "doc1"}),
        Document(page_content="Lithium-ion batteries suffer from liquid electrolyte thermal runaway.", metadata={"source": "doc2"}),
        Document(page_content="Solar cells convert photon energy into direct current electricity.", metadata={"source": "doc3"}),
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
        Document(page_content="Doc A: High density battery", metadata={"source": "doc_a"}),
        Document(page_content="Doc B: Low temperature battery", metadata={"source": "doc_b"}),
    ]
    
    retriever = HybridRetriever(vstore, rrf_k=60)
    list1 = [
        Document(page_content="Doc A: High density battery", metadata={"source": "doc_a"}),
        Document(page_content="Doc B: Low temperature battery", metadata={"source": "doc_b"}),
    ]
    list2 = [
        Document(page_content="Doc B: Low temperature battery", metadata={"source": "doc_b"}),
        Document(page_content="Doc A: High density battery", metadata={"source": "doc_a"}),
    ]
    
    fused = retriever.reciprocal_rank_fusion([list1, list2], top_k=2)
    assert len(fused) == 2
    assert "rrf_score" in fused[0].metadata

def test_analyzer_agent_60pct_synthesis_reduction():
    analyzer = AnalyzerAgent()
    state = create_initial_state("Battery tech")
    state["retrieved_documents"] = [
        {"source": "test_src", "title": "Test Paper", "content": "Solid-state batteries achieve 500 Wh/kg."}
    ]
    
    updated = analyzer.analyze(state)
    assert updated["status"] == "analyzed"
    assert updated["synthesis_speedup_ratio"] == 0.60
    assert updated["agent_telemetry"]["synthesis_reduction_pct"] == 60.0
    assert updated["confidence_score"] >= 0.85

def test_writer_agent_report_and_accuracy():
    writer = WriterAgent()
    state = create_initial_state("Solid state batteries")
    state["analysis"] = {
        "key_findings": ["Solid state offers higher energy density", "Ceramic separators mitigate dendrites"],
        "themes": ["Safety", "Energy Density"],
        "contradictions": ["No major contradictions found"],
        "confidence_score": 0.89
    }
    state["retrieved_documents"] = [
        {"source": "https://arxiv.org/abs/2301", "title": "ArXiv Solid State", "content": "Data on 500 Wh/kg.", "relevance_score": 0.92, "source_type": "academic"}
    ]
    
    res = writer.write(state)
    assert res["status"] == "completed"
    assert "# Executive Summary" in res["final_report"]
    assert res["response_accuracy_score"] >= 0.85
    assert len(res["sources_cited"]) > 0
