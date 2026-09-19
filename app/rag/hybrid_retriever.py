"""
Advanced Hybrid Retrieval module combining:
1. Dense Vector Retrieval (FAISS embeddings)
2. Sparse Keyword Retrieval (BM25)
3. Reciprocal Rank Fusion (RRF)
4. Multi-Query Expansion
5. Vectorless Knowledge Graph Retrieval
6. Agentic Corrective RAG (CRAG & Self-RAG)
"""

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.rag.vector_store import VectorStoreManager
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class BM25Retriever:
    """
    Lightweight, robust, zero-dependency BM25 retriever for sparse lexical search.
    Handles exact keyword matching, token frequencies, and inverse document frequencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple, fast regex tokenizer with lowercase normalization."""
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, documents: List[Document]) -> None:
        """Indexes documents into the BM25 corpus."""
        self.documents = documents
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            self.avgdl = 0.0
            return

        self.doc_freqs = []
        self.doc_lens = []
        df: Dict[str, int] = defaultdict(int)

        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_lens.append(len(tokens))
            freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            self.doc_freqs.append(freq)
            for token in freq.keys():
                df[token] += 1

        self.avgdl = (
            sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0.0
        )

        # Calculate standard Robertson-Spärck Jones IDF
        self.idf = {}
        for token, doc_count in df.items():
            self.idf[token] = math.log(
                (self.corpus_size - doc_count + 0.5) / (doc_count + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Scores and returns top-k documents matching the query using BM25."""
        if self.corpus_size == 0 or not self.documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return self.documents[:top_k]

        scores: List[float] = [0.0] * self.corpus_size
        for i, freq in enumerate(self.doc_freqs):
            doc_len = self.doc_lens[i]
            len_norm = 1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0))
            for token in tokens:
                if token in freq:
                    f = freq[token]
                    idf_val = self.idf.get(token, 0.1)
                    score_term = (
                        idf_val * (f * (self.k1 + 1.0)) / (f + self.k1 * len_norm)
                    )
                    scores[i] += score_term

        # Rank indices by score descending
        ranked_indices = sorted(
            range(self.corpus_size), key=lambda i: scores[i], reverse=True
        )
        results: List[Document] = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0.0 or len(results) == 0:
                doc_copy = Document(
                    page_content=self.documents[idx].page_content,
                    metadata=dict(self.documents[idx].metadata),
                )
                doc_copy.metadata["bm25_score"] = round(scores[idx], 4)
                results.append(doc_copy)
        return results


class HybridRetriever:
    """
    Production-grade Multi-Paradigmatic Retriever implementing:
    - Dense Vector similarity (FAISS)
    - Sparse Lexical matching (BM25)
    - Reciprocal Rank Fusion (RRF)
    - Multi-Query Expansion
    - Vectorless Knowledge Graph RAG
    - Agentic Corrective RAG (CRAG) & Self-RAG
    """

    def __init__(self, vector_store: VectorStoreManager, rrf_k: int = 60):
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever()
        self._synced_doc_count = -1
        self._vectorless_rag: Optional[Any] = None
        self._agentic_rag: Optional[Any] = None

    def _get_vectorless_rag(self) -> Any:
        if self._vectorless_rag is None:
            from app.rag.advanced_rag import VectorlessRAG

            self._vectorless_rag = VectorlessRAG()
            broad_docs = self.vector_store.similarity_search(
                "", k=max(self.vector_store.get_document_count(), 50)
            )
            if broad_docs:
                self._vectorless_rag.fit(broad_docs)
        return self._vectorless_rag

    def _get_agentic_rag(self) -> Any:
        if self._agentic_rag is None:
            from app.rag.advanced_rag import AgenticRAG

            self._agentic_rag = AgenticRAG(self, self._get_vectorless_rag())
        return self._agentic_rag

    def sync_bm25(self) -> None:
        """Syncs in-memory BM25 index with current vector store contents."""
        total = self.vector_store.get_document_count()
        if total != self._synced_doc_count:
            broad_docs = self.vector_store.similarity_search("", k=max(total, 50))
            if broad_docs:
                self.bm25.fit(broad_docs)
                self._synced_doc_count = total
                if self._vectorless_rag is not None:
                    self._vectorless_rag.fit(broad_docs)
                logger.info(
                    f"BM25 and Vectorless indexes synced with "
                    f"{len(broad_docs)} documents."
                )

    def generate_multi_queries(self, query: str) -> List[str]:
        """
        Expands original query into multiple semantic sub-queries to
        broaden search coverage.
        """
        clean_q = query.strip()
        queries = [clean_q]

        if (
            "vs" in clean_q.lower()
            or "compare" in clean_q.lower()
            or "difference" in clean_q.lower()
        ):
            queries.append(f"advantages disadvantages comparison {clean_q}")
            queries.append(f"technical benchmarks performance {clean_q}")
        elif (
            "how to" in clean_q.lower()
            or "architecture" in clean_q.lower()
            or "workflow" in clean_q.lower()
        ):
            queries.append(f"best practices system design {clean_q}")
            queries.append(f"implementation specifications {clean_q}")
        else:
            queries.append(f"key concepts overview analysis {clean_q}")
            queries.append(f"state of the art technical details {clean_q}")

        return queries

    def reciprocal_rank_fusion(
        self, ranked_lists: List[List[Document]], top_k: int = 8
    ) -> List[Document]:
        """
        Merges multiple ranked document lists into a single consolidated ranking
        using Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = sum_m( 1 / (k + rank_m(d)) )
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                src_key = doc.metadata.get(
                    "source_path", doc.metadata.get("source", "unknown")
                )
                doc_key = f"{src_key}:{doc.page_content[:150]}"
                rrf_scores[doc_key] += 1.0 / (self.rrf_k + rank)
                if doc_key not in doc_map:
                    doc_map[doc_key] = doc

        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
        )
        fused_docs = []
        for key in sorted_keys[:top_k]:
            doc = doc_map[key]
            doc_copy = Document(
                page_content=doc.page_content, metadata=dict(doc.metadata)
            )
            doc_copy.metadata["rrf_score"] = round(rrf_scores[key], 5)
            fused_docs.append(doc_copy)

        return fused_docs

    def retrieve(
        self, query: str, mode: str = "hybrid", top_k: int = 6
    ) -> List[Document]:
        """
        Executes retrieval according to the chosen RAG mode:
        - 'agentic': Corrective RAG (CRAG) + Self-RAG grounding + Multi-hop reasoning
        - 'vectorless': Zero-vector BM25 + Knowledge Graph entity traversal
        - 'hierarchical': Multi-scale parent-child chunking for large document synthesis
        - 'vector': Dense FAISS similarity search only
        - 'bm25': Sparse lexical BM25 keyword search only
        - 'multi_query': Multi-query expansion + RRF across all sub-queries
        - 'hybrid': Dense FAISS + Sparse BM25 fused via Reciprocal Rank Fusion (k=60)
        """
        self.sync_bm25()
        mode = mode.lower()

        # 1. Agentic RAG
        if mode == "agentic":
            agentic_engine = self._get_agentic_rag()
            docs, _ = agentic_engine.execute_agentic_rag(query, top_k=top_k)
            return list(docs)

        # 2. Vectorless RAG
        if mode == "vectorless":
            vectorless_engine = self._get_vectorless_rag()
            return list(vectorless_engine.search(query, top_k=top_k))

        # 3. Hierarchical Parent-Child RAG
        if mode == "hierarchical":
            # Retrieve targeted child chunks then expand to parent context
            child_results = self.bm25.search(query, top_k=top_k)
            parent_ids = {
                d.metadata.get("parent_id")
                for d in child_results
                if "parent_id" in d.metadata
            }
            if parent_ids:
                # Fetch full parent chunks if indexed
                broad_docs = self.vector_store.similarity_search(
                    "", k=max(self.vector_store.get_document_count(), 50)
                )
                parent_docs = [
                    d for d in broad_docs if d.metadata.get("parent_id") in parent_ids
                ]
                if parent_docs:
                    return parent_docs[:top_k]
            return child_results[:top_k]

        # 4. Dense Vector Only
        if mode == "vector":
            return self.vector_store.similarity_search(query, k=top_k)

        # 5. Sparse BM25 Only
        if mode == "bm25":
            return self.bm25.search(query, top_k=top_k)

        # 6. Multi-Query Expansion
        if mode == "multi_query":
            expanded_queries = self.generate_multi_queries(query)
            ranked_runs = []
            for q in expanded_queries:
                dense_results = self.vector_store.similarity_search(q, k=top_k)
                sparse_results = self.bm25.search(q, top_k=top_k)
                if dense_results:
                    ranked_runs.append(dense_results)
                if sparse_results:
                    ranked_runs.append(sparse_results)
            if not ranked_runs:
                return []
            return self.reciprocal_rank_fusion(ranked_runs, top_k=top_k)

        # Default: 'hybrid' / 'rrf'
        dense_results = self.vector_store.similarity_search(query, k=top_k)
        sparse_results = self.bm25.search(query, top_k=top_k)

        ranked_runs = []
        if dense_results:
            ranked_runs.append(dense_results)
        if sparse_results:
            ranked_runs.append(sparse_results)

        if not ranked_runs:
            return []

        return self.reciprocal_rank_fusion(ranked_runs, top_k=top_k)
