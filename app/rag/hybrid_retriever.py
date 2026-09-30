"""BM25, dense/RRF, diversified queries and neighboring-context retrieval."""
import math
import re
import threading
from collections import Counter, defaultdict
from langchain_core.documents import Document
from app.rag.relevance import tokens, fingerprint, relevance


def source_allowed(metadata, filters):
    if not filters:
        return True
    category = str(metadata.get("format", metadata.get("source_type", "document"))).lower().lstrip(".")
    if category in {"py", "js", "ts", "sh"}:
        category = "code"
    elif category in {"yaml", "yml", "xml"}:
        category = "configs"
    return category in filters or ("document" in filters and category not in {"web", "wikipedia", "arxiv", "image", "audio"})


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.fit([])

    def _tokenize(self, text):
        return tokens(text)

    def fit(self, documents):
        self.documents = list(documents)
        self.corpus_size = len(documents)
        self.doc_freqs = [Counter(tokens(d.page_content + " " + str(d.metadata.get("title", "")))) for d in documents]
        self.doc_lens = [sum(freq.values()) for freq in self.doc_freqs]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)
        df = Counter(term for freq in self.doc_freqs for term in freq)
        self.idf = {term: math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5)) for term, n in df.items()}

    def search(self, query, top_k=5):
        terms = set(tokens(query))
        if not terms or top_k < 1:
            return []
        scored = []
        for index, freq in enumerate(self.doc_freqs):
            norm = 1 - self.b + self.b * self.doc_lens[index] / max(self.avgdl, 1)
            score = sum(self.idf.get(t, 0) * freq[t] * (self.k1 + 1) / (freq[t] + self.k1 * norm) for t in terms if freq[t])
            if score > 0:
                doc = self.documents[index]
                scored.append((score, index, Document(page_content=doc.page_content, metadata={**doc.metadata, "bm25_score": score})))
        return [doc for _, _, doc in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]]


class HybridRetriever:
    def __init__(self, vector_store, rrf_k=60, min_relevance=0.18):
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.min_relevance = min_relevance
        self.bm25 = BM25Retriever()
        self._vectorless = None
        self._agentic = None
        self._version = None
        self._lock = threading.RLock()
        self.last_diagnostics = {}

    def _get_advanced_rag(self, documents):
        if self._vectorless is None:
            try:
                from app.rag.advanced_rag import VectorlessRAG, AgenticRAG
                self._vectorless = VectorlessRAG()
                self._vectorless.fit(documents)
                self._agentic = AgenticRAG(hybrid_retriever=self, vectorless_rag=self._vectorless)
            except Exception:
                self._vectorless = None
                self._agentic = None
        return self._vectorless, self._agentic

    def sync_bm25(self):
        with self._lock:
            version = getattr(self.vector_store, "version", None)
            documents = self.vector_store.get_all_documents()
            if not isinstance(documents, list):
                documents = []
            signature = version if isinstance(version, int) else tuple(fingerprint(d.page_content) for d in documents)
            if self._version != signature:
                self.bm25.fit(documents)  # Must clear stale documents after deletions, even at count zero.
                if self._vectorless is not None:
                    self._vectorless.fit(documents)
                self._version = signature

    def generate_multi_queries(self, query):
        chunks = [query]
        parts = re.split(r"\s+(?:versus|vs\.?|compared to|and)\s+|[?;\n]+", query, flags=re.I)
        chunks.extend(p.strip() for p in parts if len(tokens(p)) >= 2)
        return list(dict.fromkeys(chunks))[:4]

    def reciprocal_rank_fusion(self, ranked_lists, top_k=8):
        scores, mapping = defaultdict(float), {}
        for ranked in ranked_lists:
            seen = set()
            for rank, doc in enumerate(ranked, 1):
                key = fingerprint(doc.page_content)
                if key in seen:
                    continue
                seen.add(key)
                scores[key] += 1.0 / (self.rrf_k + rank)
                if key not in mapping:
                    mapping[key] = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                else:
                    mapping[key].metadata.update(doc.metadata)
        results = []
        for key in sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]:
            doc = mapping[key]
            doc.metadata["rrf_score"] = scores[key]
            results.append(doc)
        return results

    def retrieve(self, query, mode="hybrid", top_k=6, source_filters=None):
        with self._lock:
            self.sync_bm25()
            documents = [d for d in self.bm25.documents if source_allowed(d.metadata, source_filters)]
            sparse = self.bm25
            if source_filters:
                sparse = BM25Retriever()
                sparse.fit(documents)
            requested = mode
            if mode == "auto":
                mode = "hybrid"
            warnings = []

            # Specialized execution for Vectorless Knowledge Graph mode
            if mode == "vectorless":
                vectorless, _ = self._get_advanced_rag(documents)
                if vectorless and documents:
                    result = vectorless.search(query, top_k=top_k)
                    result = [d for d in result if source_allowed(d.metadata, source_filters)]
                    for doc in result:
                        if "relevance_score" not in doc.metadata:
                            doc.metadata["relevance_score"] = relevance(query, doc.page_content, str(doc.metadata.get("title", "")))
                    actual = "vectorless_graph+bm25"
                    self.last_diagnostics = {"requested": requested, "actual": actual, "candidate_count": len(result),
                                             "accepted_count": len(result), "queries": [query], "warnings": warnings}
                    return result

            # Specialized execution for Agentic Corrective/Self-RAG mode
            if mode == "agentic":
                _, agentic = self._get_advanced_rag(documents)
                if agentic and documents:
                    result, agentic_telemetry = agentic.execute_agentic_rag(query, top_k=top_k)
                    result = [d for d in result if source_allowed(d.metadata, source_filters)]
                    for doc in result:
                        if "relevance_score" not in doc.metadata:
                            doc.metadata["relevance_score"] = relevance(query, doc.page_content, str(doc.metadata.get("title", "")))
                    actual = "agentic_crag_self_rag"
                    self.last_diagnostics = {"requested": requested, "actual": actual, "candidate_count": len(result),
                                             "accepted_count": len(result), "queries": [query], "warnings": warnings,
                                             "agentic_telemetry": agentic_telemetry}
                    return result

            dense_available = getattr(self.vector_store, "dense_available", False) is True
            use_dense = mode not in {"bm25", "vectorless", "hierarchical"} and dense_available
            if mode in {"hybrid", "rrf", "multi_query", "agentic", "vector"} and not dense_available:
                warnings.append("Dense semantic retrieval is unavailable; used BM25 rather than pretending to run embeddings.")
            queries = self.generate_multi_queries(query) if mode in {"multi_query", "agentic"} else [query]
            lists = []
            for question in queries:
                if use_dense:
                    try:
                        dense = self.vector_store.similarity_search(question, k=max(top_k * 4, 20), score_threshold=0.25)
                        lists.append([d for d in dense if source_allowed(d.metadata, source_filters)])
                    except Exception:
                        warnings.append("Dense search failed; kept available lexical evidence.")
                if mode != "vector" or not dense_available:
                    lists.append(sparse.search(question, max(top_k * 3, 12)))
            fused = self.reciprocal_rank_fusion(lists, max(top_k * 4, 20))
            ranked = []
            for doc in fused:
                lexical = relevance(query, doc.page_content, str(doc.metadata.get("title", "")))
                semantic = float(doc.metadata.get("dense_score", 0))
                # Independent relevance gate: RRF rank alone is not relevance/confidence.
                if lexical < self.min_relevance and semantic < 0.50:
                    continue
                score = max(lexical, 0.85 * semantic)
                doc.metadata["relevance_score"] = score
                ranked.append((score + 0.25 * doc.metadata.get("rrf_score", 0), doc))
            ranked.sort(key=lambda item: item[0], reverse=True)
            result = [doc for _, doc in ranked[:top_k]]
            if mode == "hierarchical":
                expanded = []
                for doc in result:
                    position = doc.metadata.get("chunk_index")
                    parent = doc.metadata.get("document_id")
                    neighbors = [d for d in documents if parent and d.metadata.get("document_id") == parent
                                 and isinstance(position, int) and isinstance(d.metadata.get("chunk_index"), int)
                                 and abs(d.metadata["chunk_index"] - position) <= 1]
                    if neighbors:
                        neighbors.sort(key=lambda d: d.metadata["chunk_index"])
                        doc = Document(page_content="\n\n".join(d.page_content for d in neighbors), metadata={**doc.metadata, "context_expanded": True})
                    expanded.append(doc)
                result = expanded
            actual = "dense+bm25+rrf" if use_dense and mode != "vector" else "dense" if use_dense else "bm25"
            if mode == "hierarchical":
                actual += "+neighbor_context"
            if len(queries) > 1:
                actual += "+multi_query"
            if mode == "vectorless":
                warnings.append("Vectorless graph index was empty; fell back to lexical retrieval.")
            self.last_diagnostics = {"requested": requested, "actual": actual, "candidate_count": len(fused),
                                     "accepted_count": len(result), "queries": queries, "warnings": warnings}
            return result
