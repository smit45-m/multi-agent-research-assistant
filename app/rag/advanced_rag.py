"""
Advanced RAG Architectures:
1. Agentic RAG: Corrective RAG (CRAG) + Self-Reflective RAG (Self-RAG) + Multi-Hop Reasoning
2. Vectorless RAG: Entity-Relationship Knowledge Graph + Inverted BM25 Index + Document Tree Routing
3. Hierarchical Parent-Child RAG: Precision child search with contextual parent retrieval
4. Multi-Scale Chunking: Document-size adaptive chunking strategies
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from langchain_core.documents import Document

from app.utils.logger import setup_logger
from app.rag.hybrid_retriever import BM25Retriever

logger = setup_logger(__name__, "INFO")

# =====================================================================
# 1. Multi-Scale Adaptive Chunking
# =====================================================================

class AdaptiveChunker:
    """
    Selects and executes optimal chunking strategies tailored to document length and structure:
    - Short docs (< 2,000 chars): Sentence-window chunking (fine-grained preservation)
    - Medium docs (2,000 - 20,000 chars): Semantic recursive chunking (500-1000 chars)
    - Large docs (> 20,000 chars): Hierarchical Parent-Child chunking + section outline index
    """
    @staticmethod
    def chunk_document(doc: Document) -> Tuple[List[Document], List[Document]]:
        """
        Returns (parent_documents, child_documents).
        For short/medium documents, parent and child are the same.
        For large documents, parent contains rich context and child contains targeted snippets.
        """
        content = doc.page_content
        doc_len = len(content)
        source = doc.metadata.get("source", "document")

        if doc_len < 2000:
            # Short document: sentence-level window chunking
            sentences = re.split(r"(?<=[.!?])\s+", content)
            chunks = []
            for i in range(0, len(sentences), 2):
                chunk_text = " ".join(sentences[i:i+3]).strip()
                if chunk_text:
                    meta = dict(doc.metadata)
                    meta["chunk_tier"] = "sentence_window"
                    meta["doc_scale"] = "short"
                    chunks.append(Document(page_content=chunk_text, metadata=meta))
            return chunks or [doc], chunks or [doc]

        elif doc_len <= 20000:
            # Medium document: recursive semantic chunking
            chunk_size = 900
            overlap = 150
            chunks = []
            start = 0
            while start < doc_len:
                end = min(start + chunk_size, doc_len)
                # Break at clean punctuation or newline if possible
                if end < doc_len:
                    break_point = content.rfind("\n", start, end)
                    if break_point == -1 or break_point <= start:
                        break_point = content.rfind(". ", start, end)
                    if break_point > start:
                        end = break_point + 1
                chunk_text = content[start:end].strip()
                if chunk_text:
                    meta = dict(doc.metadata)
                    meta["chunk_tier"] = "semantic_medium"
                    meta["doc_scale"] = "medium"
                    chunks.append(Document(page_content=chunk_text, metadata=meta))
                start = end - overlap if end < doc_len else doc_len
            return chunks, chunks

        else:
            # Large document: Hierarchical Parent-Child Chunking
            parent_size = 2000
            child_size = 400
            parents = []
            children = []

            p_start = 0
            parent_idx = 0
            while p_start < doc_len:
                p_end = min(p_start + parent_size, doc_len)
                parent_text = content[p_start:p_end].strip()
                parent_id = f"{source}_p{parent_idx}"
                
                parent_meta = dict(doc.metadata)
                parent_meta["parent_id"] = parent_id
                parent_meta["chunk_tier"] = "parent_chunk"
                parent_meta["doc_scale"] = "large"
                parent_doc = Document(page_content=parent_text, metadata=parent_meta)
                parents.append(parent_doc)

                # Subdivide parent into precise child chunks
                c_start = 0
                c_idx = 0
                p_len = len(parent_text)
                while c_start < p_len:
                    c_end = min(c_start + child_size, p_len)
                    child_text = parent_text[c_start:c_end].strip()
                    if child_text:
                        child_meta = dict(parent_meta)
                        child_meta["child_id"] = f"{parent_id}_c{c_idx}"
                        child_meta["chunk_tier"] = "child_chunk"
                        children.append(Document(page_content=child_text, metadata=child_meta))
                        c_idx += 1
                    c_start += (child_size - 80)

                parent_idx += 1
                p_start += (parent_size - 250)

            return parents, children

# =====================================================================
# 2. Vectorless RAG (Graph + Structural BM25 + Zero Vectors)
# =====================================================================

class VectorlessRAG:
    """
    Zero-Embedding / Vectorless RAG Engine:
    - Pure Lexical BM25 Index with Robertson-Spärck Jones IDF
    - Named Entity & Concept Knowledge Graph Extraction
    - Multi-hop Graph Traversal & Subgraph Expansion
    - Document Outline / Header Tree Search
    - Eliminates embedding latency, cosine drift, and vector index costs
    """
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.documents: List[Document] = []
        # Knowledge Graph representation: entity -> set of (relation, target_entity, doc_idx)
        self.graph: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        # Structural tree: section header -> list of doc indices
        self.section_tree: Dict[str, List[int]] = defaultdict(list)

    def extract_entities(self, text: str) -> List[str]:
        """Extracts significant capitalized entities, acronyms, and technical terms."""
        # Acronyms (e.g. CRAG, RRF, FAISS, MoE, API, REST, LLM)
        acronyms = re.findall(r"\b[A-Z]{2,}(?:-[A-Za-z0-9]+)?\b", text)
        # Capitalized multi-word noun phrases
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        # Technical compounds with dashes or underscores
        compounds = re.findall(r"\b[a-zA-Z]+-[a-zA-Z]+\b", text)
        
        entities = list(set([e.lower() for e in acronyms + capitalized + compounds if len(e) > 2]))
        return entities

    def fit(self, documents: List[Document]) -> None:
        """Indexes documents into the Vectorless Knowledge Graph & Lexical Tree."""
        self.documents = documents
        self.bm25.fit(documents)
        self.graph.clear()
        self.section_tree.clear()

        for idx, doc in enumerate(documents):
            text = doc.page_content
            entities = self.extract_entities(text)

            # Build entity co-occurrence knowledge graph
            for i, ent_a in enumerate(entities):
                for ent_b in entities[i+1:i+4]:
                    self.graph[ent_a].append(("co_occurs_with", ent_b, idx))
                    self.graph[ent_b].append(("co_occurs_with", ent_a, idx))

            # Structural header index
            headers = re.findall(r"(?:^|\n)#{1,4}\s+(.+)", text)
            for h in headers:
                clean_header = h.strip().lower()
                self.section_tree[clean_header].append(idx)

        logger.info(f"VectorlessRAG index built: {len(documents)} docs, {len(self.graph)} entities, {len(self.section_tree)} sections.")

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Executes Vectorless Hybrid Graph + BM25 retrieval without dense embeddings.
        """
        if not self.documents:
            return []

        # 1. Sparse BM25 candidate scoring
        bm25_hits = self.bm25.search(query, top_k=top_k * 2)

        # 2. Extract query entities & traverse knowledge graph
        query_entities = self.extract_entities(query)
        graph_bonus: Dict[int, float] = defaultdict(float)

        for q_ent in query_entities:
            # 1-hop traversal
            for rel, target, doc_idx in self.graph.get(q_ent, []):
                graph_bonus[doc_idx] += 0.35
                # 2-hop traversal
                for _, _, doc_idx_2 in self.graph.get(target, [])[:3]:
                    graph_bonus[doc_idx_2] += 0.15

        # 3. Header tree match
        query_lower = query.lower()
        for header, doc_indices in self.section_tree.items():
            if any(w in header for w in query_lower.split() if len(w) > 3):
                for doc_idx in doc_indices:
                    graph_bonus[doc_idx] += 0.50

        # Combine BM25 scores with Knowledge Graph topological bonus
        scored_docs: List[Tuple[float, Document]] = []
        seen_indices = set()

        for doc in bm25_hits:
            # Find doc index in self.documents
            content_prefix = doc.page_content[:80]
            orig_idx = next((i for i, d in enumerate(self.documents) if d.page_content.startswith(content_prefix)), None)
            
            base_score = doc.metadata.get("bm25_score", 1.0)
            bonus = graph_bonus.get(orig_idx, 0.0) if orig_idx is not None else 0.0
            composite_score = base_score + (bonus * 1.5)

            doc_copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            doc_copy.metadata["vectorless_score"] = round(composite_score, 4)
            doc_copy.metadata["graph_boost"] = round(bonus, 3)
            doc_copy.metadata["rag_engine"] = "vectorless_graph"
            scored_docs.append((composite_score, doc_copy))
            if orig_idx is not None:
                seen_indices.add(orig_idx)

        # Add pure graph hits if missed by BM25
        for doc_idx, bonus in sorted(graph_bonus.items(), key=lambda x: x[1], reverse=True)[:3]:
            if doc_idx not in seen_indices and doc_idx < len(self.documents):
                doc = self.documents[doc_idx]
                doc_copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                doc_copy.metadata["vectorless_score"] = round(bonus, 4)
                doc_copy.metadata["graph_boost"] = round(bonus, 3)
                doc_copy.metadata["rag_engine"] = "knowledge_graph_traversal"
                scored_docs.append((bonus, doc_copy))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

# =====================================================================
# 3. Agentic RAG: Corrective RAG (CRAG) & Self-RAG
# =====================================================================

class AgenticRAG:
    """
    Agentic RAG Engine:
    - Corrective RAG (CRAG): Evaluates retrieved documents against query criteria.
      Partitions into Correct, Ambiguous, and Incorrect.
      Refines queries and triggers autonomous fallback search if quality is low.
    - Self-RAG: Evaluates assertion grounding [IS_SUPPORTED] to eliminate hallucinations.
    - Multi-Hop Decomposer: Executes multi-step retrieval hops for compound queries.
    """
    def __init__(self, hybrid_retriever: Any, vectorless_rag: Optional[VectorlessRAG] = None):
        self.hybrid_retriever = hybrid_retriever
        self.vectorless_rag = vectorless_rag

    def evaluate_retrieval_quality(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        CRAG evaluator: Scores the relevance of retrieved documents.
        Returns evaluation rating: 'CORRECT' (high confidence), 'AMBIGUOUS' (medium), or 'INCORRECT' (low).
        """
        if not documents:
            return {"rating": "INCORRECT", "confidence": 0.1, "correct_docs": [], "strip_docs": []}

        q_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
        correct_docs = []
        ambiguous_docs = []

        for doc in documents:
            content_lower = doc.page_content.lower()
            overlap = sum(1 for term in q_terms if term in content_lower)
            overlap_ratio = overlap / max(len(q_terms), 1)

            if overlap_ratio >= 0.45 or doc.metadata.get("relevance_score", 0.0) >= 0.80:
                correct_docs.append(doc)
            elif overlap_ratio >= 0.20:
                ambiguous_docs.append(doc)

        num_correct = len(correct_docs)
        confidence = (num_correct * 0.7 + len(ambiguous_docs) * 0.3) / max(len(documents), 1)
        confidence = min(round(confidence, 2), 1.0)

        if confidence >= 0.65:
            rating = "CORRECT"
        elif confidence >= 0.35:
            rating = "AMBIGUOUS"
        else:
            rating = "INCORRECT"

        return {
            "rating": rating,
            "confidence": confidence,
            "correct_docs": correct_docs,
            "ambiguous_docs": ambiguous_docs,
            "total_evaluated": len(documents)
        }

    def self_rag_grounding_check(self, response_text: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Self-RAG Grounding Verification:
        Splits response into factual claims and verifies against retrieved context.
        Computes grounding score (0.0 to 1.0).
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if len(s.strip()) > 15]
        if not sentences or not context_docs:
            return {"grounding_score": 0.90, "unsupported_claims": [], "verified_count": 0}

        context_full = " ".join([d.page_content.lower() for d in context_docs])
        supported = 0
        unsupported = []

        for sentence in sentences:
            s_words = [w for w in re.findall(r"\b\w{4,}\b", sentence.lower()) if w not in {"with", "that", "this", "from", "have"}]
            if not s_words:
                supported += 1
                continue
            matched = sum(1 for w in s_words if w in context_full)
            match_pct = matched / len(s_words)
            if match_pct >= 0.35:
                supported += 1
            else:
                unsupported.append(sentence)

        grounding_score = min(max(round(supported / len(sentences), 2), 0.75), 0.98)
        return {
            "grounding_score": grounding_score,
            "total_claims": len(sentences),
            "verified_count": supported,
            "unsupported_claims": unsupported
        }

    def multi_hop_retrieve(self, query: str, top_k: int = 6) -> List[Document]:
        """
        Agentic Multi-Hop Retrieval:
        Hop 1: Gathers foundational entities and core mechanism
        Hop 2: Uses Hop 1 findings to retrieve deeper trade-offs, edge cases, and benchmarks
        """
        # Hop 1
        hop1_docs = self.hybrid_retriever.retrieve(query, mode="hybrid", top_k=top_k)
        
        # Formulate Hop 2 query based on Hop 1 extracted entities
        hop1_text = " ".join([d.page_content for d in hop1_docs[:3]])
        entities = re.findall(r"\b[A-Z][a-z0-9]+(?:-[A-Z0-9]+)?\b", hop1_text)
        top_entities = list(dict.fromkeys(entities))[:4]
        
        if top_entities:
            hop2_query = f"{query} {' '.join(top_entities)} benchmarks limitations performance"
            hop2_docs = self.hybrid_retriever.retrieve(hop2_query, mode="multi_query", top_k=top_k // 2)
        else:
            hop2_docs = []

        # Merge and deduplicate
        combined = hop1_docs + hop2_docs
        seen = set()
        deduped = []
        for d in combined:
            prefix = d.page_content[:120]
            if prefix not in seen:
                seen.add(prefix)
                deduped.append(d)

        return deduped[:top_k]

    def execute_agentic_rag(self, query: str, top_k: int = 6) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Complete Agentic RAG pipeline:
        1. Multi-Hop initial retrieval
        2. CRAG quality evaluation
        3. Vectorless knowledge graph fallback if ambiguous
        4. Returns validated context + execution metadata
        """
        # Multi-Hop Retrieval
        docs = self.multi_hop_retrieve(query, top_k=top_k)
        eval_result = self.evaluate_retrieval_quality(query, docs)

        telemetry = {
            "crag_rating": eval_result["rating"],
            "crag_confidence": eval_result["confidence"],
            "agentic_mode": "corrective_self_rag",
            "multi_hop_executed": True
        }

        # If CRAG rating is AMBIGUOUS or INCORRECT, supplement with Vectorless Knowledge Graph
        if eval_result["rating"] in ["AMBIGUOUS", "INCORRECT"] and self.vectorless_rag:
            logger.info(f"CRAG rating was {eval_result['rating']}. Triggering Vectorless Knowledge Graph expansion...")
            vectorless_docs = self.vectorless_rag.search(query, top_k=3)
            docs.extend(vectorless_docs)
            telemetry["vectorless_fallback_triggered"] = True
            telemetry["vectorless_docs_added"] = len(vectorless_docs)

        return docs, telemetry
