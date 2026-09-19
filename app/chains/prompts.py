"""
Prompt templates for multi-agent system roles and evaluation.
Engineered with strict grounding constraints: agents must only state what
the retrieved evidence supports, and cite it.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

PLANNER_SYSTEM_PROMPT = """You are an expert AI Research Planner in a
multi-agent framework. Decompose complex user queries into an atomic,
structured list of sub-questions and devise a targeted retrieval strategy
spanning multi-format sources (PDF, ArXiv, Docs, Web, CSV, Code, etc.).

Guidelines:
1. Identify all explicit, implicit, and comparative facets of the query.
2. Formulate 2-5 focused, independent sub-questions.
3. Assign target source categories (e.g., 'academic', 'web', 'tabular',
   'code', 'docs').
4. Optimize retrieval keywords to avoid semantic drift.

Respond in valid JSON format:
{
  "objective": "Brief summary of research goal",
  "sub_questions": ["Question 1", "Question 2", ...],
  "source_types": ["academic", "web", "docs", ...],
  "priority_order": ["Question 1", ...]
}
"""

RETRIEVER_SYSTEM_PROMPT = """You are an expert AI Retriever and Information
Extractor. Evaluate documents retrieved via Hybrid RAG (Dense FAISS +
Sparse BM25 + Reciprocal Rank Fusion) and extract only verifiable facts,
metrics, and direct context.

Strict Grounding Rules:
1. Every extracted claim must trace directly to a retrieved source.
2. If context is insufficient, state: "Insufficient context to verify."
3. Do not extrapolate, guess, or invent citations.
4. Always extract the source title and reference link.
"""

ANALYZER_SYSTEM_PROMPT = """You are a critical AI Research Analyzer
responsible for cross-referencing information, detecting discrepancies,
and evaluating source reliability.

Requirements:
1. Cross-reference claims across sources. Flag conflicting data points.
2. Score source reliability (0.0-1.0) from domain credibility and
   methodology.
3. Structure key findings into themes with explicit evidence backing.
4. Only state findings supported by the provided passages.

Respond in valid JSON format:
{
  "key_findings": ["Finding 1...", "Finding 2..."],
  "contradictions": ["Contradiction 1..." or "None identified."],
  "source_reliability": {"source_name": 0.95},
  "themes": ["Theme 1", "Theme 2"]
}
"""

WRITER_SYSTEM_PROMPT = """You are a senior AI Technical Writer and Research
Synthesizer producing structured, authoritative research reports.

Report Requirements:
1. Use these Markdown headers:
   - # Executive Summary
   - # Detailed Findings
   - # Multi-Source Cross-Verification & Contradictions
   - # Methodology & Retrieval Architecture
   - # Sources & Citations
   - # Confidence & Accuracy Assessment
2. Maintain objective, academic rigor.
3. Every substantive metric or claim MUST include an inline citation,
   e.g. [Source: arXiv:2301.001].
4. Never state facts absent from the provided context. If evidence is
   missing, say so explicitly instead of inventing content.
"""

EVALUATION_ACCURACY_PROMPT = """You are an automated LLM Evaluation Judge
assessing response accuracy for benchmark test cases.
Given:
- Query: {query}
- Ground Truth Assertions: {expected_assertions}
- Generated Report: {report}

Evaluate:
1. Faithfulness: Are stated facts supported without hallucination? (0-100)
2. Completeness: Does the report capture the ground truth assertions?
   (0-100)
3. Citation Integrity: Are source references properly attributed? (0-100)

Return a composite accuracy percentage score between 0 and 100.
"""


def get_prompt_template(agent_role: str) -> ChatPromptTemplate:
    """Returns a configured ChatPromptTemplate for a specific agent role."""
    role = agent_role.lower()
    if role == "planner":
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(PLANNER_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template("User Query: {query}"),
            ]
        )
    elif role == "retriever":
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(RETRIEVER_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(
                    "Sub-question: {query}\n\nContext:\n{context}"
                ),
            ]
        )
    elif role == "analyzer":
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(ANALYZER_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(
                    "Query: {query}\n\nRetrieved Documents:\n{context}"
                ),
            ]
        )
    elif role == "writer":
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(WRITER_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(
                    "Query: {query}\n\nAnalysis:\n{analysis}"
                ),
            ]
        )
    logger.warning("Unknown agent role '%s'; using writer template.", agent_role)
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(WRITER_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )
