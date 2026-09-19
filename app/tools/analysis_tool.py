"""
Analysis tool for comparing sources and summarizing findings.
"""

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


class AnalysisTool:
    """Tool for analyzing and synthesizing information from multiple sources."""

    def __init__(self) -> None:
        """Initialize the analysis tool with an LLM."""
        from app.utils.llm import build_chat_llm

        self.llm = build_chat_llm(temperature=0.2)

    def compare_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple sources to identify agreements, contradictions, and key themes.

        Args:
            sources (List[Dict[str, Any]]): List of sources containing
                content and metadata.

        Returns:
            Dict[str, Any]: Analysis results including agreement_score,
                contradictions, and key_themes.
        """
        sources_text = json.dumps(sources, indent=2)

        system_prompt = """
        You are an expert analyst. Compare the provided sources and identify:
        1. Agreement score (0.0 to 1.0) representing how much the sources align.
        2. Contradictions or conflicting information between sources.
        3. Key themes present across the sources.
        
        Respond with a valid JSON object containing exactly these keys:
        - agreement_score (float)
        - contradictions (list of strings)
        - key_themes (list of strings)
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Sources:\n{sources_text}"),
        ]

        try:
            if self.llm is None:
                raise RuntimeError("No LLM configured")
            response = self.llm.invoke(messages)
            content = str(response.content).strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            parsed: Dict[str, Any] = json.loads(content)
            return parsed
        except Exception as e:
            logger.warning("Error comparing sources: %s", e)
            return {
                "agreement_score": 0.0,
                "contradictions": ["Error performing analysis."],
                "key_themes": [],
            }

    def summarize_findings(self, documents: List[str]) -> str:
        """
        Summarize a list of document contents into a cohesive finding.

        Args:
            documents (List[str]): List of document contents to summarize.

        Returns:
            str: A comprehensive summary.
        """
        docs_text = "\\n\\n---\\n\\n".join(documents)

        system_prompt = (
            "You are an expert summarizer. Provide a concise, "
            "comprehensive summary of the following documents."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Documents:\\n{docs_text}"),
        ]

        try:
            if self.llm is None:
                raise RuntimeError("No LLM configured")
            response = self.llm.invoke(messages)
            return str(response.content)
        except Exception as e:
            logger.warning("Error summarizing findings: %s", e)
            return "Error summarizing findings."
