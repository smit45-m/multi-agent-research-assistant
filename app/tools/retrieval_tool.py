"""
Vector store retrieval tool.
"""

from typing import Any, Dict, List

from langchain_core.tools import tool

from app.rag.vector_store import VectorStoreManager
from app.utils.logger import setup_logger


class RetrievalTool:
    """Wrapper for VectorStoreManager for agent use."""

    def __init__(self, vector_store: VectorStoreManager):
        """
        Initialize the retrieval tool.

        Args:
            vector_store (VectorStoreManager): The vector store manager instance.
        """
        self.vector_store = vector_store

    def retrieve(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store.

        Args:
            query (str): The search query.
            num_results (int, optional): Maximum number of documents to
                retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: Retrieved documents with content,
                source, and relevance_score.
        """
        try:
            results = self.vector_store.similarity_search(query, k=num_results)
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": doc.metadata.get("score", 0.0),
                }
                for doc in results
            ]
        except Exception as e:
            logger.warning("Error during retrieval: %s", e)
            return []


logger = setup_logger(__name__, "INFO")


def create_retrieval_tool(vector_store: VectorStoreManager) -> Any:
    """
    Factory to create a CrewAI compatible tool with injected vector store.
    """

    @tool
    def retrieve_documents(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the internal knowledge base.

        Args:
            query (str): The search query.
            num_results (int): Maximum number of documents to retrieve.
        """
        tool_instance = RetrievalTool(vector_store)
        return tool_instance.retrieve(query, num_results)

    return retrieve_documents
