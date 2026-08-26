"""
Agent 2 - RAG Retriever.
"""
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.chains.prompts import RETRIEVER_SYSTEM_PROMPT
from app.agents.state import ResearchState
from app.rag.vector_store import VectorStoreManager
from app.tools.search_tool import WebSearchTool

class RetrieverAgent:
    """Agent responsible for retrieving relevant documents and web pages."""
    
    def __init__(self, vector_store: VectorStoreManager, search_tool: WebSearchTool, llm=None):
        """
        Initialize RetrieverAgent.
        
        Args:
            vector_store (VectorStoreManager): Vector store for document retrieval.
            search_tool (WebSearchTool): Tool for web searches.
            llm (Optional[ChatOpenAI]): Language model.
        """
        self.vector_store = vector_store
        self.search_tool = search_tool
        
        if llm is None:
            settings = get_settings()
            kwargs = {
                "model": settings.OPENAI_MODEL_NAME,
                "api_key": settings.OPENAI_API_KEY,
                "temperature": 0.1
            }
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            self.llm = ChatOpenAI(**kwargs)
        else:
            self.llm = llm

    def retrieve(self, state: ResearchState) -> ResearchState:
        """
        Searches vector store and web for each sub-question, deduplicates, and ranks results.
        
        Args:
            state (ResearchState): Current research state.
            
        Returns:
            ResearchState: Updated state with retrieved_documents.
        """
        sub_questions = state.get("sub_questions", [])
        if not sub_questions:
            # Fallback to main query if no sub-questions
            sub_questions = [state["query"]]
            
        all_retrieved = []
        seen_content = set()
        
        for sq in sub_questions:
            # 1. Vector Store Retrieval
            try:
                vs_results = self.vector_store.similarity_search(sq, k=3)
                for doc in vs_results:
                    content_hash = hash(doc.page_content[:200]) # simple deduplication
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_retrieved.append({
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", "internal_kb"),
                            "relevance_score": doc.metadata.get("score", 0.8), # Placeholder score if not provided
                            "source_type": "vector_store"
                        })
            except Exception as e:
                state["errors"].append(f"Vector store retrieval error for '{sq}': {str(e)}")

            # 2. Web Search
            try:
                web_results = self.search_tool.search(sq, max_results=2)
                for r in web_results:
                    content_hash = hash(r["snippet"][:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_retrieved.append({
                            "content": r["snippet"],
                            "source": r["url"],
                            "relevance_score": 0.7, # Default web score
                            "source_type": "web"
                        })
            except Exception as e:
                state["errors"].append(f"Web search error for '{sq}': {str(e)}")
                
        state["retrieved_documents"] = all_retrieved
        state["status"] = "retrieved"
        state["iteration_count"] += 1
        return state
