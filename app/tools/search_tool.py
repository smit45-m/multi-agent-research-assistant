"""
Web search tool for agents using DuckDuckGo.
"""
from typing import List, Dict, Any
from duckduckgo_search import DDGS
from langchain_core.tools import tool

class WebSearchTool:
    """Wrapper for DuckDuckGo search."""
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a web search using DuckDuckGo.
        
        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to return. Defaults to 5.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing title, url, and snippet.
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    }
                    for r in results
                ]
        except Exception as e:
            # Handle rate limits or other errors gracefully
            print(f"Error during web search: {e}")
            return []

@tool
def web_search_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a web search to find relevant information.
    
    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
    """
    searcher = WebSearchTool()
    return searcher.search(query, max_results)
