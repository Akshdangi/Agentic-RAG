"""Tavily Web Search Tool.

Provides web search fallback when the internal knowledge base
lacks relevant documents for a query."""

import os
from langchain_core.documents import Document
from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS


def search_web(query: str) -> list:
    """Search the web using Tavily API.
    
    Args:
        query: Search query string.
    
    Returns:
        List of LangChain Document objects with web search results.
    """
    if not TAVILY_API_KEY:
        print("⚠️ Tavily API key not set. Web search disabled.")
        return []
    
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        
        # Fetch a larger pool of results to ensure diversity
        search_tool = TavilySearchResults(
            max_results=TAVILY_MAX_RESULTS * 3,
        )
        
        results = search_tool.invoke({"query": query})
        
        from urllib.parse import urlparse
        seen_domains = set()
        documents = []
        
        for result in results:
            content = result.get("content", "")
            url = result.get("url", "unknown")
            
            # Extract domain to ensure different sources
            try:
                domain = urlparse(url).netloc
            except:
                domain = url
                
            if domain in seen_domains or domain == "unknown":
                continue
                
            seen_domains.add(domain)
            
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": url, "type": "web_search", "domain": domain},
                )
            )
            
            # Stop once we have enough diverse sources
            if len(documents) >= TAVILY_MAX_RESULTS:
                break
        
        return documents
    
    except Exception as e:
        print(f"⚠️ Web search error: {e}")
        return []
