"""Corrective Router Agent.

Decides the next action based on document grading results:
- All relevant → proceed to generation
- Some/all irrelevant → trigger web search or query rewrite
- Max retries exceeded → force generation with available context"""

from config import MAX_RETRIES


def route_after_grading(relevant_count: int, total_count: int, retry_count: int) -> str:
    """Determine the routing decision after document grading.
    
    Args:
        relevant_count: Number of documents graded as relevant.
        total_count: Total number of retrieved documents.
        retry_count: Current rewrite cycle count.
    
    Returns:
        One of: 'generate', 'web_search', 'rewrite'
    """
    # Safety valve: force generation if we've retried too many times
    if retry_count >= MAX_RETRIES:
        return "generate"
    
    relevance_ratio = relevant_count / max(total_count, 1)
    
    if relevance_ratio >= 0.6:
        # Majority of documents are relevant — generate answer
        return "generate"
    elif relevance_ratio > 0:
        # Some relevant, some not — supplement with web search
        return "web_search"
    else:
        # No relevant documents — query is fundamentally flawed, rewrite it
        return "rewrite"


def format_routing_decision(decision: str, relevant_count: int, total_count: int) -> str:
    """Format a human-readable routing decision for the agent trace log."""
    emoji_map = {
        "generate": "✅",
        "web_search": "🌐",
        "rewrite": "✏️",
    }
    emoji = emoji_map.get(decision, "❓")
    return (
        f"{emoji} Router: {relevant_count}/{total_count} docs relevant → "
        f"{decision.upper()}"
    )
