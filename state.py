"""
Graph State Definition for the CRAG Multi-Agent System.

Defines the shared state schema that flows between all agents
in the LangGraph state machine. Every node reads from and writes
to this typed dictionary.
"""

from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
import operator


class GraphState(TypedDict):
    """
    Shared state for the CRAG multi-agent state graph.

    Attributes:
        question:          The current query (may be rewritten by the rewriter agent).
        original_question: The original user question, preserved for reference.
        generation:        The LLM-generated answer from the generator agent.
        documents:         Retrieved documents (from hybrid retriever or web search).
        web_search:        Flag indicating if web search was used ("Yes" / "No").
        retry_count:       Counter for query rewrite cycles (prevents infinite loops).
        agent_log:         Trace of agent decisions for UI display. Uses operator.add
                           as reducer so each node can append entries without overwriting.
    """
    question: str
    original_question: str
    generation: str
    documents: List[Document]
    web_search: str
    retry_count: int
    agent_log: Annotated[List[str], operator.add]
