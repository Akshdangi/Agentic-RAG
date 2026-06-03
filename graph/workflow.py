"""CRAG State Graph Assembly.

Builds and compiles the complete LangGraph workflow connecting
all agents, retrievers, and tools into a cyclic state machine."""

from langgraph.graph import StateGraph, START, END
from state import GraphState
from graph.nodes import (
    retrieve_node,
    grade_documents_node,
    generate_node,
    web_search_node,
    rewrite_query_node,
)
from graph.edges import (
    decide_to_generate,
    grade_generation,
    check_max_retries,
)


def build_workflow():
    """Build and compile the CRAG multi-agent state graph.
    
    Returns:
        Compiled LangGraph application ready for invocation.
    
    Graph Structure:
        START → retrieve → grade_documents → [decide_to_generate]
            ├── 'generate'    → generate → [grade_generation]
            │                      ├── 'useful'        → END
            │                      ├── 'not_useful'    → rewrite_query
            │                      └── 'hallucination' → generate (retry)
            ├── 'web_search'  → web_search → generate
            └── 'rewrite'     → rewrite_query → [check_max_retries]
                                    ├── 'continue'   → retrieve (cycle)
                                    └── 'max_retries' → generate (force)
    """
    workflow = StateGraph(GraphState)
    
    # ── Add Nodes ─────────────────────────────────────────────────────────
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    
    # ── Add Edges ─────────────────────────────────────────────────────────
    # Entry point
    workflow.add_edge(START, "retrieve")
    
    # Retrieve → Grade
    workflow.add_edge("retrieve", "grade_documents")
    
    # Grade → Conditional routing
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "web_search": "web_search",
            "rewrite": "rewrite_query",
        },
    )
    
    # Web search → Generate
    workflow.add_edge("web_search", "generate")
    
    # Generate → Hallucination guardrail
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "useful": END,
            "not_useful": "rewrite_query",
            "hallucination": "generate",
        },
    )
    
    # Rewrite → Check retries
    workflow.add_conditional_edges(
        "rewrite_query",
        check_max_retries,
        {
            "continue": "retrieve",
            "max_retries": "generate",
        },
    )
    
    # ── Compile ───────────────────────────────────────────────────────────
    app = workflow.compile()
    
    print("\n🔗 CRAG State Graph compiled successfully!")
    print("   Nodes: retrieve → grade → generate/web_search/rewrite")
    print("   Cycles: rewrite → retrieve (max 3 retries)")
    print("   Guards: hallucination + usefulness checks")
    
    return app


def stream_query(app, question: str):
    """Run a query through the CRAG pipeline and stream intermediate states.
    
    Args:
        app: Compiled LangGraph workflow.
        question: User's question.
    
    Yields:
        Intermediate states containing agent_log updates, and finally the full state dict.
    """
    initial_state = {
        "question": question,
        "original_question": question,
        "generation": "",
        "documents": [],
        "web_search": "No",
        "retry_count": 0,
        "agent_log": [f"🧑 User query: '{question}'"],
    }
    
    final_state = None
    # Stream mode "values" yields the full state after each node executes
    for state in app.stream(initial_state, stream_mode="values", config={"recursion_limit": 25}):
        final_state = state
        if "agent_log" in state:
            yield {"log": state["agent_log"]}
            
    if final_state:
        yield final_state

