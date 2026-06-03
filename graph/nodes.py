"""Graph Node Functions.

Each function is a node in the LangGraph state machine.
Nodes receive the current GraphState, perform their work,
and return a partial state update dict."""

import time
from state import GraphState
from agents.router import route_after_grading, format_routing_decision
from agents.rewriter import rewrite_query, generate_hyde_query
from agents.generator import generate_answer
from agents.hallucination import evaluate_generation
from tools.web_search import search_web
from retriever.hybrid import get_hybrid_retriever
from retriever.reranker import rerank_documents


def retrieve_node(state: GraphState) -> dict:
    """Retrieve documents using the hybrid retriever."""
    question = state["question"]
    print(f"\n📥 RETRIEVE: Searching for '{question[:80]}...'")
    
    retriever = get_hybrid_retriever()
    print(f"DEBUG retrieve_node: question type={type(question)}, question={repr(question)}")
    
    if isinstance(question, list):
        # Flatten it if it's a list for some reason
        question = question[0] if len(question) > 0 else ""
        if isinstance(question, dict) and "text" in question:
            question = question["text"]
        question = str(question)
        
    documents = retriever.invoke(question)
    
    print(f"   📥 Hybrid search returned {len(documents)} documents. Re-ranking...")
    documents = rerank_documents(question, documents)
    
    log_entry = f"📥 Retrieved and re-ranked top {len(documents)} documents"
    print(f"   {log_entry}")
    
    return {
        "documents": documents,
        "question": question,
        "agent_log": [log_entry],
    }


def grade_documents_node(state: GraphState) -> dict:
    """Grade each retrieved document for relevance."""
    question = state["question"]
    documents = state["documents"]
    
    from agents.grader import grade_documents_bulk
    
    print(f"\n🔍 GRADING: Evaluating {len(documents)} documents in bulk...")
    
    relevant_docs = []
    
    relevant_indices = grade_documents_bulk(documents, question)
    
    for i, doc in enumerate(documents):
        if i in relevant_indices:
            relevant_docs.append(doc)
            print(f"   ✅ Doc {i+1}: Relevant")
        else:
            print(f"   ❌ Doc {i+1}: Irrelevant")
    
    # Route decision
    retry_count = state.get("retry_count", 0)
    decision = route_after_grading(len(relevant_docs), len(documents), retry_count)
    routing_log = format_routing_decision(decision, len(relevant_docs), len(documents))
    
    grading_log = f"🔍 Graded: {len(relevant_docs)}/{len(documents)} relevant"
    print(f"   {grading_log}")
    print(f"   {routing_log}")
    
    return {
        "documents": relevant_docs,
        "question": question,
        "web_search": "Yes" if decision == "web_search" else "No",
        "agent_log": [grading_log, routing_log],
    }


def generate_node(state: GraphState) -> dict:
    """Generate an answer using the strong LLM."""
    question = state["question"]
    documents = state["documents"]
    
    print(f"\n🤖 GENERATING answer...")
    
    generation = generate_answer(question, documents)
    
    log_entry = f"🤖 Generated answer ({len(generation)} chars)"
    print(f"   {log_entry}")
    
    return {
        "generation": generation,
        "documents": documents,
        "question": question,
        "agent_log": [log_entry],
    }


def web_search_node(state: GraphState) -> dict:
    """Fetch additional context from web search."""
    question = state.get("original_question", state["question"])
    documents = state.get("documents", [])
    
    print(f"\n🌐 WEB SEARCH: '{question[:80]}...'")
    
    web_docs = search_web(question)
    
    # Merge web results with existing relevant documents
    all_documents = documents + web_docs
    
    log_entry = f"🌐 Web search returned {len(web_docs)} results (total: {len(all_documents)})"
    print(f"   {log_entry}")
    
    return {
        "documents": all_documents,
        "question": question,
        "web_search": "Yes",
        "agent_log": [log_entry],
    }


def rewrite_query_node(state: GraphState) -> dict:
    """Rewrite the query using HyDE for better retrieval."""
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    
    print(f"\n✏️ REWRITING query (attempt {retry_count + 1})...")
    
    # Use HyDE for the first rewrite, direct rewrite for subsequent
    if retry_count == 0:
        new_question = generate_hyde_query(question)
        strategy = "HyDE"
    else:
        new_question = rewrite_query(question)
        strategy = "Direct Rewrite"
    
    log_entry = f"✏️ Query rewritten ({strategy}): '{new_question[:80]}...'"
    print(f"   {log_entry}")
    
    return {
        "question": new_question,
        "retry_count": retry_count + 1,
        "agent_log": [log_entry],
    }
