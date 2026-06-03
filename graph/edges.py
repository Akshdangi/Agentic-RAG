"""Conditional Edge Functions for the CRAG State Graph.

These functions determine the flow between nodes based on
the current state, enabling the cyclic self-correction behavior."""

from state import GraphState
from agents.hallucination import evaluate_generation
from agents.router import route_after_grading
from config import MAX_RETRIES
import time


def decide_to_generate(state: GraphState) -> str:
    """Route after document grading: generate, web_search, or rewrite.
    
    This is the core CRAG routing decision.
    """
    documents = state.get("documents", [])
    web_search = state.get("web_search", "No")
    retry_count = state.get("retry_count", 0)
    
    # If router already decided web search
    if web_search == "Yes":
        return "web_search"
    
    # If no documents survived grading
    if not documents:
        if retry_count >= MAX_RETRIES:
            return "web_search"  # Fall back to web as last resort
        return "rewrite"
    
    return "generate"


def grade_generation(state: GraphState) -> str:
    """Evaluate generation quality: useful, hallucination, or not_useful.
    
    Runs the two-stage hallucination guardrail.
    """
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    question = state.get("original_question", state.get("question", ""))
    retry_count = state.get("retry_count", 0)
    
    print(f"\n🛡️ HALLUCINATION GUARDRAIL...")
    time.sleep(0.5)  # Rate limit buffer
    
    # Skip guardrail if we've exhausted retries
    if retry_count >= MAX_RETRIES:
        print("   ⚠️ Max retries reached — accepting generation")
        return "useful"
    
    result = evaluate_generation(generation, documents, question)
    
    emoji_map = {"useful": "✅", "hallucination": "🚫", "not_useful": "↩️"}
    emoji = emoji_map.get(result, "❓")
    print(f"   {emoji} Guardrail verdict: {result.upper()}")
    
    return result


def check_max_retries(state: GraphState) -> str:
    """Check if max rewrite retries have been exceeded."""
    retry_count = state.get("retry_count", 0)
    
    if retry_count >= MAX_RETRIES:
        print(f"   ⚠️ Max retries ({MAX_RETRIES}) reached — forcing generation")
        return "max_retries"
    
    return "continue"
