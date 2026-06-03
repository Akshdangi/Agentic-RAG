"""Hallucination Guardrail Agent.

Two-stage verification of generated answers:
1. Groundedness Check — Are claims supported by source documents?
2. Usefulness Check — Does the answer actually address the question?

Blocks hallucinated or off-topic responses before they reach the user."""

from langchain_core.prompts import ChatPromptTemplate
from config import llm_fast


# ── Stage 1: Groundedness Check ──────────────────────────────────────────────
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict factual groundedness evaluator. Your task is to determine 
whether an AI-generated answer is supported by the provided source documents.

Rules:
- Check if each claim in the answer can be traced back to the source documents.
- Minor paraphrasing is acceptable if the meaning is preserved.
- If the answer includes ANY claims not supported by the documents, grade as 'no'.
- Respond with ONLY a single word: 'yes' or 'no'. Nothing else."""
    ),
    (
        "human",
        """Source Documents:
---
{documents}
---

Generated Answer:
---
{generation}
---

Is the answer grounded in the source documents? (yes/no):"""
    ),
])

hallucination_chain = HALLUCINATION_PROMPT | llm_fast


# ── Stage 2: Usefulness Check ────────────────────────────────────────────────
USEFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an answer quality evaluator. Your task is to determine whether 
an AI-generated answer actually addresses and resolves the user's question.

Rules:
- The answer must directly address what was asked.
- Partial answers that provide useful information count as 'yes'.
- Completely off-topic or evasive answers are 'no'.
- Respond with ONLY a single word: 'yes' or 'no'. Nothing else."""
    ),
    (
        "human",
        """User Question: {question}

Generated Answer:
---
{generation}
---

Does this answer address the user's question? (yes/no):"""
    ),
])

usefulness_chain = USEFULNESS_PROMPT | llm_fast


def check_hallucination(generation: str, documents: list) -> bool:
    """Check if the generation is grounded in source documents.
    
    Returns:
        True if grounded (no hallucination), False if hallucinated.
    """
    doc_texts = "\n\n".join(
        f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(documents)
    )
    
    try:
        result = hallucination_chain.invoke({
            "documents": doc_texts,
            "generation": generation,
        })
        score = result.content.strip().lower()
        return "yes" in score
    except Exception as e:
        print(f"⚠️ Hallucination check error: {e}")
        return True  # Fail-open


def check_usefulness(generation: str, question: str) -> bool:
    """Check if the generation actually answers the question.
    
    Returns:
        True if the answer is useful, False otherwise.
    """
    try:
        result = usefulness_chain.invoke({
            "question": question,
            "generation": generation,
        })
        score = result.content.strip().lower()
        return "yes" in score
    except Exception as e:
        print(f"⚠️ Usefulness check error: {e}")
        return True  # Fail-open


def evaluate_generation(generation: str, documents: list, question: str) -> str:
    """Run the full hallucination guardrail pipeline.
    
    Returns:
        'useful' — Answer is grounded and addresses the question.
        'hallucination' — Answer contains unsupported claims.
        'not_useful' — Answer doesn't address the question.
    """
    # Stage 1: Groundedness
    is_grounded = check_hallucination(generation, documents)
    if not is_grounded:
        return "hallucination"
    
    # Stage 2: Usefulness
    is_useful = check_usefulness(generation, question)
    if not is_useful:
        return "not_useful"
    
    return "useful"
