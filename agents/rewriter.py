"""Query Rewriter Agent with HyDE (Hypothetical Document Embeddings).

Rewrites failing queries using two strategies:
1. Direct query reformulation for clarity
2. HyDE — generates a hypothetical answer to use as the search query,
   since hypothetical answers are semantically closer to real answers
   than questions are."""

from langchain_core.prompts import ChatPromptTemplate
from config import llm_fast


# ── Direct Query Rewrite Prompt ──────────────────────────────────────────────
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert query optimizer. Your task is to rewrite a search query 
to improve document retrieval results.

Strategies:
- Make implicit concepts explicit
- Add relevant synonyms or related terms
- Expand abbreviations
- Rephrase as a more specific, answerable question
- Keep the core intent unchanged

Respond with ONLY the rewritten query. No explanation."""
    ),
    (
        "human",
        "Original query: {question}\n\nRewritten query:"
    ),
])

rewrite_chain = REWRITE_PROMPT | llm_fast


# ── HyDE Prompt ──────────────────────────────────────────────────────────────
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a knowledgeable expert. Given a question, write a short, factual 
paragraph (3-5 sentences) that would serve as an ideal answer to the question.

Write confidently as if from a textbook. Do not say "I don't know" or hedge.
The paragraph will be used for semantic search, so be specific and detailed."""
    ),
    (
        "human",
        "Question: {question}\n\nIdeal answer paragraph:"
    ),
])

hyde_chain = HYDE_PROMPT | llm_fast


def rewrite_query(question: str) -> str:
    """Rewrite a query for better retrieval results."""
    try:
        result = rewrite_chain.invoke({"question": question})
        rewritten = result.content.strip()
        return rewritten if rewritten else question
    except Exception as e:
        print(f"⚠️ Query rewrite failed: {e}")
        return question


def generate_hyde_query(question: str) -> str:
    """Generate a hypothetical document for HyDE-enhanced retrieval."""
    try:
        result = hyde_chain.invoke({"question": question})
        hypothetical = result.content.strip()
        return hypothetical if hypothetical else question
    except Exception as e:
        print(f"⚠️ HyDE generation failed: {e}")
        return question
