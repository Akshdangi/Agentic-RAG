"""Answer Generator Agent.

Uses the strong LLM (Llama 3.3 70B) to generate comprehensive answers
grounded in the retrieved context documents."""

from langchain_core.prompts import ChatPromptTemplate
from config import llm_strong


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise, Enterprise-grade AI assistant. Answer the user's question 
based STRICTLY on the provided context documents.

CRITICAL INSTRUCTIONS:
1. First, think step-by-step about how to answer the question using the context. 
2. Write your thoughts inside <thinking>...</thinking> XML tags.
3. Keep your <thinking> block extremely brief (max 2-3 sentences). Identify which documents to use and formulate a quick logical flow.
4. After the <thinking> block, provide your final comprehensive, well-structured answer.
5. Use ONLY information from the provided context to answer. If the context lacks the answer, clearly state so.
6. Do NOT fabricate information. 
7. AT THE VERY END of your answer, you MUST provide a "### Sources" section containing a bulleted list of the exact [Source: URL] links provided in the context that you used."""
    ),
    (
        "human",
        """Context Documents:
---
{context}
---

Question: {question}

Answer:"""
    ),
])

generator_chain = GENERATE_PROMPT | llm_strong


def generate_answer(question: str, documents: list) -> str:
    """Generate a grounded answer from retrieved documents.
    
    Args:
        question: The user's question (possibly rewritten).
        documents: List of LangChain Document objects.
    
    Returns:
        Generated answer string.
    """
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in documents
    )
    
    if not context.strip():
        return (
            "I apologize, but I couldn't find any relevant information to answer "
            "your question. Please try rephrasing your query or asking about a "
            "different topic."
        )
    
    try:
        result = generator_chain.invoke({
            "context": context,
            "question": question,
        })
        return result.content.strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}. Please try again."
