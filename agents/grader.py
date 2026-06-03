"""Document Grader Agent.

Evaluates the semantic relevance of each retrieved document against
the user query. Uses structured LLM output for reliable binary scoring."""

from langchain_core.prompts import ChatPromptTemplate
from config import llm_fast


# Bulk Grading prompt — instructs the LLM to output a comma-separated list of relevant indices
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert document relevance grader. Your task is to assess whether 
each retrieved document contains information relevant to answering a user question.

Rules:
- If a document contains keywords, concepts, or information related to the question, grade it as relevant.
- The assessment should not be overly strict — partial relevance counts.
- You will receive multiple documents numbered [1], [2], etc.
- Respond with ONLY a comma-separated list of the numbers of the relevant documents. 
- Example response: 1, 3, 5
- If NO documents are relevant, respond with: none"""
    ),
    (
        "human",
        """User Question: {question}

Retrieved Documents:
---
{documents}
---

List the numbers of the relevant documents:"""
    ),
])

grader_chain = GRADE_PROMPT | llm_fast


def grade_documents_bulk(documents: list, question: str) -> list[int]:
    """Grade a batch of documents for relevance in a single API call.
    
    Args:
        documents: List of LangChain Document objects.
        question: The user's question.
    
    Returns:
        List of indices (0-based) of the relevant documents.
    """
    if not documents:
        return []
        
    context = "\n\n".join(
        f"[{i+1}]\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )
    
    try:
        result = grader_chain.invoke({
            "documents": context,
            "question": question,
        })
        score_str = result.content.strip().lower()
        if "none" in score_str:
            return []
            
        # Parse the comma-separated string (e.g., "1, 3, 5" -> [0, 2, 4])
        relevant_indices = []
        import re
        numbers = re.findall(r'\d+', score_str)
        for num in numbers:
            idx = int(num) - 1
            if 0 <= idx < len(documents):
                relevant_indices.append(idx)
                
        return relevant_indices
    except Exception as e:
        print(f"⚠️ Bulk grading error: {e}")
        # Fail-open: include all documents if grading fails
        return list(range(len(documents)))
