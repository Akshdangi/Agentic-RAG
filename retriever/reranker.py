"""Cross-Encoder Re-ranker.

Takes the initial top-K documents from the hybrid retriever and 
re-scores them using a Cross-Encoder for maximum precision."""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from config import CROSS_ENCODER_MODEL, RERANKER_K, EMBEDDING_DEVICE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Module-level cache for the cross-encoder
_reranker = None

def get_reranker():
    """Load and cache the cross-encoder model."""
    global _reranker
    if _reranker is None:
        print(f"🔄 Loading Cross-Encoder ({CROSS_ENCODER_MODEL}) on {EMBEDDING_DEVICE}...")
        _reranker = CrossEncoder(
            CROSS_ENCODER_MODEL, 
            device=EMBEDDING_DEVICE,
            max_length=512
        )
    return _reranker

def rerank_documents(query: str, documents: list[Document]) -> list[Document]:
    """Re-rank a list of documents based on relevance to the query.
    
    Args:
        query: The user query.
        documents: Initial list of retrieved documents.
        
    Returns:
        List of the top RERANKER_K documents, perfectly ordered.
    """
    if not documents:
        return []
        
    model = get_reranker()
    
    # Create pairs of (query, document_text)
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Score pairs
    scores = model.predict(pairs)
    
    # Add scores to metadata and sort
    scored_docs = []
    for score, doc in zip(scores, documents):
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append((score, doc))
        
    # Sort descending by score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top K
    top_k_docs = [doc for score, doc in scored_docs[:RERANKER_K]]
    
    return top_k_docs
