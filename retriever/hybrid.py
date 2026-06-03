"""Hybrid Retriever combining Dense (BGE) + Sparse (BM25) search.

Uses LangChain's EnsembleRetriever with reciprocal rank fusion to
merge results from both retrieval strategies."""

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from retriever.dense import get_dense_retriever, get_vectorstore
from retriever.sparse import get_sparse_retriever
from config import BM25_WEIGHT, DENSE_WEIGHT


# Module-level cache for the hybrid retriever
_hybrid_retriever = None
_vectorstore = None


def build_hybrid_retriever(documents=None):
    """Build the hybrid retriever combining dense and sparse search.
    
    Args:
        documents: Document splits to index. Required on first call.
    
    Returns:
        EnsembleRetriever with reciprocal rank fusion.
    """
    global _hybrid_retriever, _vectorstore
    
    dense_retriever = get_dense_retriever(documents)
    _vectorstore = get_vectorstore()  # For HyDE similarity_search_by_vector
    
    if documents:
        sparse_retriever = get_sparse_retriever(documents)
        
        _hybrid_retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=[BM25_WEIGHT, DENSE_WEIGHT],
        )
        
        print(f"🔀 Hybrid retriever ready (BM25={BM25_WEIGHT}, Dense={DENSE_WEIGHT})")
    else:
        # No documents = use dense only (BM25 needs in-memory docs)
        _hybrid_retriever = dense_retriever
        print("🔀 Dense-only retriever ready (no documents for BM25)")
    
    return _hybrid_retriever


def get_hybrid_retriever():
    """Get the cached hybrid retriever instance."""
    if _hybrid_retriever is None:
        return build_hybrid_retriever()
    return _hybrid_retriever


def get_vectorstore_instance():
    """Get the cached vectorstore for direct similarity search (used by HyDE)."""
    if _vectorstore is None:
        return get_vectorstore()
    return _vectorstore
