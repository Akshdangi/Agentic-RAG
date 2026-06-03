"""BM25 Sparse Keyword Retriever for lexical matching."""

from langchain_community.retrievers import BM25Retriever
from config import RETRIEVER_K


def get_sparse_retriever(documents):
    """Create a BM25 retriever from document splits.
    
    Args:
        documents: List of LangChain Document objects to index.
    
    Returns:
        BM25Retriever configured with top-k results.
    """
    if not documents:
        raise ValueError("BM25 requires documents to build an index.")
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RETRIEVER_K
    
    print(f"🔤 Created BM25 index with {len(documents)} documents (k={RETRIEVER_K})")
    return bm25_retriever
