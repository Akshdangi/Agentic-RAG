"""Dense Vector Retriever using BGE-large-en-v1.5 embeddings with ChromaDB."""

from langchain_chroma import Chroma
from config import embeddings, CHROMA_PERSIST_DIR, CHROMA_COLLECTION, RETRIEVER_K


def get_vectorstore(documents=None):
    """Get or create the ChromaDB vector store.
    
    If documents are provided, creates a new collection from them.
    Otherwise, loads the existing persisted collection.
    """
    if documents:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION,
        )
        print(f"📦 Created vector store with {len(documents)} documents")
        return vectorstore
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )
    count = vectorstore._collection.count()
    print(f"📦 Loaded existing vector store ({count} documents)")
    return vectorstore


def get_dense_retriever(documents=None):
    """Create a dense retriever backed by ChromaDB."""
    vectorstore = get_vectorstore(documents)
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
