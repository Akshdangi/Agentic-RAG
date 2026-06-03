"""Document Ingestion Pipeline.

Loads, splits, and indexes documents into the hybrid retrieval system
(ChromaDB dense vectors + BM25 sparse index).

Usage:
    python ingest.py --sample    # Load sample AI/ML articles
    python ingest.py --dir ./docs  # Load from a directory
"""

import os
import sys
import argparse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Sample Data ──────────────────────────────────────────────────────────────
SAMPLE_DOCUMENTS = [
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": """Retrieval-Augmented Generation (RAG) is a technique that enhances large language model (LLM) outputs by incorporating external knowledge retrieval. Instead of relying solely on the model's training data, RAG systems first retrieve relevant documents from a knowledge base, then use those documents as context for generating responses.

The RAG pipeline consists of several key stages: document ingestion, where source materials are chunked and embedded into vector representations; retrieval, where user queries are matched against these embeddings to find relevant documents; and generation, where the LLM produces answers grounded in the retrieved context.

RAG addresses several fundamental limitations of LLMs: knowledge cutoff dates (the model only knows information up to its training date), hallucination (generating plausible but incorrect information), and lack of domain-specific knowledge. By grounding responses in retrieved documents, RAG systems can provide more accurate, up-to-date, and verifiable answers.

Advanced RAG techniques include hybrid retrieval (combining dense vector search with sparse keyword matching), re-ranking (using cross-encoders to refine retrieval results), and corrective RAG (CRAG), which adds self-correction mechanisms to detect and fix retrieval failures."""
    },
    {
        "title": "Vector Embeddings and Semantic Search",
        "content": """Vector embeddings are dense numerical representations of text that capture semantic meaning in high-dimensional space. Unlike traditional keyword-based search, vector embeddings enable semantic search — finding documents that are conceptually similar to a query, even when they don't share exact keywords.

Popular embedding models include OpenAI's text-embedding-3, Google's Gecko, and open-source alternatives like BGE (BAAI General Embedding), E5, and GTE. These models transform text into fixed-dimensional vectors (typically 768 or 1024 dimensions) that can be compared using cosine similarity or dot product.

Vector databases such as Pinecone, Qdrant, Weaviate, Milvus, and ChromaDB are purpose-built for storing and querying these embeddings efficiently. They use approximate nearest neighbor (ANN) algorithms like HNSW (Hierarchical Navigable Small World) to enable fast similarity search even with millions of vectors.

The quality of embeddings directly impacts retrieval accuracy. Techniques like query instruction prefixing, where the embedding model receives different instructions for queries vs documents, can significantly improve retrieval performance. Models like BGE-large-en-v1.5 recommend prefixing queries with 'Represent this sentence for searching relevant passages' to align the query embedding space with document embeddings."""
    },
    {
        "title": "LangGraph and Multi-Agent Systems",
        "content": """LangGraph is a framework for building stateful, multi-actor applications with Large Language Models. Built on top of LangChain, it provides a graph-based abstraction for orchestrating complex LLM workflows that go beyond simple sequential chains.

Unlike traditional LangChain chains (which are directed acyclic graphs), LangGraph supports cyclic graphs — enabling iterative processes where agents can loop, retry, and self-correct. This makes it ideal for building autonomous systems that need to evaluate their own outputs and take corrective action.

Key LangGraph concepts include: StateGraph (the graph definition with typed state), Nodes (Python functions that process and update state), Edges (connections between nodes, including conditional edges that enable branching logic), and Checkpointing (built-in state persistence for long-running workflows).

Multi-agent architectures in LangGraph typically follow patterns like: supervisor (one agent delegates to specialist agents), hierarchical (nested teams of agents), and collaborative (agents share a common state and communicate through it). The state graph pattern is particularly powerful for Corrective RAG (CRAG) systems, where different agents handle retrieval, grading, generation, and hallucination detection in a cyclic self-correcting loop."""
    },
    {
        "title": "Hallucination Detection in LLMs",
        "content": """Hallucination in large language models refers to the generation of content that appears plausible but is factually incorrect, unsupported by the input context, or entirely fabricated. This is one of the most critical challenges in deploying LLMs for knowledge-intensive tasks.

Types of hallucination include: intrinsic hallucination (contradicting the source material), extrinsic hallucination (generating information not present in the source), and factual hallucination (stating incorrect facts with confidence).

Detection methods include: NLI-based approaches (using natural language inference models to check if the generation is entailed by the source), token-level uncertainty estimation (examining model confidence for each generated token), self-consistency checking (generating multiple responses and comparing them), and LLM-as-judge (using another LLM to evaluate the faithfulness of the generation).

In RAG systems, hallucination guardrails typically implement a two-stage check: first verifying that the generated answer is grounded in the retrieved documents (groundedness check), and then verifying that the answer actually addresses the user's question (relevance check). If either check fails, the system can trigger corrective actions like re-generation or query rewriting."""
    },
    {
        "title": "BM25 and Hybrid Search",
        "content": """BM25 (Best Matching 25) is a probabilistic ranking function used in information retrieval that extends the classic TF-IDF approach. It considers term frequency (how often a term appears in a document), inverse document frequency (how rare a term is across the corpus), and document length normalization.

The BM25 formula includes two tuning parameters: k1 (controls term frequency saturation, typically 1.2-2.0) and b (controls document length normalization, typically 0.75). These parameters can be tuned for specific domains to optimize retrieval quality.

While dense vector embeddings excel at capturing semantic similarity, BM25 is superior at exact keyword matching — particularly important for technical terms, product names, error codes, and domain-specific jargon that embedding models may not handle well.

Hybrid search combines both approaches: dense retrieval for semantic understanding and BM25 for lexical precision. The results are typically merged using reciprocal rank fusion (RRF), which assigns each result a score based on its rank in each individual result list and combines them. This approach consistently outperforms either method alone in benchmarks like BEIR and MTEB.

Implementation frameworks like LangChain provide EnsembleRetriever, which handles the fusion automatically with configurable weights between sparse and dense retrievers."""
    },
    {
        "title": "Corrective RAG (CRAG)",
        "content": """Corrective Retrieval-Augmented Generation (CRAG) is an advanced RAG paradigm that introduces self-correction mechanisms to address the fundamental problem of noisy or irrelevant retrieval. Traditional RAG systems blindly pass retrieved documents to the LLM, which can lead to poor answers if the retrieval quality is low.

CRAG introduces three key innovations: a lightweight retrieval evaluator that assesses the relevance of retrieved documents, a corrective strategy that triggers different actions based on evaluation results, and web search augmentation as a fallback when internal knowledge is insufficient.

The CRAG workflow operates as follows: after initial retrieval, each document is evaluated for relevance. If the evaluation confidence is high (documents are clearly relevant), the system proceeds with normal RAG generation. If confidence is low (documents are ambiguous), the system augments with web search results. If confidence is very low (documents are irrelevant), the system completely replaces the retrieved context with web search results.

Key design principles in CRAG include: decompose-then-recompose (breaking documents into fine-grained knowledge strips for more precise evaluation), confidence-based routing (using evaluation scores to determine the correction strategy), and knowledge refinement (filtering out irrelevant information before generation).

Advanced CRAG implementations add additional self-correction layers including hallucination detection after generation, query rewriting using techniques like HyDE (Hypothetical Document Embeddings), and iterative retrieval where the system cycles through retrieve-evaluate-rewrite loops until satisfactory context is found."""
    },
]


def load_wikipedia_documents() -> list:
    """Download enterprise-scale AI/ML dataset from Wikipedia."""
    from langchain_community.document_loaders import WikipediaLoader
    
    topics = [
        "Large language model",
        "Retrieval-augmented generation",
        "Prompt engineering",
        "Vector database",
    ]
    
    documents = []
    print("\n📚 Downloading Wikipedia corpus...")
    
    import wikipedia
    # Wikipedia API requires a custom User-Agent, otherwise it returns JSONDecodeError
    wikipedia.set_user_agent("CRAG-Agentic-RAG-Bot/1.0 (https://github.com/langchain-ai)")
    
    for topic in topics:
        try:
            print(f"   Downloading: {topic}")
            loader = WikipediaLoader(query=topic, load_max_docs=1, doc_content_chars_max=10000)
            docs = loader.load()
            for doc in docs:
                source_url = doc.metadata.get("source", "")
                if not source_url.startswith("http"):
                    # Fallback if WikipediaLoader didn't provide a full URL
                    title = doc.metadata.get('title', topic).replace(" ", "_")
                    doc.metadata["source"] = f"https://en.wikipedia.org/wiki/{title}"
            documents.extend(docs)
        except Exception as e:
            print(f"   ⚠️ Failed to load '{topic}': {e}")
            
    return documents


def load_documents_from_directory(directory: str) -> list:
    """Load documents from a directory (supports .txt, .md, .pdf files)."""
    documents = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            if filename.endswith((".txt", ".md")):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": filepath, "title": filename},
                    )
                )
            elif filename.endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(filepath)
                    content = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": filepath, "title": filename},
                        )
                    )
                except ImportError:
                    print(f"⚠️ pypdf not installed. Skipping {filename}")
    
    return documents


def split_documents(documents: list) -> list:
    """Split documents into chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    splits = splitter.split_documents(documents)
    print(f"📄 Split {len(documents)} documents into {len(splits)} chunks")
    return splits


def ingest(documents: list = None, directory: str = None, use_sample: bool = False):
    """Run the full ingestion pipeline."""
    # 1. Load documents
    if use_sample:
        print("\n📚 Loading Enterprise Wikipedia corpus...")
        raw_docs = load_wikipedia_documents()
        if not raw_docs:
            print("⚠️ Wikipedia ingestion failed. Falling back to sample documents.")
            from langchain_core.documents import Document
            raw_docs = [
                Document(page_content=item["content"], metadata={"source": f"sample/{item['title']}", "title": item["title"]})
                for item in SAMPLE_DOCUMENTS
            ]
    elif directory:
        print(f"\n📚 Loading documents from {directory}...")
        raw_docs = load_documents_from_directory(directory)
    elif documents:
        raw_docs = documents
    else:
        raise ValueError("Provide documents, a directory, or use --sample")
    
    if not raw_docs:
        raise ValueError("No documents found to ingest.")
    
    print(f"   Loaded {len(raw_docs)} documents")
    
    # 2. Split into chunks
    splits = split_documents(raw_docs)
    
    # 3. Build hybrid retriever (creates ChromaDB + BM25)
    from retriever.hybrid import build_hybrid_retriever
    build_hybrid_retriever(splits)
    
    print(f"\n✅ Ingestion complete! {len(splits)} chunks indexed.")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAG Document Ingestion")
    parser.add_argument("--sample", action="store_true", help="Load sample AI/ML articles")
    parser.add_argument("--dir", type=str, help="Directory of documents to ingest")
    args = parser.parse_args()
    
    if args.sample:
        ingest(use_sample=True)
    elif args.dir:
        ingest(directory=args.dir)
    else:
        print("Usage: python ingest.py --sample  OR  python ingest.py --dir ./docs")
        sys.exit(1)
