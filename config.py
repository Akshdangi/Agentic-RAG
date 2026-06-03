"""
Centralized Configuration for the CRAG Agentic RAG System.

Loads environment variables, initializes LLM instances (fast + strong),
configures embedding models, and sets retrieval/agent parameters.
"""

import os
import platform
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ── Load Environment ─────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY not found. "
        "Sign up at https://console.groq.com and add it to your .env file."
    )

if not TAVILY_API_KEY:
    print(
        "⚠️  TAVILY_API_KEY not found. Web search fallback will be disabled. "
        "Sign up at https://tavily.com for 1,000 free API calls/month."
    )

# ── LLM Configuration ────────────────────────────────────────────────────────
# Fast model: Used for grading, routing, and query rewriting (high rate limit)
llm_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,
    max_retries=3,
)

# Strong model: Used for final answer generation (best quality)
llm_strong = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=GROQ_API_KEY,
    max_retries=3,
)

# ── Embedding Configuration ──────────────────────────────────────────────────
# Auto-detect Apple Silicon for MPS acceleration
def _detect_device() -> str:
    """Detect the best available compute device."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DEVICE = _detect_device()

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Retrieval Parameters ─────────────────────────────────────────────────────
RETRIEVER_K = 10                     # Number of documents to fetch initially
RERANKER_K = 5                       # Number of documents to keep after re-ranking
BM25_WEIGHT = 0.4                    # Sparse retrieval weight
DENSE_WEIGHT = 0.6                   # Dense retrieval weight
CHUNK_SIZE = 1500                    # Document chunk size (characters)
CHUNK_OVERLAP = 200                  # Chunk overlap (characters)
CHROMA_PERSIST_DIR = "./chroma_db"   # ChromaDB persistence directory
CHROMA_COLLECTION = "crag_docs"      # ChromaDB collection name
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, effective re-ranker

# ── Agent Parameters ─────────────────────────────────────────────────────────
MAX_RETRIES = 3                      # Maximum query rewrite cycles
TAVILY_MAX_RESULTS = 3               # Max web search results per query
RELEVANCE_THRESHOLD = 0.3            # Cosine similarity pre-filter threshold

# ── Display ──────────────────────────────────────────────────────────────────
print(f"🔧 Config loaded:")
print(f"   • LLM Fast:     llama-3.1-8b-instant")
print(f"   • LLM Strong:   llama-3.3-70b-versatile")
print(f"   • Embeddings:   {EMBEDDING_MODEL} ({EMBEDDING_DEVICE})")
print(f"   • Retriever K:  {RETRIEVER_K}")
print(f"   • Max Retries:  {MAX_RETRIES}")
print(f"   • Web Search:   {'✅ Enabled' if TAVILY_API_KEY else '❌ Disabled'}")
