<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Git-logo.svg/512px-Git-logo.svg.png" width="100" />
  <h1>🧠 CRAG Agentic RAG</h1>
  <p><b>Self-Correcting Multi-Agent Retrieval-Augmented Generation Platform</b></p>
</div>

<br/>

## 🌌 Overview

**CRAG Agentic RAG** is a state-of-the-art, multi-agent AI system built to solve the hallucination and low-relevancy problems of traditional RAG pipelines. Powered by **LangGraph**, **Groq (Llama-3.3-70B)**, and **ChromaDB**, this platform acts as an intelligent reasoning engine that grades, corrects, and augments its own research before generating an answer.

We've wrapped this powerful backend in a gorgeous, cyberpunk-themed **Gradio** interface complete with **Voice-to-Text capabilities**, a live **Agent Brain Terminal** to trace reasoning steps, and visual **Confidence Score** progress bars.

## 🚀 Key Features

*   🤖 **Agentic Self-Correction**: Implements the **CRAG (Corrective RAG)** architecture. If retrieved documents are irrelevant, the Grader Agent automatically discards them and routes the query to a Web Search agent or rewrites the query for a secondary retrieval pass.
*   🎙️ **Voice-to-Text Input**: Integrated **Groq Whisper-large-v3** for seamless audio-to-text querying directly in the browser.
*   🔍 **Hybrid Search Pipeline**: Fuses **BM25 (Sparse)** and **BGE-Large (Dense)** embeddings using Reciprocal Rank Fusion, followed by a **Cross-Encoder Re-ranker** for maximum document precision.
*   💻 **Hacker Terminal UI**: A sleek dark-mode interface featuring a live, streaming "Agent Brain Log" that allows you to watch the multi-agent pipeline think, route, and grade in real-time.
*   📊 **Relevancy Visualizer**: Outputs animated progress bars showing the exact relevancy confidence percentage of the source documents utilized for the answer.
*   🌐 **Dynamic Web Fallback**: Seamless integration with the **Tavily Search API** when local knowledge is insufficient.

## 🏗️ Architecture

1.  **Hybrid Retriever**: Fetches top candidates using Dense + Sparse vectors.
2.  **Cross-Encoder Re-ranker**: Perfectly reorders candidates based on query context.
3.  **Grader Agent**: Llama-3.1 instantly evaluates each document. Irrelevant docs are tossed.
4.  **Router Agent**: If not enough docs pass, triggers query-rewriting or Web Search.
5.  **Generator Agent**: Llama-3.3 synthesizes the final response with cited sources.
6.  **Hallucination Checker**: Final safety net to ensure the generation is grounded in the retrieved facts.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/crag-agentic-rag.git
   cd crag-agentic-rag
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

4. **Launch the Application:**
   ```bash
   python app.py
   ```
   *Navigate to `http://localhost:7860` in your browser.*

## 📸 Interface Showcase

*   **Neural Background Grid:** A smooth, CSS-animated matrix grid that gives the UI a living pulse.
*   **System Status Footer:** Live diagnostics showing current models, search status, and cross-encoder availability.
*   **Dynamic Example Rotation:** Suggested queries automatically cycle every 30 seconds to keep discovery fresh.

## 🤝 Contributing
Contributions are always welcome! Feel free to open a Pull Request or create an Issue to discuss new features or bugs.

---
*Built with ❤️ using LangGraph • Groq • ChromaDB • Gradio*
