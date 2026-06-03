"""CRAG Agentic RAG — Premium Gradio Interface.

A stunning dark-themed UI for the Self-Correcting Agentic RAG system
with real-time agent trace visualization."""

import gradio as gr
import time
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Global State ─────────────────────────────────────────────────────────────
workflow_app = None
is_initialized = False


def initialize_system():
    """Initialize the CRAG system (ingestion + graph compilation)."""
    global workflow_app, is_initialized
    
    if is_initialized:
        return "✅ System already initialized."
    
    try:
        # Ingest sample documents
        from ingest import ingest
        ingest(use_sample=True)
        
        # Build the graph
        from graph.workflow import build_workflow
        workflow_app = build_workflow()
        is_initialized = True
        
        return "✅ System initialized successfully! Sample documents ingested and graph compiled."
    except Exception as e:
        return f"❌ Initialization failed: {str(e)}"


def transcribe_audio(audio_path):
    """Transcribe audio using Groq Whisper."""
    if not audio_path:
        return ""
    try:
        from groq import Groq
        from config import GROQ_API_KEY
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
            )
            return transcription.text
    except Exception as e:
        print(f"Audio Transcription Error: {e}")
        return ""


# ── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global Theme ──────────────────────────────────────────────────────── */
.gradio-container {
    background: radial-gradient(circle at top, #0c0822 0%, #05030f 100%) !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 200vw; height: 200vh;
    background-image: 
        radial-gradient(circle at center, rgba(0,243,255,0.15) 2px, transparent 2.5px),
        linear-gradient(rgba(0,243,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,243,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: grid-move 40s linear infinite;
    z-index: -1;
    pointer-events: none;
}

@keyframes grid-move {
    0% { transform: translate(0, 0); }
    100% { transform: translate(-50px, -50px); }
}

/* ── Header ────────────────────────────────────────────────────────────── */
.header-section {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 1rem;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

.header-section h1 {
    font-size: 3rem !important;
    font-weight: 900 !important;
    color: #fff !important;
    text-shadow: 0 0 10px #00f3ff, 0 0 20px #00f3ff, 0 0 40px #ff00ea;
    margin-bottom: 0.5rem !important;
    letter-spacing: 2px;
}

.header-section p {
    color: #a0aec0 !important;
    font-size: 1.1rem !important;
    max-width: 700px;
    margin: 0 auto;
}

/* ── Chat Container ──────────────────────────────────────────────────── */
.chatbot {
    background: rgba(10, 8, 30, 0.7) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(0, 243, 255, 0.3) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 30px rgba(0, 243, 255, 0.1) !important;
    min-height: 500px !important;
}

/* ── Message Bubbles ─────────────────────────────────────────────────── */
.message.user {
    background: linear-gradient(135deg, #00f3ff 0%, #0088ff 100%) !important;
    border-radius: 20px 20px 4px 20px !important;
    color: #000 !important;
    font-weight: 500 !important;
    box-shadow: 0 5px 20px rgba(0, 243, 255, 0.4) !important;
    animation: slideUp 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.message.bot {
    background: rgba(20, 15, 45, 0.8) !important;
    border: 1px solid rgba(255, 0, 234, 0.3) !important;
    border-radius: 20px 20px 20px 4px !important;
    color: #e2e8f0 !important;
    box-shadow: 0 5px 20px rgba(255, 0, 234, 0.1) !important;
    animation: slideUp 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Input Area ──────────────────────────────────────────────────────── */
.input-row textarea {
    background: rgba(15, 10, 35, 0.8) !important;
    border: 2px solid rgba(0, 243, 255, 0.3) !important;
    border-radius: 16px !important;
    color: #fff !important;
    font-size: 1rem !important;
    padding: 14px 20px !important;
    transition: all 0.3s ease !important;
}

.input-row textarea:focus {
    border-color: #00f3ff !important;
    box-shadow: 0 0 25px rgba(0, 243, 255, 0.3) !important;
    outline: none !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(90deg, #ff00ea, #00f3ff) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 12px 24px !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 0 15px rgba(255, 0, 234, 0.4), 0 0 15px rgba(0, 243, 255, 0.4) !important;
    animation: pulse-border 2s infinite;
}

button.primary:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 0 30px rgba(255, 0, 234, 0.6), 0 0 30px rgba(0, 243, 255, 0.6) !important;
}

button.secondary {
    background: transparent !important;
    border: 1px solid rgba(0, 243, 255, 0.5) !important;
    border-radius: 14px !important;
    color: #00f3ff !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover {
    background: rgba(0, 243, 255, 0.1) !important;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.3) !important;
}

/* ── Terminal Box ──────────────────────────────────────────────────────── */
.terminal-box {
    background: #020205 !important;
    border: 1px solid #00ff41 !important;
    border-radius: 8px !important;
    color: #00ff41 !important;
    font-family: 'Courier New', monospace !important;
    padding: 15px !important;
    box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.2);
    min-height: 250px;
    max-height: 400px;
    overflow-y: auto;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}

.terminal-box p {
    margin-bottom: 5px !important;
}

.terminal-box strong {
    color: #fff;
}

/* ── Example Pills ───────────────────────────────────────────────────── */
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    margin-bottom: 15px;
}

.example-pill {
    background: rgba(255, 0, 234, 0.05) !important;
    border: 1px solid rgba(255, 0, 234, 0.4) !important;
    border-radius: 20px !important;
    color: #ff00ea !important;
    padding: 6px 14px !important;
    font-size: 0.85rem !important;
    cursor: pointer;
    transition: all 0.3s ease !important;
}

.example-pill:hover {
    background: rgba(255, 0, 234, 0.2) !important;
    box-shadow: 0 0 15px rgba(255, 0, 234, 0.4) !important;
    transform: translateY(-2px);
}

/* ── Status Panel ────────────────────────────────────────────────────── */
.status-panel {
    background: rgba(15, 10, 35, 0.7) !important;
    border: 1px solid rgba(0, 243, 255, 0.2) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5) !important;
}

/* ── Animated Explainer Flowchart ────────────────────────────────────── */
.explainer-container {
    background: rgba(5, 3, 15, 0.8);
    border: 1px solid rgba(0, 243, 255, 0.4);
    border-radius: 16px;
    padding: 20px;
    margin: 20px 0;
    text-align: center;
    box-shadow: inset 0 0 30px rgba(0, 243, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.explainer-title {
    color: #00f3ff;
    font-weight: 800;
    font-size: 1.2rem;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.flow-grid {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 800px;
    margin: 0 auto;
    position: relative;
}

.flow-node {
    background: rgba(20, 15, 45, 0.9);
    border: 2px solid #ff00ea;
    border-radius: 12px;
    padding: 15px;
    color: white;
    font-weight: 600;
    width: 140px;
    position: relative;
    z-index: 2;
    box-shadow: 0 0 15px rgba(255, 0, 234, 0.2);
    animation: pulse-node 3s infinite;
}

.flow-node.cyan {
    border-color: #00f3ff;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
}

.flow-line {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    flex-grow: 1;
    position: relative;
    z-index: 1;
}

.data-packet {
    position: absolute;
    top: -3px;
    left: 0;
    width: 10px;
    height: 10px;
    background: #00f3ff;
    border-radius: 50%;
    box-shadow: 0 0 10px #00f3ff, 0 0 20px #00f3ff;
    animation: travel 4s linear infinite;
}

@keyframes travel {
    0% { left: 0; opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { left: 100%; opacity: 0; }
}

@keyframes pulse-node {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

/* ── Custom Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
::-webkit-scrollbar-thumb {
    background: #00f3ff;
    border-radius: 4px;
}
"""

# ── Example Queries ──────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "What is Corrective RAG and how does it improve standard RAG pipelines?",
    "Explain how vector embeddings enable semantic search.",
    "What are the differences between BM25 and dense retrieval?",
    "How does LangGraph handle cyclic workflows for multi-agent systems?",
    "What techniques are used for hallucination detection in LLMs?",
    "How does HyDE (Hypothetical Document Embeddings) work?",
    "What is the role of a Cross-Encoder in document re-ranking?",
    "How can you prevent prompt injection in enterprise AI models?",
    "Explain the concept of chunking strategies in Vector Databases.",
    "What is self-reflection in the context of autonomous AI agents?",
    "Compare Llama 3.3 70B with other open-weights models for reasoning.",
    "How does sparse retrieval (BM25) complement dense embeddings?",
    "What are the main advantages of using a graph database vs vector database?",
    "Describe the workflow of a multi-agent system resolving complex queries.",
    "How does an LLM judge whether a retrieved document is relevant or not?",
    "What are the trade-offs of using large embedding models like BGE-large?"
]


# ── Build UI ─────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="pink",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#05030f",
    body_background_fill_dark="#05030f",
    block_background_fill="rgba(10, 8, 30, 0.7)",
    block_background_fill_dark="rgba(10, 8, 30, 0.7)",
    block_border_color="rgba(0, 243, 255, 0.3)",
    block_border_color_dark="rgba(0, 243, 255, 0.3)",
    body_text_color="#fff",
    body_text_color_dark="#fff",
)

def build_app():
    """Build the Gradio application."""
    
    with gr.Blocks(
        title="CRAG AI | Enterprise RAG"
    ) as app:
        
        # ── Header ────────────────────────────────────────────────────────
        with gr.Column(elem_classes="header-section"):
            gr.HTML("""
                <h1>🧠 CRAG AGENTIC RAG</h1>
                <p>Self-Correcting Enterprise AI Search Engine</p>
            """)
            
        # ── Animated Explainer ────────────────────────────────────────────
        gr.HTML("""
            <div class="explainer-container">
                <div class="explainer-title">System Data Flow ⚡</div>
                <div class="flow-grid">
                    <div class="flow-node cyan">User Query</div>
                    <div class="flow-line"><div class="data-packet" style="animation-delay: 0s;"></div></div>
                    <div class="flow-node">Hybrid Search &<br>Cross-Encoder</div>
                    <div class="flow-line"><div class="data-packet" style="animation-delay: 1.5s;"></div></div>
                    <div class="flow-node cyan">Grader Agent<br>Verify Docs</div>
                    <div class="flow-line"><div class="data-packet" style="animation-delay: 3s;"></div></div>
                    <div class="flow-node">Llama 3.3 70B<br>Generation</div>
                </div>
            </div>
        """)
        
        with gr.Row():
            # ── Main Chat Column ──────────────────────────────────────────
            with gr.Column(scale=3):
                
                # Pill row for examples
                import random
                initial_qs = random.sample(EXAMPLE_QUERIES, 4)
                
                with gr.Row(elem_classes="pill-row"):
                    btn1 = gr.Button(initial_qs[0], elem_classes="example-pill")
                    btn2 = gr.Button(initial_qs[1], elem_classes="example-pill")
                    btn3 = gr.Button(initial_qs[2], elem_classes="example-pill")
                    btn4 = gr.Button(initial_qs[3], elem_classes="example-pill")
                    example_btns = [btn1, btn2, btn3, btn4]
                
                chatbot = gr.Chatbot(
                    label="",
                    elem_classes="chatbot",
                    height=550,
                    render_markdown=True,
                )
                
                with gr.Row(elem_classes="input-row"):
                    msg_input = gr.Textbox(
                        placeholder="Initialize the system, then ask a complex question...",
                        show_label=False,
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    mic_input = gr.Audio(
                        sources=["microphone"], 
                        type="filepath",
                        scale=1,
                        min_width=50,
                        show_label=False,
                        container=False
                    )
                    send_btn = gr.Button(
                        "Send ⚡",
                        variant="primary",
                        scale=1,
                        min_width=100,
                    )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
                    retry_btn = gr.Button("🔄 Retry Last", variant="secondary", size="sm")
            
            # ── Side Panel ────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                # System Status
                with gr.Group(elem_classes="status-panel"):
                    gr.Markdown("### ⚡ System Control")
                    init_btn = gr.Button(
                        "🚀 INITIALIZE SYSTEM",
                        variant="primary",
                        size="lg",
                    )
                    init_status = gr.Textbox(
                        label="Status",
                        value="⏳ Offline.",
                        interactive=False,
                        lines=2,
                    )
                    
                    gr.Markdown("### 🧠 Agent Brain Log")
                    agent_terminal = gr.Markdown(
                        "```\n> System Online.\n> Awaiting connection...\n```", 
                        elem_classes="terminal-box"
                    )
                
                # Architecture Info
                with gr.Accordion("🏗️ System Architecture", open=False, elem_classes="accordion"):
                    gr.Markdown("""
**Pipeline Steps:**
1. 📥 **Hybrid Retriever**
2. 🔍 **Grader Agent**
3. 🌐 **Web Search**
4. 🤖 **Generator (70B)**
                    """)
        
        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML("""
            <div class="footer-text">
                <strong>System Status:</strong> Hybrid Search Active • Llama 3.3 70B Online • Cross-Encoder Re-ranking Enabled<br/>
                <span style="color: #00f3ff; font-size: 0.85em;">Responses may take ~10 seconds due to rigorous fact-checking and grading.</span>
            </div>
        """)
        
        # ── Event Handlers ────────────────────────────────────────────────
        init_btn.click(
            fn=initialize_system,
            outputs=init_status,
        )
        
        def user_message(message, history):
            if isinstance(message, (list, tuple)):
                message = message[0] if len(message) > 0 else ""
                if isinstance(message, dict) and "text" in message:
                    message = message["text"]
            message = str(message)
            
            if not message.strip():
                return "", history
            history = history + [{"role": "user", "content": message}]
            return "", history
            
        mic_input.change(
            fn=transcribe_audio,
            inputs=mic_input,
            outputs=msg_input,
        )
        
        def bot_response(history):
            global workflow_app, is_initialized
            if not history:
                yield history, gr.update()
                return
            
            if not is_initialized:
                history[-1]["content"] += "\n\n⚠️ System not initialized. Click **Initialize System** first."
                yield history, gr.update()
                return
            
            user_msg = history[-1]["content"]
            history.append({"role": "assistant", "content": ""})
            
            terminal_text = "```\n> System Online. Processing query...\n"
            yield history, gr.update(value=terminal_text + "```")
            
            try:
                from graph.workflow import stream_query
                final_result = None
                
                for chunk in stream_query(workflow_app, user_msg):
                    if "log" in chunk:
                        log_lines = chunk["log"]
                        terminal_text = "```\n" + "\n".join(f"> {line}" for line in log_lines) + "\n```"
                        yield history, gr.update(value=terminal_text)
                    else:
                        final_result = chunk
                        
                if final_result:
                    generation = final_result.get("generation", "No response generated.")
                    web_search = final_result.get("web_search", "No")
                    retry_count = final_result.get("retry_count", 0)
                    documents = final_result.get("documents", [])
                    
                    score_html = ""
                    if documents:
                        score_html += "<div style='margin-top:15px; border-top:1px solid rgba(0,243,255,0.3); padding-top:10px;'>"
                        score_html += "<h4 style='color:#00f3ff; margin-bottom:10px;'>📊 Document Confidence</h4>"
                        for i, doc in enumerate(documents[:3]):
                            score = doc.metadata.get("rerank_score", 0.0)
                            import math
                            try:
                                pct = int(100 / (1 + math.exp(-score)))
                            except OverflowError:
                                pct = 100 if score > 0 else 0
                            
                            color = "#00ff41" if pct > 60 else "#ffaa00"
                            if pct < 30: color = "#ff00ea"
                            
                            score_html += f"<div style='margin-bottom:8px;'>"
                            score_html += f"<div style='display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:3px;'><span>Source {i+1} Relevancy</span><span style='color:{color}; font-weight:bold;'>{pct}%</span></div>"
                            score_html += f"<div style='width:100%; background:rgba(255,255,255,0.1); border-radius:4px; height:8px; overflow:hidden;'>"
                            score_html += f"<div style='width:{pct}%; background:{color}; height:100%; border-radius:4px; transition:width 1s ease-out;'></div>"
                            score_html += "</div></div>"
                        score_html += "</div>"
                        
                    response = f"{generation}\n\n---\n"
                    response += f"⚙️ Web Search: {'🌐 Triggered' if web_search == 'Yes' else '📚 Local RAG'} | "
                    response += f"Retries: {retry_count} | Sources: {len(documents)}"
                    response += score_html
                    
                    history[-1]["content"] = response
                    terminal_text = terminal_text.replace("```", "") + "> GENERATION COMPLETE.\n```"
                    yield history, gr.update(value=terminal_text)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                history[-1]["content"] = f"❌ Error processing query: {str(e)}"
                terminal_text = terminal_text.replace("```", "") + f"> ERROR: {str(e)}\n```"
                yield history, gr.update(value=terminal_text)
                
        def retry_last(history):
            if not history or len(history) < 2:
                return history
            history = history[:-1]
            return history
            
        def rotate_examples():
            import random
            new_qs = random.sample(EXAMPLE_QUERIES, 4)
            return [gr.update(value=q) for q in new_qs]

        # Automatic timer for example rotation
        gr.Timer(30).tick(
            fn=rotate_examples,
            outputs=example_btns
        )
        
        for btn in example_btns:
            btn.click(
                fn=lambda text: text,
                inputs=[btn],
                outputs=[msg_input],
            )
        
        send_btn.click(
            fn=user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            fn=bot_response,
            inputs=chatbot,
            outputs=[chatbot, agent_terminal],
        ).then(
            fn=rotate_examples,
            outputs=example_btns
        )
        
        msg_input.submit(
            fn=user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            fn=bot_response,
            inputs=chatbot,
            outputs=[chatbot, agent_terminal],
        ).then(
            fn=rotate_examples,
            outputs=example_btns
        )
        
        clear_btn.click(fn=lambda: [], outputs=chatbot)
        
        retry_btn.click(
            fn=retry_last,
            inputs=chatbot,
            outputs=chatbot,
        ).then(
            fn=bot_response,
            inputs=chatbot,
            outputs=[chatbot, agent_terminal],
        )
    
    return app


# Create app at top level for Vercel
app = build_app()

# Local development only
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🚀 CRAG AI — Advanced Retrieval-Augmented Generation")
    print("="*60)
    print("\nStarting application...")
    print("Make sure you have set GROQ_API_KEY and TAVILY_API_KEY in .env\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=THEME,
    )
