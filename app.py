"""
app.py
Gradio-based Multilingual PDF Constrained Conversational Agent.

Usage:
    python app.py
    # or for Hugging Face Spaces: just push this directory
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Optional

# Load .env file if present (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr
from gradio_client import utils as _gc_utils

from llm_interface import ask_llm, ask_llm_stream, DEFAULT_MODEL
from pdf_processor import PDFProcessor
from utils import build_prompt, detect_language, format_citations

# ---------------------------------------------------------------------------
# Compatibility patch — gradio_client passes bool schemas to get_type(),
# which does `"const" in schema` and crashes on non-dict types.
# ---------------------------------------------------------------------------
_orig_get_type = getattr(_gc_utils, "get_type", None)
if _orig_get_type is not None:
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "boolean" if isinstance(schema, bool) else "any"
        return _orig_get_type(schema)
    _gc_utils.get_type = _safe_get_type


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

processor: Optional[PDFProcessor] = None
pdf_display_name: str = ""
AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "query_audit.jsonl")


def _confidence_label(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.52:
        return "medium"
    return "low"


def _append_audit_log(entry: dict) -> None:
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Could not write audit log: %s", exc)


def _backend_label() -> str:
    return f"`groq` → `{DEFAULT_MODEL}`"


def load_recent_audits() -> str:
    limit = 12
    if not os.path.exists(AUDIT_LOG_PATH):
        return "No audit logs yet. Ask a question to generate entries."

    try:
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as exc:
        return f"Could not read audit log: {exc}"

    if not lines:
        return "Audit log file is empty."

    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue

    if not entries:
        return "No valid JSON audit entries found."

    output = ["### Recent Query Audit"]
    for i, e in enumerate(reversed(entries), start=1):
        query = e.get("query", "")[:100].replace("\n", " ")
        confidence = e.get("confidence", "unknown")
        best_score = e.get("best_score", 0.0)
        lang = e.get("language", "unknown")
        verified = e.get("citation_verified", False)
        pages = e.get("citations", [])
        output.append(
            f"{i}. `{lang}` | score `{best_score:.2f}` | confidence `{confidence}` | "
            f"citation_verified `{verified}` | pages `{pages}`\n"
            f"   - Q: {query}"
        )
    return "\n".join(output)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return re.sub(r"[^\w\s]+", "", lowered, flags=re.UNICODE).strip()


def _extract_ascii_quotes(text: str) -> list[str]:
    return [q.strip() for q in re.findall(r"\"([^\"]{8,})\"", text) if q.strip()]


def _verify_quotes_in_chunks(answer: str, chunks: list[dict]) -> tuple[bool, list[str]]:
    quotes = _extract_ascii_quotes(answer)
    if not quotes:
        return False, []
    chunk_blob = " ".join(c["text"] for c in chunks)
    norm_blob = _normalize_text(chunk_blob)
    missing = []
    for quote in quotes:
        if _normalize_text(quote) not in norm_blob:
            missing.append(quote)
    return len(missing) == 0, missing


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[^\W_]{2,}", _normalize_text(text), flags=re.UNICODE)}


def _max_evidence_overlap(answer: str, chunks: list[dict]) -> float:
    answer_tokens = _token_set(answer)
    if not answer_tokens:
        return 0.0
    max_overlap = 0.0
    for chunk in chunks:
        chunk_tokens = _token_set(chunk["text"])
        if not chunk_tokens:
            continue
        overlap = len(answer_tokens & chunk_tokens) / max(1, len(answer_tokens))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _is_broad_in_scope_query(query: str) -> bool:
    q = query.lower().strip()
    broad_markers = [
        "main topic", "subject of this document", "purpose", "key findings",
        "key points", "summary", "summarize", "main idea", "मुख्य विषय", "मुख्य बिंदु",
        "उद्देश्य", "resumen", "puntos principales", "objetivo",
    ]
    return any(marker in q for marker in broad_markers)


def process_pdf(file_obj) -> tuple[str, str]:
    global processor, pdf_display_name

    if file_obj is None:
        return "⚠️ No file received.", ""

    try:
        processor = PDFProcessor()
        pdf_display_name = os.path.basename(file_obj.name)
        num_chunks = processor.chunk_pdf(file_obj.name)
        status = (
            f"✅ **{pdf_display_name}** loaded successfully!\n\n"
            f"- Chunks indexed: **{num_chunks}**\n"
            f"- Embedding model: `intfloat/multilingual-e5-small`\n"
            f"- LLM backend: {_backend_label()}"
        )
        return status, pdf_display_name
    except Exception as exc:
        logger.exception("Failed to process PDF")
        return f"❌ Error loading PDF: {exc}", ""


CONF_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def _prepare_retrieval(message, history, top_k, min_similarity):
    """
    Run language detection, retrieval, and similarity threshold check.
    Returns (prompt, chunks, lang_code, lang_name, best_score, confidence)
    or (error_string, None, ...) on early exit.
    """
    lang_code, lang_name = detect_language(message)
    logger.info("Detected language: %s (%s)", lang_name, lang_code)

    if lang_code != "en":
        min_similarity = min(min_similarity, 0.30)

    chunks = processor.retrieve(message, top_k=top_k)
    if not chunks:
        error = f"❌ No relevant content found in the PDF.\n\n_(Detected language: **{lang_name}**)_"
        return error, None, lang_code, lang_name, 0.0, "low"

    best_score = float(chunks[0]["score"])
    confidence = _confidence_label(best_score)

    if best_score < min_similarity:
        top_chunk = chunks[0]
        snippet = top_chunk["text"][:260].replace("\n", " ").strip()
        error = (
            "I cannot answer based on the provided PDF.\n\n"
            f"Closest context (Page {top_chunk['page']}, similarity: {best_score:.2f}):\n"
            f'> "{snippet}..."\n\n'
            "Try a more specific question using terms from the document."
        )
        return error, None, lang_code, lang_name, best_score, confidence

    prompt = build_prompt(message, chunks, lang_name, history)
    return prompt, chunks, lang_code, lang_name, best_score, confidence



CSS = """
:root {
    --bg-color: #343541;
    --sidebar-bg: #202123;
    --text-color: #ececf1;
    --accent-color: #10a37f;
}
body, .gradio-container {
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
}
#app-container {
    min-height: 100vh !important;
}
#sidebar {
    background-color: var(--sidebar-bg) !important;
    padding: 15px !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
}
#new-chat-btn {
    background-color: transparent !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    text-align: left !important;
    padding: 12px !important;
    border-radius: 6px !important;
    margin-bottom: 20px !important;
    cursor: pointer !important;
    width: 100% !important;
}
#new-chat-btn:hover {
    background-color: rgba(255,255,255,0.1) !important;
}
#chat-list {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#chat-list label {
    background: transparent !important;
    border: none !important;
    color: white !important;
    padding: 10px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    margin-bottom: 2px !important;
}
#chat-list label:hover {
    background: rgba(255,255,255,0.05) !important;
}
#chat-list span {
    color: white !important;
    font-size: 0.9em !important;
}
#main-area {
    background-color: var(--bg-color) !important;
    padding: 0 !important;
}
#chatbot {
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
}
#input-container {
    padding: 10px 20px !important;
    background: transparent !important;
}
#input-area textarea {
    background: #40414f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: white !important;
}
#submit-btn {
    background: transparent !important;
    border: none !important;
    color: var(--accent-color) !important;
    font-weight: bold !important;
    cursor: pointer !important;
    width: auto !important;
    padding: 10px !important;
}
#pdf-status {
    font-size: 0.85em !important;
    color: #aaa !important;
    margin-top: 5px !important;
    padding: 0 20px !important;
}
#header-area {
    text-align: center;
    padding: 15px 20px 10px 20px;
    background-color: transparent;
}
#main-header {
    margin-bottom: 10px;
}
#main-header h2 {
    font-size: 1.8em;
    margin-bottom: 5px;
}
#sample-prompts {
    display: flex !important;
    justify-content: center !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
    max-width: 800px !important;
    margin: 0 auto 10px auto !important;
}
.sample-prompt {
    background-color: #40414f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: var(--text-color) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 0.9em !important;
    cursor: pointer !important;
    flex: 1 1 calc(50% - 10px) !important;
    min-width: 180px !important;
    text-align: left !important;
}
.sample-prompt:hover {
    background-color: #2A2B32 !important;
}
"""


def build_ui():
    with gr.Blocks(
        title="📄 Multilingual PDF Chat",
        theme=gr.themes.Base(),
        css=CSS
    ) as demo:
        
        # State
        chat_data = gr.State({"Chat 1": []})
        current_chat = gr.State("Chat 1")
        chat_counter = gr.State(1)

        with gr.Row(elem_id="app-container"):
            
            # --- SIDEBAR ---
            with gr.Column(elem_id="sidebar", scale=2, min_width=250):
                new_chat_btn = gr.Button("+ New Chat", elem_id="new-chat-btn")
                
                chat_list = gr.Radio(
                    choices=["Chat 1"], 
                    value="Chat 1", 
                    show_label=False,
                    elem_id="chat-list"
                )

                gr.Markdown("---")
                pdf_input = gr.File(
                    label="📎 Upload PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                pdf_status = gr.Markdown("Upload a PDF to begin chatting.", elem_id="pdf-status")



            # --- MAIN AREA ---
            with gr.Column(elem_id="main-area", scale=8):
                gr.Markdown("### 📄 Docu_Verse — Ask in any language, answers grounded in your PDF.", elem_id="main-header")
                active_pdf_label = gr.Markdown("", elem_id="pdf-status")

                chatbot = gr.Chatbot(height=380, show_label=False, elem_id="chatbot")

                with gr.Row(elem_id="input-container"):
                    msg_input = gr.Textbox(show_label=False, placeholder="Send a message...", container=False, scale=9)
                    submit_btn = gr.Button("➤", elem_id="submit-btn", scale=1)

                with gr.Row(elem_id="sample-prompts"):
                    prompt1 = gr.Button("What is the main topic?", elem_classes="sample-prompt")
                    prompt2 = gr.Button("Summarize the key findings.", elem_classes="sample-prompt")
                    prompt3 = gr.Button("¿Cuáles son las conclusiones?", elem_classes="sample-prompt")
                    prompt4 = gr.Button("इस दस्तावेज़ का मुख्य विषय?", elem_classes="sample-prompt")



        # --- LOGIC ---
        
        # 1. New Chat
        def on_new_chat(data, counter):
            counter += 1
            name = f"Chat {counter}"
            data[name] = []
            return data, counter, name, gr.update(choices=list(data.keys()), value=name), []

        new_chat_btn.click(
            on_new_chat,
            inputs=[chat_data, chat_counter],
            outputs=[chat_data, chat_counter, current_chat, chat_list, chatbot]
        )

        # 2. Switch Chat
        def on_switch_chat(selected, data):
            return selected, data.get(selected, [])

        chat_list.change(
            on_switch_chat,
            inputs=[chat_list, chat_data],
            outputs=[current_chat, chatbot]
        )

        # 3. Submit User Message
        def on_user_submit(msg, curr, data):
            if not msg or not msg.strip():
                return msg, data.get(curr, []), data
            
            history = data.get(curr, [])
            history.append([msg, None])
            data[curr] = history
            return "", history, data

        # 4. Stream Bot Reply
        def on_bot_reply_stream(curr, data):
            top_k = 4
            min_sim = 0.30

            history = data.get(curr, [])
            if not history or history[-1][1] is not None:
                yield history, data
                return

            user_msg = history[-1][0]

            # Show thinking indicator immediately
            history[-1][1] = "_⏳ Searching PDF..._"
            yield history, data

            if processor is None or not processor.is_ready():
                history[-1][1] = "⚠️ Please upload a PDF first."
                data[curr] = history
                yield history, data
                return

            prompt_or_err, chunks, lang_code, lang_name, best_score, confidence = _prepare_retrieval(
                user_msg, history[:-1], top_k, min_sim
            )

            if chunks is None:
                history[-1][1] = prompt_or_err
                data[curr] = history
                yield history, data
                return

            # Stream tokens
            full_response = ""
            for partial in ask_llm_stream(prompt_or_err):
                full_response = partial
                history[-1][1] = partial
                yield history, data

            # Post-process: verify grounding
            citation_verified, _ = _verify_quotes_in_chunks(full_response, chunks)
            evidence_overlap = _max_evidence_overlap(full_response, chunks)
            overlap_threshold = 0.08 if _is_broad_in_scope_query(user_msg) else 0.12
            overlap_verified = evidence_overlap >= overlap_threshold

            if not citation_verified and not overlap_verified:
                full_response += "\n\n> ⚠️ *Note: Could not fully verify grounding against source chunks.*"

            # Build final reply with colour-coded confidence
            conf_icon = CONF_ICON.get(confidence, "⚪")
            citations = format_citations(chunks)
            final_reply = (
                f"{full_response}\n\n"
                f"---\n"
                f"📖 **Source** \u2014 {citations} \u00b7 🌐 **Lang**: {lang_name} \u00b7 "
                f"🎯 **Confidence**: {conf_icon} {confidence} ({best_score:.2f})"
            )

            history[-1][1] = final_reply
            data[curr] = history

            _append_audit_log({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "pdf_name": pdf_display_name,
                "query": user_msg,
                "language": lang_name,
                "language_code": lang_code,
                "best_score": best_score,
                "confidence": confidence,
                "citations": [c["page"] for c in chunks],
                "citation_verified": citation_verified,
                "overlap_verified": overlap_verified,
                "answer": full_response,
            })

            yield history, data

        # Wire up both enter key and submit button
        msg_input.submit(
            on_user_submit,
            inputs=[msg_input, current_chat, chat_data],
            outputs=[msg_input, chatbot, chat_data]
        ).then(
            on_bot_reply_stream,
            inputs=[current_chat, chat_data],
            outputs=[chatbot, chat_data]
        )

        submit_btn.click(
            on_user_submit,
            inputs=[msg_input, current_chat, chat_data],
            outputs=[msg_input, chatbot, chat_data]
        ).then(
            on_bot_reply_stream,
            inputs=[current_chat, chat_data],
            outputs=[chatbot, chat_data]
        )

        # Sample Prompts Wiring
        for btn, text in [
            (prompt1, "What is the main topic of this document?"),
            (prompt2, "Summarize the key findings."),
            (prompt3, "¿Cuáles son las conclusiones principales?"),
            (prompt4, "इस दस्तावेज़ का मुख्य विषय क्या है?")
        ]:
            btn.click(
                lambda t=text: t,
                inputs=[],
                outputs=[msg_input]
            ).then(
                on_user_submit,
                inputs=[msg_input, current_chat, chat_data],
                outputs=[msg_input, chatbot, chat_data]
            ).then(
                on_bot_reply_stream,
                inputs=[current_chat, chat_data],
                outputs=[chatbot, chat_data]
            )

        # PDF Upload
        pdf_name_state = gr.State("")
        pdf_input.upload(
            fn=process_pdf,
            inputs=[pdf_input],
            outputs=[pdf_status, pdf_name_state],
        )
        pdf_name_state.change(
            fn=lambda name: f"📄 **Active PDF:** `{name}`" if name else "",
            inputs=[pdf_name_state],
            outputs=[active_pdf_label],
        )

    return demo

demo = build_ui()
demo.launch(server_name="0.0.0.0", server_port=7860)

