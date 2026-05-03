---
title: Docu_Verse
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
license: mit
short_description: Multilingual PDF chatbot — ask questions in any language, answers grounded strictly in your document.
---

# Docu_Verse 📄🌐

**Ask questions about any PDF in any language — answers are strictly grounded in your document.**

Built with `sentence-transformers`, `FAISS`, `BM25`, a `CrossEncoder` reranker, and `Groq` (Llama 3.3 70B).

## Features
- 🌐 **Multilingual** — detects your language automatically (30+ supported)
- 🔍 **Hybrid retrieval** — dense embeddings + BM25 + cross-encoder reranking
- 📎 **PDF-constrained** — refuses to answer outside the document
- ⚡ **Streaming** — token-by-token responses with confidence indicators
- 💬 **Multi-session** — manage multiple independent chat sessions
