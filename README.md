# Local LLM Demo (Offline) — C# / LLamaSharp / GGUF

A fully **offline**, **no-API** local LLM demo built in C# using **LLamaSharp** (llama.cpp).  
Includes a simple chat UI, streaming responses, and optional RAG (retrieval-augmented generation) using local embeddings + a vector index.

> **Goal:** Make it easy to run a capable chatbot on internal infrastructure with **zero external network dependencies**.

---

## ✨ Highlights

- ✅ **100% offline** (no OpenAI, no web calls, no SaaS)
- ✅ **Streaming chat** (token-by-token)
- ✅ **GGUF model support** via LLamaSharp/llama.cpp
- ✅ **RAG mode** (bring your own documents)
- ✅ **Embeddings** (GGUF embedding model)
- ✅ **Config-driven** (swap models, context size, temperature, etc.)
- ✅ Designed for **internal demos** and “drop-in” expansion

---

## 📸 Screens / Demo

- Chat with local LLM (streaming)
- Toggle “RAG mode” to answer from your indexed docs
- Inspect retrieved chunks + citations (optional)

_(Add screenshots/gifs here once you’re ready.)_

---

## 🧱 Tech Stack

- **.NET**: C# (MVC/Web or service-based host)
- **Inference**: LLamaSharp + llama.cpp backend
- **Models**: GGUF (e.g., Llama-2 chat)
- **Embeddings**: GGUF embedding model (e.g., all-MiniLM-L6-v2 GGUF)
- **RAG**: local chunking + local vector store/index

---

## ✅ Requirements

### Runtime
- Windows x64
- .NET Framework 4.8 (if this is the MVC app) or .NET (if you have a newer host project)
- Visual Studio 2022 recommended

### Models
You’ll need **two GGUF files**:
1. **Chat model** (e.g., Llama-2-*chat*.gguf)
2. **Embedding model** (e.g., `all-minilm-l6-v2-*.gguf`)

> Keep them on a fast disk. SSD strongly recommended.

---

## 🚀 Quick Start

### 1) Clone
```bash
git clone <your-repo-url>
cd <repo-folder>
