# Advanced RAG and Agentic AI (Learning Lab)

This repository is a **step-by-step curriculum**: each folder is a **self-contained FastAPI service** that demonstrates one layer of modern RAG. Together they form a pipeline from naive dense retrieval to production-style caching and evaluation.

All language-model calls use **Google Gemini** (generation and embeddings via the `google-genai` SDK where implemented).

---

## Repository map (what each phase is)

| Directory | Phase focus | Primary tactics |
|-----------|-------------|-----------------|
| [`01_rag_foundational_dense_vectors`](01_rag_foundational_dense_vectors/) | Baseline RAG | Chunking (recursive character split), **dense retrieval** (embeddings + cosine-style similarity), ChromaDB, single-pass **query → retrieve → generate** |
| [`02_rag_retrieval_hybrid_bm25_dense`](02_rag_retrieval_hybrid_bm25_dense/) | Retrieval quality | **Sparse retrieval (BM25)**, dense vectors, **hybrid ensemble** (weighted LangChain `EnsembleRetriever`), per-request **`strategy`** routing |
| [`03_rag_context_hyde_parent_child`](03_rag_context_hyde_parent_child/) | Context and indexing | **Parent–child (small-to-big)** retrieval, **HyDE** (hypothetical document for embedding) |
| [`04_rag_post_retrieval_rerank_compress`](04_rag_post_retrieval_rerank_compress/) | After retrieval | **Cross-encoder–style reranking (FlashRank)**, **contextual compression**, **long-context reordering** (“lost in the middle” mitigation) |
| [`05_rag_agentic_langgraph`](05_rag_agentic_langgraph/) | Non-linear pipelines | **LangGraph**, **CRAG / Self-RAG–style** loops, **LLM grading** of documents, **query rewriting** |
| [`06_rag_production_semantic_cache_eval`](06_rag_production_semantic_cache_eval/) | Production concerns | **Semantic caching (GPTCache + FAISS + SQLite)**, **RAGAS**-style evaluation hooks |

Read the **README inside each directory** for the exact API, request fields, and run commands.

---

## How the phases align with common RAG vocabulary

- **Naive vs advanced RAG** — Phase 01 is deliberately naive; phases 02–06 add routing, context tricks, post-retrieval filters, agents, and ops.
- **Sparse vs dense** — Phase 02 (BM25 vs embeddings).
- **Hybrid / ensemble** — Phase 02 (combined retrieval).
- **Vector DB** — ChromaDB across modules (persistent dirs per app).
- **Chunking** — Phase 01 (recursive splits); phase 03 extends with hierarchical parent–child.
- **HyDE** — Phase 03.
- **Reranking (cross-encoder class)** — Phase 04 (FlashRank as a lightweight reranker).
- **Contextual compression** — Phase 04.
- **Lost in the middle / reordering** — Phase 04.
- **Agentic workflows / LangGraph** — Phase 05.
- **Caching and evaluation** — Phase 06.

Topics named in many curricula but **not implemented as separate top-level apps** here (good next extensions): **Self-querying retrieval** (metadata filters from NL), **RAG Fusion / multi-query + RRF**, **Merger retriever across DBs**, **Cohere Rerank API**, **LLM quantization** (BitsAndBytes/GGUF), **Weaviate**. The current layout keeps each service runnable and focused.

---

## Prerequisites

- Python 3.10+ recommended  
- A **Google AI API key** in a shared `.env` at the repo root:

```env
GOOGLE_API_KEY=your_key_here
```

Each module’s `app/core/config.py` points at `../.env` (or equivalent) so one key serves all phases.

---

## How to run (pattern for every phase)

1. `cd <phase_directory>` (see table above).  
2. Create and activate a virtual environment.  
3. `pip install -r requirements.txt`  
4. `uvicorn app.main:app --reload`  
5. Open `http://127.0.0.1:8000/docs` for OpenAPI.

Use **one server at a time** unless you change ports; each module is an independent app.

---

## Suggested learning order

Work through folders **01 → 06** in order. Each phase assumes you understand the previous one’s limitations, which the next README calls out.

---

## License

See [`LICENSE`](LICENSE).
