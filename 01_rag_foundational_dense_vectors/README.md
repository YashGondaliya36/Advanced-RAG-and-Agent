# Phase 01 — Foundational RAG (dense vectors + chunking)

**Directory:** `01_rag_foundational_dense_vectors`  
**Role in the curriculum:** Baseline **naive RAG** — one embedding model, one vector store, fixed top‑k retrieval, no hybrid search, no reranking, no agent loop.

## Tactics implemented (aligned with common RAG checklists)

| Tactic | Implementation notes |
|--------|----------------------|
| Document chunking | `RecursiveCharacterTextSplitter` (overlap preserves boundary context). |
| Dense retrieval | Query and chunks embedded with **Gemini** (`gemini-embedding-001`); similarity search in **ChromaDB**. |
| Naive pipeline | Ingest → embed → store → query → embed → retrieve top‑k → **Gemini** generation (`gemini-2.5-flash` or as configured in code). |
| Async API | FastAPI with async ingestion path where applicable. |

## End-to-end flow

1. **Ingest:** raw text → split → embed → persist in Chroma.  
2. **Query:** question → embed → top‑k chunks → prompt with context → LLM answer + sources.

## Limitations (intentionally fixed in later phases)

- No **BM25** or keyword-first retrieval (see phase 02).  
- No **HyDE** or parent–child indexing (see phase 03).  
- No **reranking** or compression (see phase 04).  
- No **grader / rewrite loop** (see phase 05).  
- No **semantic cache** or batch **eval** (see phase 06).

## How to run

1. `cd 01_rag_foundational_dense_vectors`  
2. `python -m venv venv` then activate (`.\venv\Scripts\activate` on Windows).  
3. `pip install -r requirements.txt`  
4. Ensure repo-root `.env` contains `GOOGLE_API_KEY`.  
5. `uvicorn app.main:app --reload`  
6. Test at `http://127.0.0.1:8000/docs`.

**Next phase:** [`02_rag_retrieval_hybrid_bm25_dense`](../02_rag_retrieval_hybrid_bm25_dense/) — sparse + dense + hybrid routing.
