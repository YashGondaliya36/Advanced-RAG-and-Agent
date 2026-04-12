# Phase 02 — Hybrid retrieval (BM25 + dense vectors)

**Directory:** `02_rag_retrieval_hybrid_bm25_dense`  
**Role in the curriculum:** Shows why **semantic search alone** misses exact tokens (IDs, version strings, SKUs) and how **sparse retrieval** and **ensembles** recover precision.

## Tactics implemented

| Tactic | Implementation notes |
|--------|----------------------|
| Sparse retrieval | **BM25** (keyword statistics; strong on exact matches). |
| Dense retrieval | Chroma + **Gemini** embeddings (same family as phase 01). |
| Hybrid / ensemble | LangChain **`EnsembleRetriever`** over dense + BM25 with **weighted fusion** (e.g. 0.5 / 0.5); API **`strategy`**: `dense`, `sparse`, or `hybrid`. |
| Routing | Single API surface; client picks retrieval mode per query. |

## Problem this phase highlights

Dense vectors may rank “Kubernetes orchestration” highly while a question explicitly targets **“Kubernetes version 1.27.3”**. BM25 boosts rare token overlap; hybrid merges both signals.

## How to run

1. `cd 02_rag_retrieval_hybrid_bm25_dense`  
2. Create/activate venv, then `pip install -r requirements.txt`  
3. Repo-root `.env` with `GOOGLE_API_KEY`  
4. `uvicorn app.main:app --reload`  
5. `http://127.0.0.1:8000/docs`

**Previous:** [`01_rag_foundational_dense_vectors`](../01_rag_foundational_dense_vectors/) · **Next:** [`03_rag_context_hyde_parent_child`](../03_rag_context_hyde_parent_child/)
