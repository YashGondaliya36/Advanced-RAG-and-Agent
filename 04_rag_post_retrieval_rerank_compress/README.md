# Phase 04 — Post-retrieval optimization (rerank · compress · reorder)

**Directory:** `04_rag_post_retrieval_rerank_compress`  
**Role in the curriculum:** Everything that happens **after** the first retrieval list and **before** the final LLM call — reducing noise and ordering context for attention.

## Tactics implemented

| Tactic | Implementation notes |
|--------|----------------------|
| Reranking | **FlashRank** — lightweight **cross-encoder–style** scoring of (query, document) pairs to re-order or filter chunks. |
| Contextual compression | Pipeline to drop or shorten low-value chunks before generation. |
| Long-context reordering | Mitigates **“lost in the middle”** by placing high-confidence evidence at **primacy / recency** positions in the assembled prompt. |

## API ideas

Request flags typically control rerank on/off, compression, and reordering (see `QueryRequest` in this module).

## How to run

1. `cd 04_rag_post_retrieval_rerank_compress`  
2. Venv + `pip install -r requirements.txt` (includes FlashRank / related deps).  
3. `GOOGLE_API_KEY` in repo-root `.env`  
4. `uvicorn app.main:app --reload`  
5. `/docs`

**Previous:** [`03_rag_context_hyde_parent_child`](../03_rag_context_hyde_parent_child/) · **Next:** [`05_rag_agentic_langgraph`](../05_rag_agentic_langgraph/)
