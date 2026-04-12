# Phase 05 — Agentic RAG (LangGraph · grading · rewrite loops)

**Directory:** `05_rag_agentic_langgraph`  
**Role in the curriculum:** Replace a **fixed linear chain** with a **state machine / graph**: retrieve → grade → branch (generate vs rewrite vs re-retrieve).

## Tactics implemented

| Tactic | Implementation notes |
|--------|----------------------|
| LangGraph | Stateful graph: nodes for retrieval, **document grading**, generation, **query rewriting**. |
| CRAG / Self-RAG–style behavior | LLM judges whether retrieved text is usable; on failure, query can be rewritten and retrieval repeated (bounded by `max_retries` etc.). |
| Observable trajectory | Response metadata can reflect the path taken (e.g. grade accept/reject, rewrites). |

## Problem addressed

**Garbage in, garbage out:** linear RAG still answers from bad retrieval. Agentic flow adds a **verification gate** before committing tokens to a final answer.

## How to run

1. `cd 05_rag_agentic_langgraph`  
2. Venv + `pip install -r requirements.txt` (includes **langgraph**).  
3. `GOOGLE_API_KEY` in repo-root `.env`  
4. `uvicorn app.main:app --reload`  
5. `/docs`

**Previous:** [`04_rag_post_retrieval_rerank_compress`](../04_rag_post_retrieval_rerank_compress/) · **Next:** [`06_rag_production_semantic_cache_eval`](../06_rag_production_semantic_cache_eval/)
