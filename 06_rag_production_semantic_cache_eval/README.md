# Phase 06 — Production patterns (semantic cache + RAGAS-style eval)

**Directory:** `06_rag_production_semantic_cache_eval`  
**Role in the curriculum:** **Cost and latency** (avoid repeated LLM calls for paraphrased questions) and **quality measurement** (automated scores on answers vs context).

## Tactics implemented

| Tactic | Implementation notes |
|--------|----------------------|
| Semantic caching | **GPTCache** with **SQLite** scalar store + **FAISS** ANN search; embeddings from **Gemini**; **cosine** threshold on stored vectors vs query (see `rag_service.py`). |
| Cache API | `use_cache` on query; response includes `cache_hit`, `latency_ms`. |
| Evaluation | **RAGAS** integration for metrics such as **faithfulness** and **answer relevancy** (see `eval_service.py` and eval routes). |

## Operational notes

- If you change embedding model or dimension, **delete** the local `semantic_cache.sqlite3` and `faiss.index` in this folder and let the app rebuild (dimension mismatch otherwise).  
- Install **faiss-cpu** per `requirements.txt` for vector search in the cache layer.

## How to run

1. `cd 06_rag_production_semantic_cache_eval`  
2. Venv + `pip install -r requirements.txt`  
3. `GOOGLE_API_KEY` in repo-root `.env`  
4. `uvicorn app.main:app --reload`  
5. `/docs` — try `/query` twice with paraphrases to observe **cache hit** behavior.

**Previous:** [`05_rag_agentic_langgraph`](../05_rag_agentic_langgraph/)

Complex sample text for retrieval tests: `test_data_complex.json`.
