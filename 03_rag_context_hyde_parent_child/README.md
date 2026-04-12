# Phase 03 — Context engineering (HyDE + parent–child)

**Directory:** `03_rag_context_hyde_parent_child`  
**Role in the curriculum:** Improves **what gets embedded** vs **what gets read** — addressing asymmetric query/document length and fragmented chunks.

## Tactics implemented

| Tactic | Implementation notes |
|--------|----------------------|
| Small-to-big / parent–child | **Child** chunks for precise retrieval; **parent** spans (or larger units) returned to the LLM for broader context (LangChain `ParentDocumentRetriever` pattern). |
| HyDE | LLM drafts a **hypothetical answer**; that text is embedded and used for retrieval so the query vector lives in “answer-like” space. |
| Optional HyDE flag | Request controls whether HyDE runs (see `QueryRequest` in `app/models/schemas.py`). |

## Problems addressed

1. **Asymmetric retrieval:** short questions vs long normative prose in docs — HyDE lengthens the query side for embedding.  
2. **Granularity tradeoff:** tiny chunks retrieve well but starve context; large chunks add noise — parent–child splits the difference.

## How to run

1. `cd 03_rag_context_hyde_parent_child`  
2. Venv + `pip install -r requirements.txt`  
3. `GOOGLE_API_KEY` in repo-root `.env`  
4. `uvicorn app.main:app --reload`  
5. `/docs`

**Previous:** [`02_rag_retrieval_hybrid_bm25_dense`](../02_rag_retrieval_hybrid_bm25_dense/) · **Next:** [`04_rag_post_retrieval_rerank_compress`](../04_rag_post_retrieval_rerank_compress/)
