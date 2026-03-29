# Module 2: Advanced Retrieval Strategies

This module introduces **Hybrid Search**, resolving a major flaw in Naive RAG: Vector databases are great at semantic concepts, but terrible at exact keyword matching (like finding a specific ID, an acronym, or an exact name).

## Features Added in Module 2
- **Sparse Retrieval (BM25)**: A classic keyword-search algorithm based on term frequency-inverse document frequency (TF-IDF). Perfect for exact keyword matches.
- **Ensemble Retriever**: A LangChain component that runs multiple retrievers in parallel and merges their results using Reciprocal Rank Fusion (RRF).
- **Hybrid Search (Alpha Weighting)**: We combine Dense (ChromaDB + Gemini Embeddings) and Sparse (BM25) search.
- **Dynamic Routing**: The API now lets you choose the retrieval strategy via the request body: `"dense"`, `"sparse"`, or `"hybrid"`.

## The Flaw Solved
If you query Naive RAG for "Kubernetes version 1.27.3", dense vector search might return chunks about "Docker", "Containers", or "Orchestration" because they are semantically similar concepts, but miss the exact version number. BM25 catches the exact string "1.27.3". Hybrid Search gives you the best of both worlds.

## How to Run
1. Stop your Module 1 server.
2. Navigate into this directory: `cd 02_Advanced_Retrieval`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `.\venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Start the server: `uvicorn app.main:app --reload`
7. Test the API at `http://localhost:8000/docs`.

*(Note: We will re-use the `.env` file you created in the root directory!)*