# Module 1: Naive RAG

This is the foundational implementation of a Retrieval-Augmented Generation (RAG) system, built to demonstrate the core architecture before we introduce advanced retrieval strategies.

## Features
- **FastAPI**: Modular, asynchronous endpoints (`/api/v1/ingest`, `/api/v1/query`).
- **Google Gemini**: Uses `gemini-3.1-flash` for text generation and `models/embedding-001` for dense vector embeddings.
- **ChromaDB**: A lightweight, local vector database for fast similarity search.
- **Document Chunking**: Implements `RecursiveCharacterTextSplitter` to break down large texts with overlap to preserve context.

## The "Naive" RAG Flow
1. **Ingestion**: Raw Text -> Chunking -> Embedding -> Storage in Vector DB.
2. **Retrieval**: User Query -> Embedding -> Top-K Semantic Similarity Search -> Retrieved Chunks.
3. **Generation**: System Prompt + Retrieved Chunks + User Query -> LLM -> Grounded Answer.

## Limitations Addressed in Future Modules
- **Lost in the Middle**: No context reordering is done here, meaning the LLM might ignore facts buried in the middle of the retrieved chunks.
- **Noise**: The system blindly fetches exactly K chunks, even if they are irrelevant to the user's specific query.
- **Poor Logic**: Semantic search (Cosine Similarity) fails at strict metadata constraints (e.g., "articles from 2026"). We will fix this with Self-Querying in later modules.

## How to Run
1. Navigate into this directory: `cd 01_Naive_RAG`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file based on `.env.example` and add your `GOOGLE_API_KEY`.
6. Run the server: `uvicorn app.main:app --reload`
7. Test the API at `http://localhost:8000/docs`.