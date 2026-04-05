# Module 6: Enterprise Scale (Semantic Caching & Evaluation)

Welcome to the final module! Once a RAG pipeline is built, the two biggest issues in production are **Cost/Latency** and **Quality Assurance**.

## Features Added in Module 6
- **Semantic Caching (GPTCache)**: If User A asks "How do I reset my password?", the LLM calculates the answer. If User B asks "What is the password reset process?", we shouldn't pay Google API fees to generate the same answer again. Semantic Caching uses Vector Math to realize the *intent* is the same, and instantly returns the cached answer in 1 millisecond.
- **RAGAS Evaluation Framework**: How do you mathematically prove your RAG is good? RAGAS (Retrieval Augmented Generation Assessment) scores your pipeline automatically on two metrics:
    - **Faithfulness**: Did the LLM hallucinate, or is the answer 100% backed by the retrieved context?
    - **Answer Relevancy**: Did the LLM actually answer the user's question, or did it ramble about something else?

## The Flaw Solved
Basic RAG costs money for every single query, even duplicates. It also has no way to detect if it's slowly degrading in quality. This module provides extreme cost-savings and quantitative quality metrics.

## How to Run
1. Stop your Module 5 server.
2. Navigate into this directory: `cd 06_Enterprise_Scale`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `.\venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Start the server: `uvicorn app.main:app --reload`
7. Test the API at `http://localhost:8000/docs`.