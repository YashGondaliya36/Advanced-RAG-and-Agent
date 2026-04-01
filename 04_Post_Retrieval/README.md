# Module 4: Post-Retrieval Optimization

If Modules 2 and 3 solved problems happening **before** and **during** retrieval, Module 4 solves problems happening **after** retrieval but before we generate the answer.

## Features Added in Module 4
- **Cross-Encoder Reranking (FlashRank)**: Standard Vector Databases use Bi-Encoders (fast, but mathematically "dumb"). We use FlashRank, an ultra-lightweight Cross-Encoder that looks at the `[Query]` and `[Document]` simultaneously to calculate an extremely precise semantic score.
- **Contextual Compression**: We fetch 10 documents initially, run them through the Reranker to filter out garbage, and compress the list down to the best 3 documents before sending them to Gemini.
- **Long Context Reordering**: We solve the **"Lost in the Middle" (LiM) phenomenon**. LLMs suffer from primacy bias (remembering the top of the prompt) and recency bias (remembering the bottom), but ignore facts buried in the middle. We actively reorder the retrieved chunks so the highest-scoring chunks are placed at the very top and very bottom of the context!

## How to Run
1. Stop your Module 3 server.
2. Navigate into this directory: `cd 04_Post_Retrieval`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `.\venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Start the server: `uvicorn app.main:app --reload`
7. Test the API at `http://localhost:8000/docs`.