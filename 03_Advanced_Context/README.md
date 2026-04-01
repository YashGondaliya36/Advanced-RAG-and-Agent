# Module 3: Advanced Context & Chunking

This module solves two massive problems in RAG systems: the **Asymmetric Search Problem** and the **Lost Context Problem**. 

## Features Added in Module 3
- **Small-to-Big Retrieval (Parent-Child Indexing)**: Also known as Sentence Window Retrieval. We chunk documents twice: once into tiny sentences (Children), and once into large paragraphs (Parents). We search the Vector DB against the tiny children for extreme precision, but we pass the large Parent chunk to the LLM to give it massive surrounding context!
- **HyDE (Hypothetical Document Embeddings)**: Solves the Asymmetric Search Problem. When a user asks a short question, we use an LLM to hallucinate a fake, perfect answer. We then vectorize that *fake answer* and search the database for it, drastically improving semantic matching!

## The Flaws Solved
1. **Asymmetric Search**: Users write short questions ("What is the password policy?"). Documents are written as long answers ("The password policy requires 14 characters, expires in 90 days, and..."). Cosine similarity fails because short questions mathematically look very different from long answers. **HyDE** fixes this by converting the short question into a long hypothetical answer *before* searching.
2. **Context Loss**: If you chunk documents too small, the LLM loses the broader context of the topic. If you chunk them too large, the Vector DB fetches irrelevant noise. **Small-to-Big Retrieval** gives you the mathematical precision of tiny chunks, with the reading comprehension of massive chunks.

## How to Run
1. Stop your Module 2 server.
2. Navigate into this directory: `cd 03_Advanced_Context`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `.\venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Start the server: `uvicorn app.main:app --reload`
7. Test the API at `http://localhost:8000/docs`.