# Module 5: Agentic & Autonomous RAG

Welcome to the cutting-edge of AI. Up until now, our RAG pipelines have been **Linear**.
(Query -> Search -> Generate). 

If the Vector DB fetched garbage documents, the LLM blindly generated a garbage, hallucinated answer. 

## Features Added in Module 5
- **LangGraph Orchestration**: We move from linear chains to **Stateful Graphs**. The AI can now loop, make decisions, and route tasks autonomously.
- **Corrective RAG (CRAG) & Self-RAG**: We introduce an "LLM Grader". Before generating the final answer, an LLM evaluates the retrieved documents. 
    - If the documents are relevant -> Generate the answer.
    - If the documents are GARBAGE -> Reject them, **rewrite the user's query**, and search again!
- **Agent Trajectory**: The API now returns the exact path the AI took (e.g. `['retrieve', 'grade_documents -> REJECT', 'rewrite_query', 'retrieve', 'grade_documents -> ACCEPT', 'generate']`).

## The Flaw Solved
**Garbage In, Garbage Out.** Basic RAG has no self-correction. Agentic RAG gives the LLM the autonomy to realize it doesn't have the right information, and gives it the tools to fix its own mistakes before replying to the user.

## How to Run
1. Stop your Module 4 server.
2. Navigate into this directory: `cd 05_Agentic_RAG`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `.\venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Start the server: `uvicorn app.main:app --reload`
7. Test the API at `http://localhost:8000/docs`.