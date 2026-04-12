import logging
from typing import Dict, Any, List
from google import genai
from google.genai import types

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.core.config import settings
from app.utils.document_processor import DocumentProcessor
from app.models.state import GraphState

logger = logging.getLogger(__name__)

class GeminiEmbeddings:
    def __init__(self, client: genai.Client):
        self.client = client
        self.model = "gemini-embedding-001"
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        res = self.client.models.embed_content(model=self.model, contents=texts)
        return [emb.values for emb in res.embeddings]
        
    def embed_query(self, text: str) -> list[float]:
        res = self.client.models.embed_content(model=self.model, contents=text)
        return res.embeddings[0].values

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class AgenticRAGService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.embeddings = GeminiEmbeddings(self.client)
        
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.document_processor = DocumentProcessor()
        
        # Build the LangGraph Application
        self.workflow = self._build_graph()

    def _build_graph(self):
        """Construct the stateful Agentic RAG graph."""
        workflow = StateGraph(GraphState)

        # Define the nodes (actions the agent can take)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("rewrite_query", self.rewrite_query)

        # Build graph edges (the flow)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge: Decide whether to generate or rewrite based on grades
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query",
            }
        )
        workflow.add_edge("rewrite_query", "retrieve") # Loop back to retrieve with new query!
        workflow.add_edge("generate", END)

        # Compile
        return workflow.compile()

    # --- LANGGRAPH NODES ---

    def retrieve(self, state: GraphState):
        """Node: Retrieve documents from ChromaDB."""
        logger.info("---RETRIEVE---")
        question = state["question"]
        trajectory = state.get("trajectory", [])
        
        documents = self.retriever.invoke(question)
        trajectory.append("retrieve")
        
        return {"documents": documents, "trajectory": trajectory}

    def grade_documents(self, state: GraphState):
        """Node: Use Gemini to grade if the retrieved docs actually answer the question."""
        logger.info("---CHECK DOCUMENT RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        trajectory = state.get("trajectory", [])
        
        # Prompt to evaluate relevance
        prompt_template = f"""You are a grader assessing relevance of a retrieved document to a user question.
Here is the retrieved document: \n\n {{doc}} \n\n
Here is the user question: {question} \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        filtered_docs = []
        for d in documents:
            prompt = prompt_template.format(doc=d.page_content)
            
            # Using Structured Outputs (JSON Schema) to guarantee a yes/no response
            res = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GradeDocuments,
                    temperature=0.0
                )
            )
            
            # Very simple parsing
            score = "no"
            if "yes" in res.text.lower():
                score = "yes"
                
            if score == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                
        trajectory.append("grade_documents")
        return {"documents": filtered_docs, "trajectory": trajectory}

    def rewrite_query(self, state: GraphState):
        """Node: Rewrite the user's query if the retrieval failed."""
        logger.info("---REWRITE QUERY---")
        question = state["question"]
        retries = state.get("retries", 0)
        trajectory = state.get("trajectory", [])
        
        prompt = f"""Rewrite the question to improve vector database retrieval.

Rules:
- Use short keyword-style queries
- Replace vague words with specific system identifiers if possible
- Keep the query under 12 words

Original Question:
{question}:"""

        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        
        better_question = response.text.strip()
        trajectory.append(f"rewrite_query (from '{question}' to '{better_question}')")
        
        return {"question": better_question, "retries": retries + 1, "trajectory": trajectory}

    def generate(self, state: GraphState):
        """Node: Generate the final answer using the filtered documents."""
        logger.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        trajectory = state.get("trajectory", [])
        
        context_text = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context_text}
Answer:"""

        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        trajectory.append("generate")
        return {"generation": response.text, "trajectory": trajectory}

    # --- CONDITIONAL EDGES ---

    def decide_to_generate(self, state: GraphState):
        """Conditional Edge: Determines whether to generate an answer, or re-generate a query."""
        logger.info("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        retries = state.get("retries", 0)
        max_retries = 2 # Hardcoded maximum loops to prevent infinite recursion
        
        if not filtered_documents:
            # All documents were filtered out as garbage
            logger.info("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, REWRITE QUERY---")
            if retries >= max_retries:
                logger.info("---DECISION: MAX RETRIES REACHED, FORCING GENERATION---")
                return "generate"
            return "rewrite_query"
        else:
            # We have relevant documents, so generate answer!
            logger.info("---DECISION: GENERATE---")
            return "generate"

    # --- PUBLIC API METHODS ---

    async def ingest_text(self, text: str, source: str) -> int:
        docs = self.document_processor.process_text(text, source)
        if docs:
            await self.vectorstore.aadd_documents(documents=docs)
            logger.info(f"Ingested {len(docs)} chunks into ChromaDB.")
        return len(docs)

    async def query(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        """Execute the Agentic LangGraph workflow."""
        
        # Initialize the state dictionary for the graph
        inputs = {
            "question": question,
            "retries": 0,
            "trajectory": []
        }
        
        # Run the graph (invoke is sync, but we wrap it in async API)
        result_state = self.workflow.invoke(inputs, {"recursion_limit": 10})
        
        sources = list(set([doc.metadata.get("source", "unknown") for doc in result_state.get("documents", [])]))
        
        return {
            "answer": result_state.get("generation", "Failed to generate an answer."),
            "sources": sources,
            "agent_trajectory": result_state.get("trajectory", [])
        }