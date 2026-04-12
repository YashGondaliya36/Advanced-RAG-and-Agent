import logging
from typing import Dict, Any
from google import genai
from google.genai import types
from langchain_chroma import Chroma

from app.core.config import settings
from app.utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class GeminiEmbeddings:
    """Custom wrapper for Google GenAI embeddings to work with Langchain Chroma."""
    def __init__(self, client: genai.Client):
        self.client = client
        self.model = "gemini-embedding-001"
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return [emb.values for emb in response.embeddings]
        
    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return response.embeddings[0].values

class NaiveRAGService:
    def __init__(self):
        # 1. Initialize official google-genai Client
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        
        # 2. Wrapper for Chroma to use the official genai client
        self.embeddings = GeminiEmbeddings(self.client)
        
        # 3. Initialize Chroma Vector Database
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
        self.document_processor = DocumentProcessor()
        
        # 4. Define the core Naive RAG Prompt
        self.prompt_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise and direct.

Question: {question} 

Context: {context} 

Answer:"""

    async def ingest_text(self, text: str, source: str) -> int:
        """Process text, chunk it, and store dense vectors in ChromaDB."""
        # Chunking
        docs = self.document_processor.process_text(text, source)
        
        if docs:
            # Storing Embeddings in Vector DB
            await self.vectorstore.aadd_documents(documents=docs)
            logger.info(f"Ingested {len(docs)} chunks into ChromaDB from source: {source}.")
        
        return len(docs)

    async def query(self, question: str) -> Dict[str, Any]:
        """Perform semantic retrieval and generate an answer using Gemini."""
        # 1. Setup the Retriever (K=4 chunks)
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # 2. Retrieve chunks based on semantic similarity
        retrieved_docs = await retriever.ainvoke(question)
        
        # Extract context text and track sources
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = list(set([doc.metadata.get("source", "unknown") for doc in retrieved_docs]))
        
        # 3. Build prompt and execute generation using official google-genai
        prompt = self.prompt_template.format(question=question, context=context_text)
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        
        return {
            "answer": response.text,
            "sources": sources
        }
