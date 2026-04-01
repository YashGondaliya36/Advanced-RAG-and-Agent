import logging
import os
from typing import Dict, Any, List
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

from app.core.config import settings
from app.utils.document_processor import ParentChildProcessor

logger = logging.getLogger(__name__)

class GeminiEmbeddings:
    """Custom wrapper for Google GenAI embeddings."""
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

class ContextRAGService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.embeddings = GeminiEmbeddings(self.client)
        
        # 1. Initialize Vector Database for the tiny Child Chunks
        self.vectorstore = Chroma(
            collection_name="split_parents",
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
        # 2. Initialize a robust Key-Value store for the massive Parent Chunks
        os.makedirs(settings.BYTE_STORE_DIR, exist_ok=True)
        fs = LocalFileStore(settings.BYTE_STORE_DIR)
        self.store = create_kv_docstore(fs)
        
        # 3. Initialize the Parent-Child "Small-to-Big" Retriever
        self.processor = ParentChildProcessor()
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.processor.child_splitter,
            parent_splitter=self.processor.parent_splitter,
        )
        
        # Prompts
        self.qa_prompt_template = """You are an expert AI assistant.
Answer the question based ONLY on the following retrieved context.

Question: {question}

Context:
{context}

Answer:"""

        self.hyde_prompt_template = """Please write a short paragraph answering the following question. 
You do not need to be perfectly accurate, but act as if you are a document that contains the answer.
Do not include conversational filler, just the hypothetical answer text.

Question: {question}

Hypothetical Document:"""

    async def ingest_text(self, text: str, source: str) -> Dict[str, int]:
        """Process text using Small-to-Big hierarchy."""
        parent_docs, child_docs = self.processor.process_text(text, source)
        
        if not parent_docs:
            return {"parent_chunks": 0, "child_chunks": 0}

        # ParentDocumentRetriever automatically adds children to VectorDB 
        # and parents to the ByteStore when you call add_documents!
        self.retriever.add_documents(parent_docs, ids=[doc.metadata["doc_id"] for doc in parent_docs])
        
        logger.info(f"Ingested {len(parent_docs)} Parents mapped to {len(child_docs)} Children.")
        
        return {
            "parent_chunks": len(parent_docs),
            "child_chunks": len(child_docs)
        }

    def _generate_hyde_document(self, question: str) -> str:
        """Generate a hypothetical answer to solve the Asymmetric Search problem."""
        prompt = self.hyde_prompt_template.format(question=question)
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7) # Higher temp for hallucination
        )
        return response.text

    async def query(self, question: str, use_hyde: bool = False, k: int = 4) -> Dict[str, Any]:
        """Perform Small-to-Big retrieval, optionally using HyDE."""
        self.retriever.search_kwargs = {"k": k}
        
        search_query = question
        hyde_doc = None
        
        # Apply HyDE Strategy
        if use_hyde:
            logger.info("Generating Hypothetical Document (HyDE)...")
            hyde_doc = self._generate_hyde_document(question)
            logger.info(f"HyDE Document Generated: {hyde_doc[:100]}...")
            # We search the vector DB using the FAKE answer, not the user's question!
            search_query = hyde_doc
            
        # Execute Small-to-Big Retrieval
        # This will search `search_query` against the tiny child vectors, 
        # find the best matches, look up their parent IDs, and return the massive parent chunks!
        retrieved_parents = self.retriever.invoke(search_query)
        
        # Format context and send to Gemini Generation
        context_text = "\n\n".join([doc.page_content for doc in retrieved_parents])
        sources = list(set([doc.metadata.get("source", "unknown") for doc in retrieved_parents]))
        
        prompt = self.qa_prompt_template.format(question=question, context=context_text)
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        return {
            "answer": response.text,
            "sources": sources,
            "hypothetical_answer_generated": hyde_doc
        }