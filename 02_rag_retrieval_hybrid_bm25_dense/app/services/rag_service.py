import logging
import os
import pickle
from typing import Dict, Any, List
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document

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

class HybridRAGService:
    def __init__(self):
        # 1. Initialize official google-genai Client
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.embeddings = GeminiEmbeddings(self.client)
        
        # 2. Initialize Chroma Vector Database (Dense Retrieval)
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
        # 3. Initialize BM25 (Sparse Retrieval)
        self.bm25_retriever = None
        self._load_bm25_index()
        
        self.document_processor = DocumentProcessor()
        
        self.prompt_template = """You are an expert AI assistant.
Answer the question based ONLY on the following retrieved context.
If you cannot find the answer in the context, say "I don't know based on the provided context."

Question: {question}

Context:
{context}

Answer:"""

    def _load_bm25_index(self):
        """Load BM25 index from disk if it exists, so we don't lose sparse data on restart."""
        if os.path.exists(settings.BM25_PERSIST_PATH):
            try:
                with open(settings.BM25_PERSIST_PATH, 'rb') as f:
                    self.bm25_retriever = pickle.load(f)
                logger.info("Successfully loaded BM25 sparse index from disk.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")

    def _save_bm25_index(self):
        """Save BM25 index to disk."""
        if self.bm25_retriever:
            with open(settings.BM25_PERSIST_PATH, 'wb') as f:
                pickle.dump(self.bm25_retriever, f)

    async def ingest_text(self, text: str, source: str) -> int:
        """Process text and update BOTH dense (Chroma) and sparse (BM25) indexes."""
        docs = self.document_processor.process_text(text, source)
        
        if not docs:
            return 0

        # 1. Update Dense Index (Vector DB)
        await self.vectorstore.aadd_documents(documents=docs)
        
        # 2. Update Sparse Index (BM25 Keyword DB)
        if self.bm25_retriever is None:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
        else:
            # BM25Retriever doesn't support incremental adds natively,
            # so we extract existing docs, append new ones, and rebuild.
            existing_docs = self.bm25_retriever.docs
            all_docs = existing_docs + docs
            self.bm25_retriever = BM25Retriever.from_documents(all_docs)
            
        self._save_bm25_index()
        logger.info(f"Ingested {len(docs)} chunks into Dense & Sparse indexes. Source: {source}")
        
        return len(docs)

    async def query(self, question: str, strategy: str = "hybrid", k: int = 4) -> Dict[str, Any]:
        """Perform retrieval using the specified strategy, and generate an answer."""
        retrieved_docs: List[Document] = []
        
        # 1. Dense Strategy (Vector Semantic Search only)
        if strategy == "dense":
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            retrieved_docs = await retriever.ainvoke(question)
            
        # 2. Sparse Strategy (BM25 Exact Keyword Search only)
        elif strategy == "sparse":
            if not self.bm25_retriever:
                raise ValueError("Sparse index is empty. Please ingest documents first.")
            self.bm25_retriever.k = k
            retrieved_docs = await self.bm25_retriever.ainvoke(question)
            
        # 3. Hybrid Strategy (The Ensemble of both!)
        elif strategy == "hybrid":
            if not self.bm25_retriever:
                raise ValueError("Sparse index is empty. Please ingest documents first.")
            
            dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            self.bm25_retriever.k = k
            
            # Reciprocal Rank Fusion combines the results, weighting dense/sparse equally (0.5)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, self.bm25_retriever],
                weights=[0.5, 0.5]
            )
            # EnsembleRetriever doesn't fully support async ainvoke nicely without wrappers,
            # so we use synchronous invoke wrapped.
            retrieved_docs = ensemble_retriever.invoke(question)
            
            # Since ensemble merges lists, we ensure we only return top K overall
            retrieved_docs = retrieved_docs[:k]
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose 'dense', 'sparse', or 'hybrid'.")

        # Format context and send to Gemini Generation
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = list(set([doc.metadata.get("source", "unknown") for doc in retrieved_docs]))
        
        prompt = self.prompt_template.format(question=question, context=context_text)
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        return {
            "answer": response.text,
            "sources": sources,
            "strategy": strategy
        }