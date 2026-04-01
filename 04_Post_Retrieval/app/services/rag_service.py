import logging
from typing import Dict, Any, List
from google import genai
from google.genai import types

from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from app.core.config import settings
from app.utils.document_processor import DocumentProcessor

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

class PostRetrievalService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.embeddings = GeminiEmbeddings(self.client)
        
        # Initialize Vector Database
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
        # Initialize the FlashRank Reranker (Cross-Encoder)
        # This downloads a tiny, ultra-fast ~4MB ONNX model optimized for CPU
        self.compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
        
        # Initialize the 'Lost in the Middle' Reorderer
        self.reorderer = LongContextReorder()
        
        self.document_processor = DocumentProcessor()
        
        self.prompt_template = """You are an expert AI assistant.
Answer the question based ONLY on the following retrieved context.

Question: {question}

Context:
{context}

Answer:"""

    async def ingest_text(self, text: str, source: str) -> int:
        docs = self.document_processor.process_text(text, source)
        if docs:
            await self.vectorstore.aadd_documents(documents=docs)
            logger.info(f"Ingested {len(docs)} chunks into ChromaDB.")
        return len(docs)

    async def query(self, question: str, use_reranker: bool = True, use_reordering: bool = True, initial_k: int = 10, final_k: int = 3) -> Dict[str, Any]:
        optimization_steps = []
        
        # Step 1: Base Retrieval
        # We fetch a LARGE number of chunks (initial_k=10) from the "dumb" Bi-Encoder Vector DB
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": initial_k})
        
        retrieved_docs: List[Document] = []
        
        # Step 2: Contextual Compression (Cross-Encoder Reranking)
        if use_reranker:
            optimization_steps.append(f"FlashRank Reranking (Compressing top {initial_k} -> {final_k})")
            # We wrap the base retriever in a CompressionRetriever
            # The FlashRank model takes the User Query + Document simultaneously and re-scores them!
            self.compressor.top_n = final_k
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor, 
                base_retriever=base_retriever
            )
            # Fetch the massively improved, compressed, re-scored top 3 docs
            retrieved_docs = compression_retriever.invoke(question)
        else:
            # If no reranking, just fetch top K directly
            base_retriever.search_kwargs = {"k": final_k}
            retrieved_docs = await base_retriever.ainvoke(question)
            
        # Step 3: Long Context Reordering
        # Fixes the "Lost in the Middle" bias by putting highest scoring chunks at index 0 and -1
        if use_reordering and len(retrieved_docs) > 2:
            optimization_steps.append("Long Context Reordering (Lost-in-the-Middle fix)")
            retrieved_docs = self.reorderer.transform_documents(retrieved_docs)
            
        # Step 4: Generation
        context_text = "\n\n".join([f"[Document {i+1}]: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
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
            "optimization_steps_applied": optimization_steps
        }