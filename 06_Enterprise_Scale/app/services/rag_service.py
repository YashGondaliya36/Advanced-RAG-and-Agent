import logging
import time
from typing import Dict, Any, List
from google import genai
from google.genai import types

from langchain_chroma import Chroma
from gptcache import cache
from gptcache.manager import get_data_manager
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

from app.core.config import settings
from app.utils.document_processor import DocumentProcessor

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

class EnterpriseRAGService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.embeddings = GeminiEmbeddings(self.client)
        
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.document_processor = DocumentProcessor()
        
        # Initialize Semantic Cache
        self._init_semantic_cache()
        
    def _init_semantic_cache(self):
        """
        Initializes GPTCache.
        It uses a local ONNX model to embed the user's query, 
        and SQLite to store the generated answer.
        """
        onnx = Onnx()
        data_manager = get_data_manager(
            data_path=settings.CACHE_PERSIST_DIR, 
            max_size=1000,
            # We store the question vector and the cached text answer
        )
        cache.init(
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            # If a new query is 90% semantically similar to an old query, return the cache!
            similarity_threshold=0.9 
        )
        logger.info("Semantic Cache Initialized.")

    async def ingest_text(self, text: str, source: str) -> int:
        docs = self.document_processor.process_text(text, source)
        if docs:
            await self.vectorstore.aadd_documents(documents=docs)
            logger.info(f"Ingested {len(docs)} chunks into ChromaDB.")
        return len(docs)

    async def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        
        # --- SEMANTIC CACHE CHECK ---
        if use_cache:
            # We search the cache for a similar question
            cache_result = cache.get(question)
            if cache_result:
                logger.info("---CACHE HIT!--- Bypassing LLM completely.")
                return {
                    "answer": cache_result,
                    "sources": ["semantic_cache"],
                    "cache_hit": True,
                    "latency_ms": round((time.time() - start_time) * 1000, 2)
                }

        # --- NORMAL RAG WORKFLOW (CACHE MISS) ---
        logger.info("---CACHE MISS. Executing RAG workflow.---")
        retrieved_docs = await self.retriever.ainvoke(question)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = list(set([doc.metadata.get("source", "unknown") for doc in retrieved_docs]))
        
        prompt = f"""You are an expert AI assistant. Answer the question based ONLY on the context.
Question: {question}
Context: {context_text}
Answer:"""

        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        final_answer = response.text
        
        # --- SAVE TO CACHE FOR FUTURE QUERIES ---
        if use_cache:
            cache.put(question, final_answer)
            logger.info("Answer saved to Semantic Cache.")
            
        return {
            "answer": final_answer,
            "sources": sources,
            "cache_hit": False,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }