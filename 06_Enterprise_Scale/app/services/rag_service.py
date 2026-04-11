import logging
import time
from typing import Dict, Any, List
from google import genai
from google.genai import types

from langchain_chroma import Chroma
from gptcache import cache
from gptcache.manager import get_data_manager

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
        
        # We define an embedding function matching GPTCache's expected signature
        def gptcache_embedding_func(query: str):
            import numpy as np
            emb = self.embeddings.embed_query(query)
            # GPTCache's SQLite backend struggles with raw bytes, so we convert to a hex string
            return np.array(emb, dtype=float).tobytes().hex()
            
        self.gptcache_embedding_func = gptcache_embedding_func
        
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
        Initializes GPTCache to use SQLite.
        We will use our own Gemini embeddings function to bypass ONNX completely.
        """
        data_manager = get_data_manager(
            data_path=settings.CACHE_PERSIST_DIR, 
            max_size=1000
        )
        
        # We define a custom evaluation function so we can specify the threshold safely
        def evaluate_func(query_data, cache_data):
            import numpy as np
            # Convert hex strings back to numpy arrays for math
            q_emb = np.frombuffer(bytes.fromhex(query_data), dtype=float)
            c_emb = np.frombuffer(bytes.fromhex(cache_data), dtype=float)
            score = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
            logger.info(f"Semantic Cache Comparison Score: {score}")
            # Dropped threshold to 0.85 to allow for more linguistic variation!
            return score > 0.85

        cache.init(
            embedding_func=self.gptcache_embedding_func,
            data_manager=data_manager,
            similarity_evaluation=evaluate_func
        )
        logger.info("Semantic Cache Initialized using Gemini Embeddings.")

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
            # Embed the question using Gemini and convert to bytes
            embedded_question = self.gptcache_embedding_func(question)
            
            # We search the cache for a similar question using the embedded vector bytes
            cache_data_list = cache.data_manager.search(
                embedded_question
            )
            
            if cache_data_list:
                # We need to manually run the similarity evaluation on the results
                for cache_data in cache_data_list:
                    if cache.similarity_evaluation(embedded_question, cache_data[1]):
                        logger.info("---CACHE HIT!--- Bypassing LLM completely.")
                        best_match_answer = cache.data_manager.get(cache_data[0]).answer
                        return {
                            "answer": best_match_answer,
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
            cache.data_manager.save(
                question,
                final_answer,
                self.gptcache_embedding_func(question)
            )
            logger.info("Answer saved to Semantic Cache.")
            
        return {
            "answer": final_answer,
            "sources": sources,
            "cache_hit": False,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }