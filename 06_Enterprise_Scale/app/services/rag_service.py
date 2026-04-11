import logging
import time
from typing import Any, Dict

import numpy as np
from google import genai
from google.genai import types

from langchain_chroma import Chroma
from gptcache import cache
from gptcache.manager.scalar_data.base import Answer
from gptcache.similarity_evaluation import ExactMatchEvaluation

from app.core.config import settings
from app.utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

CACHE_SIMILARITY_THRESHOLD = 0.85


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
            emb = self.embeddings.embed_query(query)
            # Just return the raw list of floats. Faiss will handle it.
            return np.array(emb, dtype=np.float32)
            
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
        """
        from gptcache.manager import get_data_manager, CacheBase, VectorBase
        
        # Test embed to dynamically find exactly how many dimensions Gemini is returning
        test_emb = self.embeddings.embed_query("test")
        gemini_dim = len(test_emb)
        logger.info(f"Gemini Embedding Dimension Detected: {gemini_dim}")
        
        data_manager = get_data_manager(
            CacheBase("sqlite", sql_url=f"sqlite:///{settings.CACHE_PERSIST_DIR}"),
            VectorBase("faiss", dimension=gemini_dim, top_k=5),
            max_size=1000,
        )

        # Similarity for HTTP path is computed in query(); this satisfies cache.init API.
        cache.init(
            embedding_func=self.gptcache_embedding_func,
            data_manager=data_manager,
            similarity_evaluation=ExactMatchEvaluation(),
        )
        logger.info("Semantic Cache Initialized using Gemini Embeddings.")

    async def ingest_text(self, text: str, source: str) -> int:
        docs = self.document_processor.process_text(text, source)
        if docs:
            await self.vectorstore.aadd_documents(documents=docs)
            logger.info(f"Ingested {len(docs)} chunks into ChromaDB.")
        return len(docs)

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, cache_vec: np.ndarray) -> float:
        query_arr = np.asarray(query_vec, dtype=np.float64).ravel()
        cache_arr = np.asarray(cache_vec, dtype=np.float64).ravel()
        if query_arr.size == 0 or cache_arr.size == 0 or query_arr.size != cache_arr.size:
            return -1.0
        query_norm = float(np.linalg.norm(query_arr))
        cache_norm = float(np.linalg.norm(cache_arr))
        if query_norm == 0.0 or cache_norm == 0.0:
            return -1.0
        query_unit = query_arr / query_norm
        cache_unit = cache_arr / cache_norm
        return float(np.dot(query_unit, cache_unit))

    @staticmethod
    def _first_answer_text(cache_row) -> str:
        first = cache_row.answers[0]
        if isinstance(first, Answer):
            return first.answer
        return first

    async def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        
        # --- SEMANTIC CACHE CHECK ---
        if use_cache:
            # Embed the question using Gemini and convert to Hex
            embedded_question = self.gptcache_embedding_func(question)
            
            logger.info("🔍 Searching SQLite Cache for similar vectors...")
            
            # FAISS returns [(distance, row_id), ...]; embeddings live in scalar storage.
            search_rows = cache.data_manager.search(embedded_question, top_k=5)
            if search_rows is None:
                search_rows = []

            if search_rows:
                logger.info(
                    "📁 Found %s potential match(es). Scoring cosine similarity vs stored embeddings.",
                    len(search_rows),
                )
                for search_row in search_rows:
                    cache_row = cache.data_manager.get_scalar_data(search_row)
                    if cache_row is None or cache_row.embedding_data is None:
                        logger.info("❌ ---CACHE MISS--- Row missing or has no embedding_data.")
                        continue

                    score_val = self._cosine_similarity(
                        embedded_question, cache_row.embedding_data
                    )
                    logger.info(
                        "📊 Semantic cache score: %.4f (threshold: %.2f)",
                        score_val,
                        CACHE_SIMILARITY_THRESHOLD,
                    )
                    if score_val >= CACHE_SIMILARITY_THRESHOLD:
                        logger.info(
                            "✅ ---CACHE HIT--- Similarity >= %.2f. Bypassing LLM.",
                            CACHE_SIMILARITY_THRESHOLD,
                        )
                        cache.data_manager.hit_cache_callback(search_row)
                        return {
                            "answer": self._first_answer_text(cache_row),
                            "sources": ["semantic_cache"],
                            "cache_hit": True,
                            "latency_ms": round((time.time() - start_time) * 1000, 2),
                        }
                    logger.info(
                        "❌ ---CACHE MISS (THRESHOLD)--- Neighbor below %.2f.",
                        CACHE_SIMILARITY_THRESHOLD,
                    )
            else:
                logger.info("❌ ---CACHE MISS (EMPTY)--- No nearby vectors in FAISS.")

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