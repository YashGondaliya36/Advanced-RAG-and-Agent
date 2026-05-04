import logging
import os
import uuid
import mimetypes
from typing import Dict, Any, List
from google import genai
from google.genai import types
import chromadb
from PIL import Image
import io

# We import SentenceTransformer for the open-source CLIP model
# This runs locally and embeds both text and images into the same 512-dimensional space.
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

class HybridMultimodalRAGService:
    """
    Hybrid Architecture:
    - Embeddings: Local Open-Source (CLIP via SentenceTransformers) [NO API LIMITS]
    - Storage: ChromaDB
    - Generation: Google Gemini 2.5 Flash API (Passed raw retrieved images)
    """
    def __init__(self):
        # 1. Initialize official google-genai Client ONLY for Generation
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.generation_model = "gemini-2.5-flash"
        
        # 2. Initialize Local Open-Source Embeddings (CLIP Model)
        # Using clip-ViT-B-32 as it is incredibly fast and standard for image/text hybrid search
        logger.info("Loading Local Open-Source CLIP Model...")
        self.local_embedder = SentenceTransformer('clip-ViT-B-32')
        logger.info("Local Model Loaded successfully.")
        
        # 3. Initialize Chroma Vector Database (Hybrid Collection)
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        
        # We create a separate collection so it doesn't conflict with the Gemini-embedded collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="hybrid_multimodal_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Directory to store raw files
        self.asset_dir = "./sample_data/hybrid_assets"
        os.makedirs(self.asset_dir, exist_ok=True)
        
        self.prompt_template = """You are an advanced analytical assistant. 
Review the provided context (which may be text or images) and answer the user's question.

Question: {question}
"""

    def _get_mime_type(self, filename: str) -> str:
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"

    async def ingest_file(self, file_bytes: bytes, filename: str) -> int:
        """Process an image file, embed it LOCALLY, and store it."""
        doc_id = str(uuid.uuid4())
        mime_type = self._get_mime_type(filename)
        
        # Save raw asset to disk
        file_ext = os.path.splitext(filename)[1]
        asset_path = os.path.join(self.asset_dir, f"{doc_id}{file_ext}")
        with open(asset_path, "wb") as f:
            f.write(file_bytes)
            
        logger.info(f"Saved asset to {asset_path}")

        # 1. Generate Embeddings using LOCAL OPEN-SOURCE MODEL
        if mime_type.startswith("image/"):
            # Embed Image Locally
            image = Image.open(io.BytesIO(file_bytes))
            # encode() directly accepts PIL images for CLIP models
            embedding_values = self.local_embedder.encode(image).tolist()
            
            # 2. Store in ChromaDB
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding_values],
                metadatas=[{"filename": filename, "mime_type": mime_type, "asset_path": asset_path}],
                documents=["[IMAGE FILE]"] 
            )
            logger.info(f"Ingested 1 image using Local CLIP: {filename}.")
            return 1
        elif mime_type.startswith("text/"):
            text_content = file_bytes.decode('utf-8')
            return await self.ingest_text(text_content, filename)
        else:
            # We skip PDF parsing logic here for brevity, but you would use PyMuPDF 
            # to iterate pages and save each as an image, then call this function recursively.
            logger.warning(f"File type {mime_type} not natively handled in basic hybrid ingest yet.")
            return 0

    async def ingest_text(self, text: str, source: str) -> int:
        """Ingest raw text using LOCAL OPEN-SOURCE MODEL."""
        doc_id = str(uuid.uuid4())
        
        # 1. Generate Embedding Locally
        embedding_values = self.local_embedder.encode(text).tolist()
        
        # 2. Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding_values],
            metadatas=[{"filename": source, "mime_type": "text/plain"}],
            documents=[text]
        )
        logger.info(f"Ingested text using Local CLIP from: {source}.")
        return 1

    async def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the vector db using Local Embeddings, generate answer using Gemini Vision API."""
        # 1. Embed the user's question LOCALLY
        query_embedding = self.local_embedder.encode(question).tolist()
        
        # 2. Retrieve nearest neighbors from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents"]
        )
        
        if not results['ids'] or len(results['ids'][0]) == 0:
            return {"answer": "No relevant context found.", "sources": []}
            
        retrieved_metadatas = results['metadatas'][0]
        retrieved_documents = results['documents'][0]
        
        # 3. Prepare the contents for generation using Gemini API
        contents = [self.prompt_template.format(question=question)]
        sources = []
        
        for i, meta in enumerate(retrieved_metadatas):
            mime_type = meta.get("mime_type", "text/plain")
            source_name = meta.get("filename", "unknown")
            sources.append({"source": source_name, "type": mime_type})
            
            if mime_type.startswith("text/"):
                contents.append(f"Context Document {i+1}:\n{retrieved_documents[i]}\n")
            else:
                asset_path = meta.get("asset_path")
                if asset_path and os.path.exists(asset_path):
                    with open(asset_path, "rb") as f:
                        file_bytes = f.read()
                        
                    contents.append(f"Context Document {i+1} ({source_name}):")
                    contents.append(
                        types.Part.from_bytes(
                            data=file_bytes,
                            mime_type=mime_type
                        )
                    )
                    
        # 4. Generate the final answer using Cloud Gemini API
        logger.info(f"Generating answer with {len(contents)} parts via Gemini API.")
        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        
        return {
            "answer": response.text,
            "sources": sources
        }


