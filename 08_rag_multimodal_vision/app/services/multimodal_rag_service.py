import logging
import os
import uuid
import mimetypes
from typing import Dict, Any, List
from google import genai
from google.genai import types
import chromadb

from app.core.config import settings

logger = logging.getLogger(__name__)

class MultimodalRAGService:
    def __init__(self):
        # 1. Initialize official google-genai Client
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.embedding_model = "gemini-embedding-2"
        self.generation_model = "gemini-2.5-flash"
        
        # 2. Initialize Chroma Vector Database (Native)
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        
        # Using cosine similarity for embeddings
        self.collection = self.chroma_client.get_or_create_collection(
            name="multimodal_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Directory to store raw files for later retrieval
        self.asset_dir = "./sample_data/assets"
        os.makedirs(self.asset_dir, exist_ok=True)
        
        # 3. Define the Prompt
        self.prompt_template = """You are a helpful assistant capable of answering questions based on the provided text, images, or documents.
Use the provided retrieved context (which may include images, documents, or text) to answer the user's question.
If the answer is not in the context, say so. Do not guess.

Question: {question}
"""

    def _get_mime_type(self, filename: str) -> str:
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"

    async def ingest_file(self, file_bytes: bytes, filename: str) -> int:
        """Process a file (image, pdf, or text), embed it, and store it."""
        doc_id = str(uuid.uuid4())
        mime_type = self._get_mime_type(filename)
        
        # Save raw asset to disk so we can retrieve it during generation
        file_ext = os.path.splitext(filename)[1]
        asset_path = os.path.join(self.asset_dir, f"{doc_id}{file_ext}")
        with open(asset_path, "wb") as f:
            f.write(file_bytes)
            
        logger.info(f"Saved asset to {asset_path}")

        # 1. Generate Embeddings using gemini-embedding-2
        if mime_type.startswith("text/"):
            # Embed text
            text_content = file_bytes.decode('utf-8')
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text_content
            )
        else:
            # Embed image, pdf, etc.
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=[
                    types.Part.from_bytes(
                        data=file_bytes,
                        mime_type=mime_type
                    )
                ]
            )
            
        embedding_values = response.embeddings[0].values
        
        # 2. Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding_values],
            metadatas=[{"filename": filename, "mime_type": mime_type, "asset_path": asset_path}],
            # For non-text data, we store the path in metadata. We'll leave documents empty or put filename.
            documents=[filename] 
        )
        
        logger.info(f"Ingested 1 multimodal document: {filename}.")
        return 1

    async def ingest_text(self, text: str, source: str) -> int:
        """Ingest raw text."""
        doc_id = str(uuid.uuid4())
        
        # 1. Generate Embedding
        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )
        embedding_values = response.embeddings[0].values
        
        # 2. Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding_values],
            metadatas=[{"filename": source, "mime_type": "text/plain"}],
            documents=[text]
        )
        return 1

    async def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the vector db with text, retrieve multimodal chunks, and generate an answer."""
        # 1. Embed the user's question
        embed_response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=question
        )
        query_embedding = embed_response.embeddings[0].values
        
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
        
        # 3. Prepare the contents for generation
        contents = [self.prompt_template.format(question=question)]
        sources = []
        
        # Add retrieved assets to the Gemini generation call
        for i, meta in enumerate(retrieved_metadatas):
            mime_type = meta.get("mime_type", "text/plain")
            source_name = meta.get("filename", "unknown")
            sources.append({"source": source_name, "type": mime_type})
            
            if mime_type.startswith("text/"):
                # Add text content
                contents.append(f"Context Document {i+1}:\n{retrieved_documents[i]}\n")
            else:
                # Add file/image content from disk
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
                    
        # 4. Generate the final answer using gemini-2.5-flash
        logger.info(f"Generating answer with {len(contents)} parts.")
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

