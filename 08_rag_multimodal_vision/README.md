# Phase 08: Multi-Modal RAG (Native Vision Embeddings)

This module demonstrates how to implement **Multi-Modal RAG** using the latest native multimodal embedding capabilities of Google Gemini (`gemini-embedding-2`). 

Instead of relying on workarounds (like using a vision model to summarize an image into text and embedding the text), this architecture directly embeds raw images, PDFs, and text into the **same latent vector space**. This allows users to ask a text-based question and retrieve the exact raw image or PDF page that answers it.

## Key Concepts
1. **Direct Image/PDF Ingestion:** Using `gemini-embedding-2` to convert `bytes` of images and PDFs into 768-dimensional vectors.
2. **Unified Vector Storage:** Storing text vectors and image vectors together in ChromaDB.
3. **Cross-Modal Retrieval:** Text queries seamlessly matching against visual embedding representations.
4. **Multi-Modal Generation:** Passing the retrieved raw assets directly to `gemini-2.5-flash` so it can "see" the charts/tables while answering.

## Setup
1. Create a virtual environment and activate it.
2. `pip install -r requirements.txt`
3. Ensure your `GOOGLE_API_KEY` is in the `.env` file at the root.

## Running the API
```bash
uvicorn app.main:app --reload
```

## Endpoints
- `POST /api/v1/ingest/file`: Upload a PDF or Image (PNG, JPEG) to be embedded and stored natively.
- `POST /api/v1/ingest/text`: Upload raw text.
- `POST /api/v1/query`: Ask a text question. The system will retrieve the best matching assets (including images) and generate an answer using Gemini Vision.


