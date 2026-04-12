from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_text(self, text: str, source: str) -> List[Document]:
        """Convert raw text into LangChain Document chunks."""
        docs = [Document(page_content=text, metadata={"source": source})]
        return self.text_splitter.split_documents(docs)