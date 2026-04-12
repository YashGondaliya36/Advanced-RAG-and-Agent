from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process_text(self, text: str, source: str) -> List[Document]:
        docs = [Document(page_content=text, metadata={"source": source})]
        return self.text_splitter.split_documents(docs)