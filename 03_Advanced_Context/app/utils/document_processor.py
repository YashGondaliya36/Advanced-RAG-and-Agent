from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Tuple
import uuid

class ParentChildProcessor:
    def __init__(self):
        # The Parent Splitter creates massive chunks (rich context for the LLM)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        # The Child Splitter creates tiny chunks (high precision for the Vector DB)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

    def process_text(self, text: str, source: str) -> Tuple[List[Document], List[Document]]:
        """
        Creates a Hierarchical relationship.
        Returns (parent_docs, child_docs).
        """
        raw_doc = [Document(page_content=text, metadata={"source": source})]
        
        # 1. Create large Parent Chunks
        parent_docs = self.parent_splitter.split_documents(raw_doc)
        
        child_docs = []
        for parent in parent_docs:
            # Generate a unique ID for this specific parent chunk
            parent_id = str(uuid.uuid4())
            parent.metadata["doc_id"] = parent_id
            
            # 2. Break this specific Parent down into tiny Children
            children = self.child_splitter.split_documents([parent])
            
            # 3. Link every Child back to its Parent's ID
            for child in children:
                child.metadata["doc_id"] = parent_id
                child_docs.append(child)
                
        return parent_docs, child_docs