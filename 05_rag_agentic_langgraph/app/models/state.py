from typing import List, TypedDict
from langchain_core.documents import Document

# This is the "Memory" of our Agent. 
# As it traverses the LangGraph, it carries this state with it.
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[Document]
    retries: int
    trajectory: List[str] # To track the steps the agent takes