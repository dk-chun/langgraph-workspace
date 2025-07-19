"""
Multi-VectorStore RAG Agent State Definition.
"""

from typing import Annotated, List, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_search_scores(existing: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    """
    Merge search scores from multiple vectorstore nodes.
    
    Args:
        existing: Existing search scores dictionary
        new: New search scores to merge
        
    Returns:
        Merged search scores dictionary
    """
    merged = existing.copy()
    merged.update(new)
    return merged


class MultiRAGState(BaseModel):
    """
    Multi-VectorStore RAG state with parallel search results.
    """
    # Input query
    question: str = Field(default="", description="Extracted question from user message")
    
    # Individual vectorstore results
    vs1_documents: List[Any] = Field(default_factory=list, description="VectorStore 1 retrieved documents")
    vs2_documents: List[Any] = Field(default_factory=list, description="VectorStore 2 retrieved documents")
    vs3_documents: List[Any] = Field(default_factory=list, description="VectorStore 3 retrieved documents")
    
    # Merged results
    merged_documents: List[Any] = Field(default_factory=list, description="Merged and ranked documents")
    final_context: str = Field(default="", description="Final formatted context from merged results")
    
    # Search metadata - using reducer to handle concurrent updates from parallel nodes
    search_scores: Annotated[Dict[str, float], merge_search_scores] = Field(
        default_factory=dict, 
        description="Search quality scores per vectorstore"
    )
    merge_strategy: str = Field(default="simple", description="Strategy used for merging results")
    
    # Conversation messages
    messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list, 
        description="Conversation messages"
    ) 