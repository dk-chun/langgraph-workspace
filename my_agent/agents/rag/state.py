"""
State definition for the RAG agent.
"""

from typing import List, Optional, Dict, Any
from my_agent.core.base_state import BaseState


class RAGState(BaseState):
    """
    State for the RAG agent graph.
    
    Extends BaseState with RAG-specific state variables.
    """
    # Query processing
    query: Optional[str] = None
    
    # Document retrieval
    retrieved_docs: List[Dict[str, Any]] = []
    context: Optional[str] = None
    
    # Generation
    response: Optional[str] = None
    
    # Metadata
    retrieval_score: Optional[float] = None
    num_docs_retrieved: int = 0 