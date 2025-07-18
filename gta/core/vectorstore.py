"""
Pure vectorstore functions - Framework independent.
"""

from typing import List, Tuple, Any
from langchain_core.vectorstores import VectorStore


def search_documents(query: str, vectorstore: VectorStore, top_k: int = 5) -> List[Tuple[Any, float]]:
    """
    Pure document search function.
    
    Args:
        query: Search query text
        vectorstore: Vectorstore instance
        top_k: Number of documents to return
        
    Returns:
        List of (document, score) tuples
    """
    return vectorstore.similarity_search_with_score(query, k=top_k)


def format_search_results(results: List[Tuple[Any, float]]) -> str:
    """
    Format search results into context string.
    
    Args:
        results: List of (document, score) tuples
        
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant documents found."
    
    context_parts = []
    for i, (doc, score) in enumerate(results):
        context_parts.append(f"[Document {i+1}] (Score: {score:.3f})\n{doc.page_content}")
    
    return "\n\n".join(context_parts) 