"""
Qdrant vectorstore state definition.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class QdrantState(BaseModel):
    """
    State for Qdrant vectorstore operations.
    
    Supports document insertion, similarity search, and metadata filtering.
    """
    
    # Operation type
    operation: str = Field(default="search", description="Operation type: 'insert', 'search', 'delete', 'update'")
    
    # Document operations
    documents: List[Document] = Field(default_factory=list, description="Documents to insert/update")
    document_ids: List[str] = Field(default_factory=list, description="Document IDs for delete operations")
    
    # Search operations
    query: Optional[str] = Field(default=None, description="Search query text")
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results with scores")
    
    # Metadata and filtering
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for operations")
    filter_conditions: Optional[Dict[str, Any]] = Field(default=None, description="Filter conditions for search")
    
    # Results and status
    success: bool = Field(default=True, description="Operation success status")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
    count: int = Field(default=0, description="Number of documents processed or found")
    
    # Additional context
    collection_info: Dict[str, Any] = Field(default_factory=dict, description="Collection information")
    operation_stats: Dict[str, Any] = Field(default_factory=dict, description="Operation statistics") 