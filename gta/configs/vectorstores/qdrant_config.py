"""
Qdrant-specific configuration schema.
"""

from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict


class QdrantConfig(TypedDict):
    """
    Qdrant vectorstore configuration schema.
    
    Used for runtime configuration of Qdrant client and embedding models.
    """
    
    # Qdrant server configuration
    url: Optional[str]              # Qdrant server URL (default: "http://localhost:6333")
    api_key: Optional[str]          # Qdrant API key for authentication
    collection_name: str            # Collection name for vectors
    
    # Vector configuration
    vector_size: Optional[int]      # Vector dimensions (auto-detected from embedding model)
    distance: Optional[str]         # Distance metric: "cosine", "dot", "euclidean" (default: "cosine")
    
    # Embedding model configuration
    embedding_model: str            # Embedding model name (e.g., "mxbai-embed-large")
    embedding_base_url: Optional[str]  # Ollama/OpenAI base URL for embeddings
    embedding_api_key: Optional[str]   # API key for embedding service
    embedding_provider: Optional[str]  # "ollama" or "openai" (default: "ollama")
    
    # Operation settings
    batch_size: Optional[int]       # Batch size for bulk operations (default: 100)
    top_k: Optional[int]           # Number of results to return (default: 5)
    score_threshold: Optional[float] # Minimum similarity score (default: 0.0)
    
    # Qdrant-specific options
    prefer_grpc: Optional[bool]     # Use gRPC instead of HTTP (default: False)
    timeout: Optional[float]        # Request timeout in seconds (default: 60.0)
    
    # Metadata filtering
    metadata_filter: Optional[Dict[str, Any]]  # Filter conditions for search
    
    # Additional model kwargs
    model_kwargs: Optional[Dict[str, Any]]     # Additional embedding model parameters 