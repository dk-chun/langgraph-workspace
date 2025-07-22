"""
Multi-RAG Agent Configuration Schema.
"""

from typing import List, Optional, Tuple
from typing_extensions import TypedDict


class LLMConfig(TypedDict):
    """LLM provider configuration."""
    provider: str  # "ollama", "openai", "anthropic"
    model: str
    temperature: Optional[float]
    base_url: Optional[str]  # For Ollama
    api_key: Optional[str]   # For OpenAI/Anthropic


class EmbeddingConfig(TypedDict):
    """Embedding provider configuration."""
    provider: str  # "ollama", "openai", "huggingface"
    model: str
    base_url: Optional[str]      # For Ollama
    api_key: Optional[str]       # For OpenAI
    model_kwargs: Optional[dict] # For HuggingFace
    encode_kwargs: Optional[dict] # For HuggingFace


class VectorStoreConfig(TypedDict):
    """VectorStore provider configuration."""
    provider: str  # "qdrant", "chroma", "pinecone", "faiss"
    collection_name: str
    embedding: EmbeddingConfig
    
    # Provider-specific fields
    url: Optional[str]           # Qdrant
    path: Optional[str]          # Chroma, FAISS
    index_name: Optional[str]    # Pinecone
    api_key: Optional[str]       # Pinecone
    environment: Optional[str]   # Pinecone
    index_path: Optional[str]    # FAISS


class MultiRAGConfigSchema(TypedDict):
    """Complete Multi-RAG configuration schema."""
    llm: LLMConfig
    vectorstores: List[VectorStoreConfig]  # Exactly 3 vectorstores
    template_messages: Optional[List[Tuple[str, str]]]
    top_k_per_store: Optional[int]
    final_top_k: Optional[int]
    merge_strategy: Optional[str]


# Default configurations for common setups
DEFAULT_OLLAMA_LLM = LLMConfig(
    provider="ollama",
    model="qwen3:0.6b",
    temperature=0.3,
    base_url="http://localhost:11434"
)

DEFAULT_OLLAMA_EMBEDDING = EmbeddingConfig(
    provider="ollama", 
    model="bge-m3:latest",
    base_url="http://localhost:11434"
)

DEFAULT_QDRANT_VECTORSTORE = VectorStoreConfig(
    provider="qdrant",
    collection_name="test_langgraph",
    url="http://localhost:6333",
    embedding=DEFAULT_OLLAMA_EMBEDDING
)

DEFAULT_MULTI_RAG_CONFIG = MultiRAGConfigSchema(
    llm=DEFAULT_OLLAMA_LLM,
    vectorstores=[
        {**DEFAULT_QDRANT_VECTORSTORE, "collection_name": "test_langgraph"},
        {**DEFAULT_QDRANT_VECTORSTORE, "collection_name": "test_langgraph"}, 
        {**DEFAULT_QDRANT_VECTORSTORE, "collection_name": "test_langgraph"}
    ],
    top_k_per_store=3,
    final_top_k=5,
    merge_strategy="simple"
) 