"""
Simple RAG Agent Configuration Schema.
"""

from typing import Optional, List, Tuple
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


class SimpleRAGConfigSchema(TypedDict):
    """Complete Simple RAG configuration schema."""
    llm: LLMConfig
    vectorstore: VectorStoreConfig
    template_messages: Optional[List[Tuple[str, str]]]
    top_k: Optional[int]


# Default configurations
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

DEFAULT_SIMPLE_RAG_CONFIG = SimpleRAGConfigSchema(
    llm=DEFAULT_OLLAMA_LLM,
    vectorstore=DEFAULT_QDRANT_VECTORSTORE,
    top_k=5
) 