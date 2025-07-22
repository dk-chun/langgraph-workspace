"""
Provider factories using Registry pattern for dynamic instantiation.
"""

from typing import Dict, Callable, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings


# ===== LLM FACTORIES =====

def create_ollama_llm(config: dict) -> BaseChatModel:
    """Create Ollama LLM instance."""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config["model"],
        temperature=config.get("temperature", 0.7),
        base_url=config.get("base_url", "http://localhost:11434")
    )


def create_openai_llm(config: dict) -> BaseChatModel:
    """Create OpenAI LLM instance."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=config["model"],
        temperature=config.get("temperature", 0.7),
        api_key=config.get("api_key")
    )


def create_anthropic_llm(config: dict) -> BaseChatModel:
    """Create Anthropic LLM instance."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=config["model"],
        temperature=config.get("temperature", 0.7),
        api_key=config.get("api_key")
    )


# LLM Registry
LLM_REGISTRY: Dict[str, Callable[[dict], BaseChatModel]] = {
    "ollama": create_ollama_llm,
    "openai": create_openai_llm,
    "anthropic": create_anthropic_llm,
}


# ===== EMBEDDING FACTORIES =====

def create_ollama_embedding(config: dict) -> Embeddings:
    """Create Ollama embedding instance."""
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=config["model"],
        base_url=config.get("base_url", "http://localhost:11434")
    )


def create_openai_embedding(config: dict) -> Embeddings:
    """Create OpenAI embedding instance."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=config["model"],
        api_key=config.get("api_key")
    )


def create_huggingface_embedding(config: dict) -> Embeddings:
    """Create HuggingFace embedding instance."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config["model"],
        model_kwargs=config.get("model_kwargs", {}),
        encode_kwargs=config.get("encode_kwargs", {})
    )


# Embedding Registry
EMBEDDING_REGISTRY: Dict[str, Callable[[dict], Embeddings]] = {
    "ollama": create_ollama_embedding,
    "openai": create_openai_embedding,
    "huggingface": create_huggingface_embedding,
}


# ===== VECTORSTORE FACTORIES =====

def create_qdrant_vectorstore(config: dict, embedding: Embeddings) -> VectorStore:
    """Create Qdrant VectorStore instance."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    
    client = QdrantClient(url=config["url"])
    return QdrantVectorStore(
        client=client,
        collection_name=config["collection_name"],
        embedding=embedding
    )


def create_chroma_vectorstore(config: dict, embedding: Embeddings) -> VectorStore:
    """Create Chroma VectorStore instance."""
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=config["collection_name"],
        embedding_function=embedding,
        persist_directory=config.get("path", "./chroma_db")
    )


def create_pinecone_vectorstore(config: dict, embedding: Embeddings) -> VectorStore:
    """Create Pinecone VectorStore instance."""
    from langchain_pinecone import PineconeVectorStore
    return PineconeVectorStore(
        index_name=config["index_name"],
        embedding=embedding,
        pinecone_api_key=config["api_key"],
        environment=config["environment"]
    )


def create_faiss_vectorstore(config: dict, embedding: Embeddings) -> VectorStore:
    """Create FAISS VectorStore instance."""
    from langchain_community.vectorstores import FAISS
    
    # FAISS는 기존 인덱스를 로드하거나 새로 생성
    if config.get("index_path"):
        return FAISS.load_local(config["index_path"], embedding)
    else:
        # 빈 FAISS 인덱스 생성 (문서 추가 필요)
        return FAISS.from_texts([""], embedding)


# VectorStore Registry  
VECTORSTORE_REGISTRY: Dict[str, Callable[[dict, Embeddings], VectorStore]] = {
    "qdrant": create_qdrant_vectorstore,
    "chroma": create_chroma_vectorstore,
    "pinecone": create_pinecone_vectorstore,
    "faiss": create_faiss_vectorstore,
}


# ===== MAIN FACTORY FUNCTIONS =====

def create_llm(config: dict) -> BaseChatModel:
    """
    Create LLM instance based on provider configuration.
    
    Args:
        config: Configuration dict with 'provider' key and provider-specific settings
        
    Returns:
        BaseChatModel instance
        
    Raises:
        ValueError: If provider is not supported
    """
    provider = config["provider"]
    if provider not in LLM_REGISTRY:
        available = ", ".join(LLM_REGISTRY.keys())
        raise ValueError(f"Unsupported LLM provider: {provider}. Available: {available}")
    
    return LLM_REGISTRY[provider](config)


def create_vectorstore(config: dict) -> VectorStore:
    """
    Create VectorStore instance based on provider configuration.
    
    Args:
        config: Configuration dict with 'provider' key, 'embedding' config, and provider-specific settings
        
    Returns:
        VectorStore instance
        
    Raises:
        ValueError: If provider is not supported
    """
    # Create embedding first
    embedding_config = config["embedding"]
    embedding_provider = embedding_config["provider"]
    
    if embedding_provider not in EMBEDDING_REGISTRY:
        available = ", ".join(EMBEDDING_REGISTRY.keys())
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}. Available: {available}")
    
    embedding = EMBEDDING_REGISTRY[embedding_provider](embedding_config)
    
    # Create vectorstore
    vs_provider = config["provider"]
    if vs_provider not in VECTORSTORE_REGISTRY:
        available = ", ".join(VECTORSTORE_REGISTRY.keys())
        raise ValueError(f"Unsupported vectorstore provider: {vs_provider}. Available: {available}")
    
    return VECTORSTORE_REGISTRY[vs_provider](config, embedding)


# ===== REGISTRY EXTENSION HELPERS =====

def register_llm_provider(name: str, factory_func: Callable[[dict], BaseChatModel]):
    """Register a new LLM provider."""
    LLM_REGISTRY[name] = factory_func


def register_embedding_provider(name: str, factory_func: Callable[[dict], Embeddings]):
    """Register a new embedding provider."""
    EMBEDDING_REGISTRY[name] = factory_func


def register_vectorstore_provider(name: str, factory_func: Callable[[dict, Embeddings], VectorStore]):
    """Register a new vectorstore provider."""
    VECTORSTORE_REGISTRY[name] = factory_func 