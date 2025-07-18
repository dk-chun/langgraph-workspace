"""
RAG Agent Graph Definition.
"""

from typing import List, Tuple, Optional
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient

from gta.agents.simple_rag.state import RAGState
from gta.agents.simple_rag.nodes import create_extract_query_node, create_search_node, create_prompt_node, create_chat_node


def create_rag_graph(
    llm: Optional[BaseChatModel] = None,
    vectorstore: Optional[VectorStore] = None,
    template_messages: Optional[List[Tuple[str, str]]] = None,
    top_k: int = 5
):
    """
    Create RAG graph with customizable components.
    
    Args:
        llm: Language model instance (default: ChatOllama)
        vectorstore: Vector store instance (default: QdrantVectorStore)
        template_messages: Prompt template messages (default: RAG template)
        top_k: Number of documents to retrieve (default: 5)
        
    Returns:
        Compiled RAG graph
    """
    
    # Default LLM
    if llm is None:
        llm = ChatOllama(model="qwen3:0.6b", temperature=0.3)
    
    # Default vectorstore
    if vectorstore is None:
        embeddings = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url="http://localhost:11434"
        )
        client = QdrantClient(url="http://localhost:6333")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="test_langgraph",
            embedding=embeddings
        )
    
    # Default template
    if template_messages is None:
        template_messages = [
            ("system", "You are a helpful assistant. Use the following context to answer the user's question accurately and concisely."),
            ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ]
    
    # Create nodes with injected dependencies
    extract_query_node = create_extract_query_node()
    search_node = create_search_node(vectorstore, top_k)
    prompt_node = create_prompt_node(template_messages)
    chat_node = create_chat_node(llm)
    
    # Build graph
    builder = StateGraph(RAGState)
    
    # Add nodes
    builder.add_node("extract_query", extract_query_node)
    builder.add_node("search", search_node)
    builder.add_node("prompt", prompt_node)
    builder.add_node("generate", chat_node)
    
    # Add edges
    builder.add_edge(START, "extract_query")
    builder.add_edge("extract_query", "search")
    builder.add_edge("search", "prompt")
    builder.add_edge("prompt", "generate")
    builder.add_edge("generate", END)
    
    # Compile and return graph
    return builder.compile()


# Create default graph instance for backward compatibility
graph = create_rag_graph() 