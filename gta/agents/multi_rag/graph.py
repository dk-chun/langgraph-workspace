"""
Multi-VectorStore RAG Agent Graph Definition.
"""

from typing import List, Tuple, Optional, Callable
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient

from gta.agents.multi_rag.state import MultiRAGState
from gta.agents.multi_rag.nodes import (
    create_extract_query_node,
    create_parallel_search_node,
    create_vectorstore_search_node,
    create_conditional_search_node,
    create_merge_results_node,
    create_prompt_node,
    create_chat_node
)


def create_multi_rag_graph(
    llm: Optional[BaseChatModel] = None,
    vectorstores: Optional[List[VectorStore]] = None,
    template_messages: Optional[List[Tuple[str, str]]] = None,
    top_k_per_store: int = 3,
    final_top_k: int = 5,
    merge_strategy: str = "simple"
) -> CompiledStateGraph:
    """
    Create Multi-VectorStore RAG graph with customizable components.
    
    Args:
        llm: Language model instance (default: ChatOllama)
        vectorstores: List of 3 vectorstore instances (default: 3 QdrantVectorStores)
        template_messages: Prompt template messages (default: RAG template)
        top_k_per_store: Number of documents to retrieve from each store (default: 3)
        final_top_k: Final number of documents after merging (default: 5)
        merge_strategy: Strategy for merging results (default: "simple")
        
    Returns:
        Compiled Multi-RAG graph
    """
    
    # Default LLM
    if llm is None:
        llm = ChatOllama(model="qwen3:0.6b", temperature=0.3)
    
    # Default vectorstores (3 identical stores with different collection names)
    if vectorstores is None:
        embeddings = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url="http://localhost:11434"
        )
        client = QdrantClient(url="http://localhost:6333")
        
        vectorstores = [
            QdrantVectorStore(
                client=client,
                collection_name="test_langgraph",
                embedding=embeddings
            ),
            QdrantVectorStore(
                client=client,
                collection_name="test_langgraph", 
                embedding=embeddings
            ),
            QdrantVectorStore(
                client=client,
                collection_name="test_langgraph",
                embedding=embeddings
            )
        ]
    
    # Ensure we have exactly 3 vectorstores
    if len(vectorstores) != 3:
        raise ValueError("Exactly 3 vectorstores are required for Multi-RAG")
    
    # Default template
    if template_messages is None:
        template_messages = [
            ("system", "You are a helpful assistant. Use the following context from multiple sources to answer the user's question accurately and concisely."),
            ("user", "Context from multiple sources:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ]
    
    # Create nodes with injected dependencies
    extract_query_node = create_extract_query_node()
    parallel_search_node = create_parallel_search_node(vectorstores, top_k_per_store)
    merge_results_node = create_merge_results_node(merge_strategy, final_top_k)
    prompt_node = create_prompt_node(template_messages)
    chat_node = create_chat_node(llm)
    
    # Build graph
    builder = StateGraph(MultiRAGState)
    
    # Add nodes
    builder.add_node("extract_query", extract_query_node)
    builder.add_node("parallel_search", parallel_search_node)
    builder.add_node("merge_results", merge_results_node)
    builder.add_node("prompt", prompt_node)
    builder.add_node("generate", chat_node)
    
    # Add edges - linear flow with parallel search in the middle
    builder.add_edge(START, "extract_query")
    builder.add_edge("extract_query", "parallel_search")
    builder.add_edge("parallel_search", "merge_results")
    builder.add_edge("merge_results", "prompt")
    builder.add_edge("prompt", "generate")
    builder.add_edge("generate", END)
    
    # Compile and return graph
    return builder.compile()


def create_multi_rag_graph_with_individual_nodes(
    llm: Optional[BaseChatModel] = None,
    vectorstores: Optional[List[VectorStore]] = None,
    template_messages: Optional[List[Tuple[str, str]]] = None,
    top_k_per_store: int = 3,
    final_top_k: int = 5,
    merge_strategy: str = "simple",
    use_conditional_search: bool = False,
    search_conditions: Optional[List[Callable[[MultiRAGState], bool]]] = None
) -> CompiledStateGraph:
    """
    Create Multi-VectorStore RAG graph with individual search nodes for each vectorstore.
    
    This approach provides:
    - Independent control over each vectorstore
    - Parallel execution using LangGraph's built-in parallelism
    - Conditional search capability
    - Better debugging and monitoring per vectorstore
    
    Args:
        llm: Language model instance (default: ChatOllama)
        vectorstores: List of 3 vectorstore instances (default: 3 QdrantVectorStores)
        template_messages: Prompt template messages (default: RAG template)
        top_k_per_store: Number of documents to retrieve from each store (default: 3)
        final_top_k: Final number of documents after merging (default: 5)
        merge_strategy: Strategy for merging results (default: "simple")
        use_conditional_search: Whether to use conditional search nodes (default: False)
        search_conditions: List of condition functions for each vectorstore (optional)
        
    Returns:
        Compiled Multi-RAG graph with individual search nodes
    """
    
    # Default LLM
    if llm is None:
        llm = ChatOllama(model="qwen3:0.6b", temperature=0.3)
    
    # Default vectorstores (3 identical stores with different collection names)
    if vectorstores is None:
        embeddings = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url="http://localhost:11434"
        )
        client = QdrantClient(url="http://localhost:6333")
        
        vectorstores = [
            QdrantVectorStore(
                client=client,
                collection_name="test_langgraph",
                embedding=embeddings
            ),
            QdrantVectorStore(
                client=client,
                collection_name="test_langgraph", 
                embedding=embeddings
            ),
            QdrantVectorStore(
                client=client,
                collection_name="test_langgraph",
                embedding=embeddings
            )
        ]
    
    # Ensure we have exactly 3 vectorstores
    if len(vectorstores) != 3:
        raise ValueError("Exactly 3 vectorstores are required for Multi-RAG")
    
    # Default template
    if template_messages is None:
        template_messages = [
            ("system", "You are a helpful assistant. Use the following context from multiple sources to answer the user's question accurately and concisely."),
            ("user", "Context from multiple sources:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ]
    
    # Create common nodes
    extract_query_node = create_extract_query_node()
    merge_results_node = create_merge_results_node(merge_strategy, final_top_k)
    prompt_node = create_prompt_node(template_messages)
    chat_node = create_chat_node(llm)
    
    # Create individual search nodes
    if use_conditional_search and search_conditions and len(search_conditions) == 3:
        # Use conditional search nodes
        vs1_search_node = create_conditional_search_node(
            vectorstores[0], "vs1", search_conditions[0], top_k_per_store
        )
        vs2_search_node = create_conditional_search_node(
            vectorstores[1], "vs2", search_conditions[1], top_k_per_store
        )
        vs3_search_node = create_conditional_search_node(
            vectorstores[2], "vs3", search_conditions[2], top_k_per_store
        )
    else:
        # Use regular individual search nodes
        vs1_search_node = create_vectorstore_search_node(vectorstores[0], "vs1", top_k_per_store)
        vs2_search_node = create_vectorstore_search_node(vectorstores[1], "vs2", top_k_per_store)
        vs3_search_node = create_vectorstore_search_node(vectorstores[2], "vs3", top_k_per_store)
    
    # Build graph
    builder = StateGraph(MultiRAGState)
    
    # Add nodes
    builder.add_node("extract_query", extract_query_node)
    builder.add_node("vs1_search", vs1_search_node)
    builder.add_node("vs2_search", vs2_search_node)
    builder.add_node("vs3_search", vs3_search_node)
    builder.add_node("merge_results", merge_results_node)
    builder.add_node("prompt", prompt_node)
    builder.add_node("generate", chat_node)
    
    # Add edges for parallel execution
    builder.add_edge(START, "extract_query")
    
    # Parallel search execution - all three searches run in parallel
    builder.add_edge("extract_query", "vs1_search")
    builder.add_edge("extract_query", "vs2_search")
    builder.add_edge("extract_query", "vs3_search")
    
    # Wait for all searches to complete before merging
    builder.add_edge(["vs1_search", "vs2_search", "vs3_search"], "merge_results")
    
    # Continue with linear flow
    builder.add_edge("merge_results", "prompt")
    builder.add_edge("prompt", "generate")
    builder.add_edge("generate", END)
    
    # Compile and return graph
    return builder.compile()


# Create default graph instances
# Legacy parallel search approach (kept for backward compatibility)
legacy_graph = create_multi_rag_graph() 

# New individual nodes approach (recommended)
multi_rag_graph = create_multi_rag_graph_with_individual_nodes()


# Example usage and utility functions
def create_search_condition_by_score_threshold(min_score: float = 0.5):
    """
    Create a condition function that only searches if previous search scores are below threshold.
    
    Args:
        min_score: Minimum score threshold
        
    Returns:
        Condition function for conditional search
    """
    def condition_fn(state: MultiRAGState) -> bool:
        if not state.search_scores:
            return True  # Always search if no previous scores
        
        avg_score = sum(state.search_scores.values()) / len(state.search_scores)
        return avg_score < min_score
    
    return condition_fn


def create_search_condition_by_query_length(min_length: int = 10):
    """
    Create a condition function that only searches for queries above certain length.
    
    Args:
        min_length: Minimum query length
        
    Returns:
        Condition function for conditional search
    """
    def condition_fn(state: MultiRAGState) -> bool:
        return len(state.question) >= min_length
    
    return condition_fn


# Example: Create graph with conditional search
def create_conditional_multi_rag_example():
    """
    Example of creating multi-rag graph with conditional search.
    
    - vs1: Always searches
    - vs2: Only searches if query is long enough  
    - vs3: Only searches if previous scores are low
    """
    conditions = [
        lambda state: True,  # vs1 always searches
        create_search_condition_by_query_length(15),  # vs2 for longer queries
        create_search_condition_by_score_threshold(0.7)  # vs3 for low-scoring results
    ]
    
    return create_multi_rag_graph_with_individual_nodes(
        use_conditional_search=True,
        search_conditions=conditions,
        top_k_per_store=5,
        final_top_k=8
    ) 