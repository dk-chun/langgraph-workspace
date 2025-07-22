"""
Multi-VectorStore RAG Agent Graph Definition with Runtime Configuration.
"""

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END, START

from gta.agents.multi_rag.state import MultiRAGState
from gta.agents.multi_rag.config import MultiRAGConfigSchema
from gta.agents.multi_rag.nodes import (
    create_extract_query_node,
    create_individual_search_node_with_config,
    create_merge_results_node_with_config,
    create_prompt_node_with_config,
    create_chat_node_with_config
)


def create_multi_rag_graph_with_runtime_config() -> CompiledStateGraph:
    """
    Create Multi-VectorStore RAG graph with runtime configuration support.
    
    This graph uses LangGraph's runtime configuration feature to dynamically
    configure LLM, embeddings, and vectorstores at execution time.
    
    Uses individual search nodes for each vectorstore to enable:
    - Independent control over each vectorstore via runtime config
    - Parallel execution using LangGraph's built-in parallelism
    - Dynamic provider configuration at runtime
    - Better debugging and monitoring per vectorstore
    
    Returns:
        Compiled Multi-RAG graph with runtime configuration schema
        
    Example:
        Basic usage with Ollama + Qdrant:
        
        ```python
        from langchain_core.messages import HumanMessage
        
        config = {
            "configurable": {
                "llm": {"provider": "ollama", "model": "qwen3:0.6b"},
                "vectorstores": [
                    {"provider": "qdrant", "collection_name": "docs", "url": "http://localhost:6333",
                     "embedding": {"provider": "ollama", "model": "bge-m3:latest"}},
                    {"provider": "qdrant", "collection_name": "code", "url": "http://localhost:6333", 
                     "embedding": {"provider": "ollama", "model": "bge-m3:latest"}},
                    {"provider": "qdrant", "collection_name": "wiki", "url": "http://localhost:6333",
                     "embedding": {"provider": "ollama", "model": "bge-m3:latest"}}
                ]
            }
        }
        
        graph = create_multi_rag_graph_with_runtime_config()
        result = graph.invoke({"messages": [HumanMessage("What is AI?")]}, config)
        ```
        
        Mixed providers (OpenAI LLM + different vectorstores):
        
        ```python
        config = {
            "configurable": {
                "llm": {"provider": "openai", "model": "gpt-4", "api_key": "sk-..."},
                "vectorstores": [
                    {"provider": "pinecone", "index_name": "docs", "api_key": "...",
                     "embedding": {"provider": "openai", "model": "text-embedding-3-small", "api_key": "sk-..."}},
                    {"provider": "chroma", "collection_name": "code", "path": "./chroma_db",
                     "embedding": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"}},
                    {"provider": "qdrant", "collection_name": "wiki", "url": "http://localhost:6333",
                     "embedding": {"provider": "ollama", "model": "nomic-embed-text"}}
                ]
            }
        }
        ```
    """
    
    # Create nodes that accept runtime configuration
    extract_query = create_extract_query_node()
    vs1_search = create_individual_search_node_with_config(0)  # First vectorstore
    vs2_search = create_individual_search_node_with_config(1)  # Second vectorstore
    vs3_search = create_individual_search_node_with_config(2)  # Third vectorstore
    merge_results = create_merge_results_node_with_config()
    prompt = create_prompt_node_with_config()
    generate = create_chat_node_with_config()
    
    # Build graph with config schema
    builder = StateGraph(MultiRAGState, config_schema=MultiRAGConfigSchema)
    
    # Add nodes
    builder.add_node("extract_query", extract_query)
    builder.add_node("vs1_search", vs1_search)
    builder.add_node("vs2_search", vs2_search)
    builder.add_node("vs3_search", vs3_search)
    builder.add_node("merge_results", merge_results)
    builder.add_node("prompt", prompt)
    builder.add_node("generate", generate)
    
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


# Main graph instance
multi_rag_graph = create_multi_rag_graph_with_runtime_config() 