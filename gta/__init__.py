"""
GTA Package - Modular LangGraph Agent Services with Runtime Configuration

This package provides independent agent services with runtime configuration support:
- Basic Agent: Simple conversational service
- Simple RAG Agent: Single vectorstore retrieval service
- Multi RAG Agent: Multiple vectorstore retrieval service with dynamic provider configuration

Each service supports runtime configuration for maximum flexibility and provider independence.
"""

from gta.agents import (
    basic_graph, 
    create_basic_graph,
    create_basic_graph_with_runtime_config,
    simple_rag_graph, 
    create_rag_graph,
    create_rag_graph_with_runtime_config,
    multi_rag_graph,
    create_multi_rag_graph_with_runtime_config
)

# Service graphs and factories
__all__ = [
    # Basic Agent
    "basic_graph",
    "create_basic_graph",
    "create_basic_graph_with_runtime_config",
    
    # Simple RAG Agent
    "simple_rag_graph",
    "create_rag_graph", 
    "create_rag_graph_with_runtime_config",
    
    # Multi RAG Agent (Runtime Config)
    "multi_rag_graph",
    "create_multi_rag_graph_with_runtime_config"
]

# Version information
__version__ = "4.0.0"
__author__ = "LangGraph Agent Services" 