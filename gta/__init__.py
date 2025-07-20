"""
GTA Package - Modular LangGraph Agent Services

This package provides independent agent services using functools.partial pattern:
- Basic Agent: Simple conversational service
- RAG Agent: Retrieval-Augmented Generation service

Each service is designed with pure functions and adapter pattern for maximum reusability.
"""

from gta.agents import simple_rag_graph, create_basic_graph, simple_rag_graph, create_rag_graph

# Service graphs and factories
__all__ = [
    "simple_rag_graph",
    "create_basic_graph", 
    "simple_rag_graph",
    "create_rag_graph"
]

# Version information
__version__ = "3.0.0"
__author__ = "LangGraph Agent Services" 