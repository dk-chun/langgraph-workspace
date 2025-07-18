"""
My Agent Package - Modular LangGraph Agent Services

This package provides independent agent services that can be deployed separately:
- Basic Agent: General purpose conversational service using runtime configuration
- RAG Agent: Retrieval-Augmented Generation with Qdrant vector search

Each service is designed to work independently and can be deployed on LangGraph server.
"""

from gta.agents.basic_agent import create_basic_agent
from gta.agents.rag_agent import create_rag_agent, create_simple_rag_agent

# Service factories
__all__ = [
    "create_basic_agent",
    "create_rag_agent",
    "create_simple_rag_agent"
]

# Version information
__version__ = "2.0.0"
__author__ = "LangGraph Agent Services" 