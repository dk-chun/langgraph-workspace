"""
Main RAG agent implementation using LangGraph and Ollama.
"""

from my_agent.agents.rag.agent import create_rag_agent

# Create the RAG agent instance with default configuration
rag_agent = create_rag_agent() 