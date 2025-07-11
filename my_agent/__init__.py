"""
My Agent Package - Modular LangGraph Agent Services

This package provides independent agent services that can be deployed separately:
- Basic Agent: General purpose conversational service using Private State pattern

Each service is designed to work independently and can be deployed on LangGraph server.
"""

from my_agent.agents.basic import create_basic_agent
from typing import Dict, Any

def chat(
    user_input: str,
    session_id: str = "default",
    model_name: str = "qwen3:0.6b",
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant.",
    base_url: str = "http://localhost:11434",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick chat function for basic agent.
    
    Args:
        user_input: User's message
        session_id: Session identifier
        model_name: Ollama model name
        temperature: Generation temperature
        system_prompt: System prompt
        base_url: Ollama server URL
        **kwargs: Additional parameters
        
    Returns:
        Agent response
    """
    # Create agent
    agent = create_basic_agent()
    
    # Prepare initial state
    initial_state = {
        "user_input": user_input,
        "session_id": session_id,
        "model_name": model_name,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "base_url": base_url,
        "messages": [],
        **kwargs
    }
    
    # Execute the graph
    config = {"configurable": {"thread_id": session_id}}
    return agent.invoke(initial_state, config)

# Service factories
__all__ = [
    "create_basic_agent",
    "chat"
]

# Version information
__version__ = "2.0.0"
__author__ = "LangGraph Agent Services" 