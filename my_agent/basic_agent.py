"""
Basic Agent - Main entry point for the basic conversational agent.

This module provides a simple, ready-to-use conversational agent
built on LangGraph with Private State pattern.
"""

from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from my_agent.agents.basic import BasicAgent, BasicCoordinator, BasicService


def create_basic_agent_graph(config: RunnableConfig):
    """
    Create a basic agent graph for LangGraph server deployment.
    
    Args:
        config: RunnableConfig from LangGraph server
        
    Returns:
        Compiled LangGraph application
    """
    agent = BasicAgent(
        model_name="qwen3:0.6b",
        temperature=0.7,
        system_prompt="You are a helpful assistant."
    )
    return agent.app


def create_basic_agent(
    model_name: str = "qwen3:0.6b",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    base_url: Optional[str] = None
) -> BasicAgent:
    """
    Create a basic conversational agent.
    
    Args:
        model_name: Ollama model name
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        system_prompt: System prompt for the model
        base_url: Ollama server base URL
        
    Returns:
        BasicAgent instance
        
            Example:
        >>> agent = create_basic_agent(
        ...     model_name="qwen3:0.6b",
        ...     temperature=0.7,
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> response = agent.invoke("Hello, how are you?")
        >>> print(response["response"])
    """
    return BasicAgent(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        base_url=base_url
    )


def create_basic_coordinator(
    default_model: str = "qwen3:0.6b",
    default_temperature: float = 0.7,
    max_sessions: int = 100
) -> BasicCoordinator:
    """
    Create a basic agent coordinator for multi-session management.
    
    Args:
        default_model: Default model for new sessions
        default_temperature: Default temperature
        max_sessions: Maximum concurrent sessions
        
    Returns:
        BasicCoordinator instance
        
    Example:
        >>> coordinator = create_basic_coordinator()
        >>> response = coordinator.chat("Hello!", session_id="user_123")
        >>> print(response["response"])
    """
    return BasicCoordinator(
        default_model=default_model,
        default_temperature=default_temperature,
        max_sessions=max_sessions
    )


def create_basic_service(
    model_name: str = "qwen3:0.6b",
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    use_coordinator: bool = True,
    **kwargs
) -> BasicService:
    """
    Create a basic agent service for deployment.
    
    Args:
        model_name: Ollama model name
        temperature: Generation temperature
        system_prompt: System prompt
        use_coordinator: Whether to use coordinator for multi-session
        **kwargs: Additional service parameters
        
    Returns:
        BasicService instance
        
            Example:
        >>> service = create_basic_service(
        ...     model_name="qwen3:0.6b",
        ...     system_prompt="You are a helpful assistant.",
        ...     use_coordinator=True
        ... )
        >>> response = service.process_message("Hello!")
        >>> print(response["response"])
    """
    return BasicService(
        model_name=model_name,
        temperature=temperature,
        system_prompt=system_prompt,
        use_coordinator=use_coordinator,
        **kwargs
    )


# Quick usage examples
def demo_basic_agent():
    """
    Demo function showing basic agent usage.
    """
    print("=== Basic Agent Demo ===")
    
    # Create agent
    agent = create_basic_agent(
        model_name="qwen3:0.6b",
        temperature=0.7,
        system_prompt="You are a helpful assistant. Be concise."
    )
    
    # Single conversation
    response = agent.invoke("What is Python?")
    print(f"Response: {response['response']}")
    
    # Follow-up in same session
    response = agent.invoke("Give me a simple example", session_id="demo")
    print(f"Follow-up: {response['response']}")


def demo_coordinator():
    """
    Demo function showing coordinator usage.
    """
    print("\n=== Coordinator Demo ===")
    
    # Create coordinator
    coordinator = create_basic_coordinator(max_sessions=5)
    
    # Multiple sessions
    response1 = coordinator.chat("Hi, I'm Alice", session_id="alice")
    response2 = coordinator.chat("Hi, I'm Bob", session_id="bob")
    
    print(f"Alice: {response1['response']}")
    print(f"Bob: {response2['response']}")
    
    # Session info
    print(f"Active sessions: {coordinator.list_sessions()}")


if __name__ == "__main__":
    # Run demos (requires OpenAI API key)
    try:
        demo_basic_agent()
        demo_coordinator()
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure Ollama is running on http://localhost:11434") 