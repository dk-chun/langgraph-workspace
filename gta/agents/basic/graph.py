"""
Basic Agent Graph Definition with Runtime Configuration.
"""

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from gta.agents.basic.state import BasicState
from gta.agents.basic.config import BasicConfigSchema
from gta.agents.basic.nodes import create_chat_node_with_config


def create_basic_graph_with_runtime_config() -> CompiledStateGraph:
    """
    Create basic chat graph with runtime configuration support.
    
    This graph uses LangGraph's runtime configuration feature to dynamically
    configure LLM and system prompt at execution time.
    
    Returns:
        Compiled basic chat graph with runtime configuration schema
        
    Example:
        Basic usage with Ollama:
        
        ```python
        from langchain_core.messages import HumanMessage
        
        config = {
            "configurable": {
                "llm": {"provider": "ollama", "model": "qwen3:0.6b"},
                "system_prompt": "You are a helpful coding assistant."
            }
        }
        
        graph = create_basic_graph_with_runtime_config()
        result = graph.invoke({"messages": [HumanMessage("Hello!")]}, config)
        ```
        
        With OpenAI:
        
        ```python
        config = {
            "configurable": {
                "llm": {"provider": "openai", "model": "gpt-4", "api_key": "sk-..."},
                "system_prompt": "You are a creative writing assistant."
            }
        }
        ```
    """
    
    # Create node that accepts runtime configuration
    chat = create_chat_node_with_config()
    
    # Build graph with config schema
    builder = StateGraph(BasicState, config_schema=BasicConfigSchema)
    
    # Add nodes
    builder.add_node("chat", chat)
    
    # Add edges
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    
    # Compile and return graph
    return builder.compile()


# Legacy function (static configuration)
def create_basic_graph(
    llm: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None
) -> CompiledStateGraph:
    """
    Create basic chat graph with static configuration (legacy).
    
    Args:
        llm: Language model instance (default: ChatOllama)
        system_prompt: System prompt message (default: "You are a helpful assistant.")
        
    Returns:
        Compiled basic chat graph
    """
    
    # Default LLM
    if llm is None:
        llm = ChatOllama(model="qwen3:0.6b", temperature=0.7)
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."
    
    # Legacy chat node (static config)
    def legacy_chat_node(state: BasicState) -> Dict[str, Any]:
        messages = list(state.messages)
        
        # Add system prompt if provided
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages
        
        # Generate response
        response = llm.invoke(messages)
        
        return {"messages": [AIMessage(content=response.content)]}
    
    # Build graph
    builder = StateGraph(BasicState)
    
    # Add nodes
    builder.add_node("chat", legacy_chat_node)
    
    # Add edges
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    
    # Compile and return graph
    return builder.compile()


# Main graph instance (runtime configuration)
basic_graph = create_basic_graph_with_runtime_config()

# Legacy graph instance for backward compatibility
basic_graph_legacy = create_basic_graph() 