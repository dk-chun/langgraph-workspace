"""
Basic Agent Graph Definition.
"""

from typing import Optional
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel

from gta.agents.basic.state import BasicState
from gta.agents.basic.nodes import create_chat_node


def create_basic_graph(
    llm: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None
) -> CompiledStateGraph:
    """
    Create basic chat graph with customizable components.
    
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
    
    # Create nodes with injected dependencies
    chat_node = create_chat_node(llm, system_prompt)
    
    # Build graph
    builder = StateGraph(BasicState)
    
    # Add nodes
    builder.add_node("chat", chat_node)
    
    # Add edges
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    
    # Compile and return graph
    return builder.compile()


# Create default graph instance for backward compatibility
graph = create_basic_graph() 