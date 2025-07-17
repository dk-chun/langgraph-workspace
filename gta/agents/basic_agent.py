"""Basic agent implementation using LangGraph with runtime configuration."""

from typing import Any
from langgraph.graph import StateGraph, END, START

from gta.states.messages import MessagesState
from gta.configs import OllamaConfig
from gta.nodes.models.ollama import ollama_node


def create_basic_agent() -> Any:  # returns CompiledStateGraph; not exported
    """
    Create a basic conversational agent with runtime configuration.
    
    Uses MessagesState for data flow and OllamaConfig for model settings.
    
    Returns:
        Compiled LangGraph application
    """
    
    # Create the graph with MessagesState and config schema
    workflow = StateGraph(MessagesState, config_schema=OllamaConfig)
    
    # Add nodes
    workflow.add_node("ollama", ollama_node)
    
    # Define the flow
    workflow.add_edge(START, "ollama")
    workflow.add_edge("ollama", END)
    
    # Compile with memory
    return workflow.compile(name="basic_agent") 