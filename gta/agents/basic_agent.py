"""Basic agent implementation using LangGraph with runtime configuration."""

from typing import Any
from langgraph.graph import StateGraph, END, START

from gta.states.messages_state import MessagesState
from gta.configs.chat_config import ChatConfig
from gta.nodes.chat_node import chat_node


def create_basic_agent() -> Any:  # returns CompiledStateGraph; not exported
    """
    Create a basic conversational agent with runtime configuration.
    
    Uses MessagesState for data flow and ChatConfig for model settings.
    Supports multiple LLM providers (Ollama, OpenAI) via configuration.
    
    Returns:
        Compiled LangGraph application
    """
    
    # Create the graph with MessagesState and config schema
    workflow = StateGraph(MessagesState, config_schema=ChatConfig)
    
    # Add nodes
    workflow.add_node("chat", chat_node)
    
    # Define the flow
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    
    # Compile with memory
    return workflow.compile(name="basic_agent") 