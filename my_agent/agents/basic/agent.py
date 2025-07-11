"""
Basic agent implementation using LangGraph.
Simple, functional approach following LangGraph best practices.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from my_agent.states.chat.chat_input import ChatInputState
from my_agent.states.chat.chat_output import ChatOutputState
from my_agent.states.models.ollama import OllamaState
from my_agent.nodes.chat.chat_input import chat_input_node
from my_agent.nodes.chat.chat_output import chat_output_node
from my_agent.nodes.models.ollama import ollama_node


def create_basic_agent():
    """
    Create a basic conversational agent.
    
    Returns:
        Compiled LangGraph application
    """
    
    # Define the combined state schema
    class BasicAgentState(ChatInputState, ChatOutputState, OllamaState):
        """Combined state for the basic agent."""
        pass
    
    # Create the graph
    workflow = StateGraph(BasicAgentState)
    
    # Add nodes
    workflow.add_node("chat_input", chat_input_node)
    workflow.add_node("ollama", ollama_node)
    workflow.add_node("chat_output", chat_output_node)
    
    # Define the flow
    workflow.set_entry_point("chat_input")
    workflow.add_edge("chat_input", "ollama")
    workflow.add_edge("ollama", "chat_output")
    workflow.add_edge("chat_output", END)
    
    # Compile with memory
    return workflow.compile(checkpointer=MemorySaver())
