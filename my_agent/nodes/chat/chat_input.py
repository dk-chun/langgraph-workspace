"""
Chat input node implementation.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage


def chat_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process user input and convert to message format.
    
    Args:
        state: Current chat input state as dictionary
        
    Returns:
        Updated state with processed input
    """
    
    # Get user input from dictionary
    user_input = state.get("user_input")
    
    if not user_input:
        return {
            "messages": [],
            "error": "No user input provided"
        }
    
    # Create human message
    human_message = HumanMessage(content=user_input)
    
    # Return updated state
    return {
        "messages": [human_message],
        "user_input": user_input,
        "session_id": state.get("session_id"),
        "metadata": state.get("metadata", {})
    } 