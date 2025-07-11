"""
Chat output node implementation.
"""

from typing import Dict, Any


def chat_output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and format final chat output.
    
    Args:
        state: Current chat output state as dictionary
        
    Returns:
        Final formatted output
    """
    
    # Get the last message (should be AI response)
    messages = state.get("messages", [])
    if not messages:
        return {
            "response": "No response generated",
            "confidence": 0.0,
            "sources": [],
            "tokens_used": 0,
            "finish_reason": "error"
        }
    
    # Get the last AI message
    last_message = messages[-1]
    response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Extract metadata if available
    confidence = state.get('confidence')
    sources = state.get('sources', [])
    tokens_used = state.get('tokens_used')
    finish_reason = state.get('finish_reason', 'stop')
    
    return {
        "response": response_text,
        "confidence": confidence,
        "sources": sources,
        "tokens_used": tokens_used,
        "finish_reason": finish_reason,
        "messages": messages
    } 