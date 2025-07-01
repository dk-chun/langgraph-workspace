"""
Common validation utilities.
"""

from typing import Dict, Any, List
from my_agent.core.base_state import BaseState


def validate_state(state: BaseState) -> bool:
    """
    Validate state object.
    
    Args:
        state: State to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(state, dict):
        return False
    
    # Check required fields
    if "messages" not in state:
        return False
    
    return True


def validate_config(config: Dict[str, Any], required_keys: List[str] = None) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(config, dict):
        return False
    
    if required_keys:
        for key in required_keys:
            if key not in config:
                return False
    
    return True


def validate_messages(messages: List) -> bool:
    """
    Validate messages list.
    
    Args:
        messages: Messages to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(messages, list):
        return False
    
    if not messages:
        return False
    
    # Check each message has content
    for msg in messages:
        if not hasattr(msg, 'content'):
            return False
    
    return True 