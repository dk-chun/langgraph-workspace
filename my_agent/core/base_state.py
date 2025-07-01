"""
Base state definitions for all agents.
"""

from typing import Optional, Dict, Any, List
from langgraph.graph import MessagesState


class BaseState(MessagesState):
    """
    Base state class for all agents.
    
    Provides common state variables and methods.
    """
    # Common metadata - these will be added to the state dict
    agent_type: Optional[str]
    session_id: Optional[str]
    timestamp: Optional[str]
    metadata: Dict[str, Any]


def get_last_message_content(state: BaseState) -> str:
    """Get content of the last message from state."""
    messages = state.get("messages", [])
    if not messages:
        return ""
    return str(messages[-1].content)


def add_metadata(state: BaseState, key: str, value: Any) -> None:
    """Add metadata to the state."""
    if "metadata" not in state:
        state["metadata"] = {}
    state["metadata"][key] = value


def set_agent_type(state: BaseState, agent_type: str) -> None:
    """Set the agent type."""
    state["agent_type"] = agent_type 