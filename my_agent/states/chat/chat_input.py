"""
Chat input state definition.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class ChatInputState(MessagesState):
    """
    State for handling user chat input.
    
    Inherits from MessagesState to leverage built-in message handling.
    """
    
    user_input: Optional[str] = Field(
        default=None,
        description="Raw user input text"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for conversation tracking"
    )
    
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata for the input"
    )
