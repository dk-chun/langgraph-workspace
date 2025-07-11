"""
Chat output state definition.
"""

from typing import Optional, List
from pydantic import Field
from langgraph.graph import MessagesState


class ChatOutputState(MessagesState):
    """
    State for handling agent chat output.
    
    Inherits from MessagesState to leverage built-in message handling.
    """
    
    response: Optional[str] = Field(
        default=None,
        description="Generated response text"
    )
    
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for the response (0.0-1.0)"
    )
    
    sources: Optional[List[str]] = Field(
        default_factory=list,
        description="Sources used to generate the response"
    )
    
    tokens_used: Optional[int] = Field(
        default=None,
        description="Number of tokens used in generation"
    )
    
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason for completion (stop, length, etc.)"
    ) 