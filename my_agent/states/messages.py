"""
Messages state definition with LangGraph add_messages support.
"""

from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class MessagesState(BaseModel):
    """
    Base state for message handling with LangGraph's add_messages reducer.
    
    This provides the core message list functionality that LangGraph expects.
    """
    
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list) 