"""
Basic Agent State Definition.
"""

from typing import Annotated, List
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class BasicState(BaseModel):
    """
    Simple basic chat state with message reducer.
    """
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list) 