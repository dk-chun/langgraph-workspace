"""
RAG Agent State Definition.
"""

from typing import Annotated, List, Any
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(BaseModel):
    """
    RAG state with message reducer and validation.
    """
    question: str = Field(default="", description="Extracted question from user message")
    context: str = Field(default="", description="Retrieved context from documents")
    documents: List[Any] = Field(default_factory=list, description="Retrieved documents")
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list, description="Conversation messages") 