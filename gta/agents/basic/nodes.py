"""
Basic Agent Nodes.
"""

import functools
from typing import Dict, Any, Optional, Callable
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from gta.agents.basic.state import BasicState


def chat_node(state: BasicState, llm: BaseChatModel, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Basic chat node with optional system prompt."""
    messages = list(state.messages)
    
    # Add system prompt if provided
    if system_prompt:
        messages = [SystemMessage(content=system_prompt)] + messages
    
    # Generate response
    response = llm.invoke(messages)
    
    return {"messages": [AIMessage(content=response.content)]}


# Node factory functions
def create_chat_node(llm: BaseChatModel, system_prompt: Optional[str] = None) -> Callable:
    """Create chat node with LLM and optional system prompt."""
    return functools.partial(chat_node, llm=llm, system_prompt=system_prompt) 