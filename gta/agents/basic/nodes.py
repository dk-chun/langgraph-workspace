"""
Basic Agent Adapter Nodes.
"""

import functools
from typing import Dict, Any, Optional, Callable
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel

from gta.core.chat import invoke_llm, add_system_message
from gta.agents.basic.state import BasicState


def _chat_adapter(state: BasicState, llm: BaseChatModel, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Basic chat adapter with optional system prompt."""
    messages = list(state.messages)  # Fixed: Pydantic model access
    
    # Add system prompt if provided
    if system_prompt:
        messages = add_system_message(messages, system_prompt)
    
    # Generate response
    response_content = invoke_llm(messages, llm)
    
    return {"messages": [AIMessage(content=response_content)]}


# Node factory functions
def create_chat_node(llm: BaseChatModel, system_prompt: Optional[str] = None) -> Callable:
    """Create chat node with LLM and optional system prompt."""
    return functools.partial(_chat_adapter, llm=llm, system_prompt=system_prompt) 