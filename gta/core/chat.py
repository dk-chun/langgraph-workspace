"""
Pure chat/LLM functions - Framework independent.
"""

from typing import List, Any
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_core.language_models import BaseChatModel


def invoke_llm(messages: List[AnyMessage], llm: BaseChatModel) -> str:
    """
    Pure LLM invocation function.
    
    Args:
        messages: List of messages to send to LLM
        llm: LLM instance to use
        
    Returns:
        Response content as string
    """
    response = llm.invoke(messages)
    return response.content


def add_system_message(messages: List[AnyMessage], system_prompt: str) -> List[AnyMessage]:
    """
    Add system message to the beginning of message list.
    
    Args:
        messages: Existing messages
        system_prompt: System prompt text
        
    Returns:
        New message list with system message prepended
    """
    return [SystemMessage(content=system_prompt)] + list(messages) 