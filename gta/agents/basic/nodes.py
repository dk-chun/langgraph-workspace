"""
Basic Agent Nodes with Runtime Configuration.
"""

import functools
from typing import Dict, Any, Optional, Callable
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from gta.agents.basic.state import BasicState
from gta.agents.basic.config import DEFAULT_BASIC_CONFIG
from gta.core.factories import create_llm


def _get_runtime_config(config: RunnableConfig) -> Dict[str, Any]:
    """Get runtime configuration with fallback to defaults."""
    if not config or "configurable" not in config:
        return DEFAULT_BASIC_CONFIG
    
    runtime_config = config["configurable"]
    
    # Merge with defaults for missing keys
    merged_config = {}
    merged_config.update(DEFAULT_BASIC_CONFIG)
    merged_config.update(runtime_config)
    
    return merged_config


def chat_node(state: BasicState, config: RunnableConfig) -> Dict[str, Any]:
    """Basic chat node with runtime configuration."""
    messages = list(state.messages)
    
    # Get configuration with defaults
    runtime_config = _get_runtime_config(config)
    system_prompt = runtime_config.get("system_prompt")
    llm_config = runtime_config["llm"]
    
    # Add system prompt if provided
    if system_prompt:
        messages = [SystemMessage(content=system_prompt)] + messages
    
    # Create LLM from config and generate response
    llm = create_llm(llm_config)
    response = llm.invoke(messages)
    
    return {"messages": [AIMessage(content=response.content)]}


# Node factory functions
def create_chat_node_with_config() -> Callable[[BasicState, RunnableConfig], Dict[str, Any]]:
    """Create chat node that uses runtime configuration."""
    return chat_node 