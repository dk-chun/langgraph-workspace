"""
Unified chat node implementation supporting multiple LLM providers.
"""

from typing import Dict, Any, Callable, Union
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from gta.states.messages_state import MessagesState


def create_ollama_llm(config: Dict[str, Any]) -> ChatOllama:
    """Create Ollama LLM instance from configuration."""
    # Prepare Ollama configuration
    options = {
        key: config.get(key) for key in [
            "num_ctx",
            "repeat_penalty",
            "top_k",
            "top_p"
        ] if config.get(key) is not None
    }
    
    return ChatOllama(
        base_url=config.get("base_url") or "http://localhost:11434",
        model=config.get("model_name") or "qwen3:0.6b",
        temperature=config.get("temperature") or 0.7,
        num_predict=config.get("num_predict") or config.get("max_tokens"),
        keep_alive=config.get("keep_alive"),
        **options,
        **config.get("model_kwargs") or {}
    )





# LLM factory registry
LLM_FACTORIES: Dict[str, Callable[[Dict[str, Any]], BaseChatModel]] = {
    "ollama": create_ollama_llm,
}


def create_llm(provider: str, config: Dict[str, Any]) -> BaseChatModel:
    """Create LLM instance based on provider."""
    factory = LLM_FACTORIES.get(provider)
    if not factory:
        raise ValueError(f"Unknown provider: {provider}")
    return factory(config)


def prepare_messages(state: MessagesState, config: Dict[str, Any]) -> list[AnyMessage]:
    """Prepare messages with optional system prompt."""
    messages = list(state.messages)
    
    # Add system prompt if provided
    if system_prompt := config.get("system_prompt"):
        messages.insert(0, SystemMessage(content=system_prompt))
    
    return messages


def chat_node(state: MessagesState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Process messages using configurable LLM providers.
    
    Supports Ollama and OpenAI providers with runtime configuration.
    
    Args:
        state: Current messages state
        config: Runtime configuration with provider and model settings
        
    Returns:
        Updated state dictionary with AI response
    """
    
    try:
        # Get configuration
        config_data = config.get("configurable") or {}
        provider = config_data.get("provider", "ollama").lower()
        
        # Create LLM instance
        llm = create_llm(provider, config_data)
        
        # Prepare messages
        messages = prepare_messages(state, config_data)
        
        # Generate response
        response = llm.invoke(messages)
        
        # Create AI message
        ai_message = AIMessage(content=response.content)
        
        return {
            "messages": [ai_message]  # add_messages will append this
        }
            
    except Exception as e:
        # Return error as AI message
        error_message = AIMessage(content=f"죄송합니다. 오류가 발생했습니다: {str(e)}")
        return {
            "messages": [error_message]
        }

 