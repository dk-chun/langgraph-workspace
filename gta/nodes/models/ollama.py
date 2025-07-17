"""
Ollama node implementation with runtime configuration.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from gta.states.messages import MessagesState


def ollama_node(state: MessagesState, config: RunnableConfig) -> dict:
    """
    Process messages using Ollama LLM.
    
    Uses runtime configuration for all model settings.
    
    Args:
        state: Current messages state
        config: Runtime configuration with model settings
        
    Returns:
        Updated state dictionary with AI response
    """
    
    try:
        # Get configuration from config
        config_data = config.get("configurable") or {}
        
        # Prepare Ollama configuration
        options = {
            key: config_data.get(key) for key in [
                "num_ctx",
                "repeat_penalty",
                "top_k",
                "top_p"
            ] if config_data.get(key) is not None
        }
        
        # Initialize Ollama client
        llm = ChatOllama(
            base_url=config_data.get("base_url") or "http://localhost:11434",
            model=config_data.get("model_name") or "qwen3:0.6b",
            temperature=config_data.get("temperature") or 0.7,
            num_predict=config_data.get("num_predict") or config_data.get("max_tokens"),
            keep_alive=config_data.get("keep_alive"),
            **options,
            **config_data.get("model_kwargs") or {}
        )
        
        # Prepare messages
        messages = list(state.messages)
        
        # Add system prompt if provided
        if system_prompt := config_data.get("system_prompt"):
            messages.insert(0, SystemMessage(content=system_prompt))
        
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