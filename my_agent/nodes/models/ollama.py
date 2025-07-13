"""
Ollama node implementation with runtime configuration.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from my_agent.states.messages import MessagesState


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
        config_data = config.get("configurable", {})
        
        base_url = config_data.get("base_url", "http://localhost:11434")
        model_name = config_data.get("model_name", "qwen3:0.6b")
        temperature = config_data.get("temperature", 0.7)
        max_tokens = config_data.get("num_predict") or config_data.get("max_tokens")
        
        # Prepare Ollama options
        options = {}
        if config_data.get("num_ctx"):
            options["num_ctx"] = config_data.get("num_ctx")
        if config_data.get("repeat_penalty"):
            options["repeat_penalty"] = config_data.get("repeat_penalty")
        if config_data.get("top_k"):
            options["top_k"] = config_data.get("top_k")
        if config_data.get("top_p"):
            options["top_p"] = config_data.get("top_p")
        
        # Initialize Ollama client
        llm = ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=config_data.get("timeout"),
            keep_alive=config_data.get("keep_alive"),
            **options,
            **config_data.get("model_kwargs", {})
        )
        
        # Prepare messages
        messages = list(state.messages)
        
        # Add system prompt if provided
        system_prompt = config_data.get("system_prompt")
        if system_prompt:
            system_message = SystemMessage(content=system_prompt)
            messages.insert(0, system_message)
        
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