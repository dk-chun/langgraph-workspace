"""
Ollama node implementation.
"""

import os
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage


def ollama_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process messages using Ollama LLM.
    
    Args:
        state: Current Ollama state as dictionary
        
    Returns:
        Updated state with AI response
    """
    
    try:
        # Get configuration from dictionary
        base_url = state.get("base_url") or "http://localhost:11434"
        model_name = state.get("model_name") or "qwen3:0.6b"
        temperature = state.get("temperature") or 0.7
        max_tokens = state.get("max_tokens") or state.get("num_predict")
        
        # Prepare Ollama options
        options = {}
        if state.get("num_ctx"):
            options["num_ctx"] = state.get("num_ctx")
        if state.get("num_predict"):
            options["num_predict"] = state.get("num_predict")
        if state.get("repeat_penalty"):
            options["repeat_penalty"] = state.get("repeat_penalty")
        if state.get("top_k"):
            options["top_k"] = state.get("top_k")
        if state.get("top_p"):
            options["top_p"] = state.get("top_p")
        
        # Merge with additional options
        options.update(state.get("options", {}))
        
        # Initialize Ollama client
        llm = ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=state.get("timeout"),
            keep_alive=state.get("keep_alive"),
            format=state.get("format"),
            **options,
            **state.get("model_kwargs", {})
        )
        
        # Prepare messages
        messages = list(state.get("messages", []))
        
        # Add system prompt if provided
        system_prompt = state.get("system_prompt")
        if system_prompt:
            system_message = SystemMessage(content=system_prompt)
            messages.insert(0, system_message)
        
        # Generate response
        response = llm.invoke(messages)
        
        # Create AI message
        ai_message = AIMessage(content=response.content)
        
        return {
            "messages": state.get("messages", []) + [ai_message],
            "response": response.content,
            "finish_reason": "stop",
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": base_url
        }
        
    except Exception as e:
        return {
            "messages": state.get("messages", []),
            "error": f"Ollama API error: {str(e)}",
            "finish_reason": "error"
        } 