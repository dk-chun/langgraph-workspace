"""
OpenAI node implementation.
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage


def openai_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process messages using OpenAI LLM.
    
    Args:
        state: Current OpenAI state as dictionary
        
    Returns:
        Updated state with AI response
    """
    
    try:
        # Get API key from state or environment
        api_key = state.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "messages": state.get("messages", []),
                "error": "OpenAI API key not provided"
            }
        
        # Initialize OpenAI client
        model_name = state.get("model_name") or "gpt-3.5-turbo"
        temperature = state.get("temperature") or 0.7
        max_tokens = state.get("max_tokens")
        
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            organization=state.get("organization"),
            streaming=state.get("stream", False),
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
        
        # Extract usage information if available
        usage = None
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', None)
        
        # Create AI message
        ai_message = AIMessage(content=response.content)
        
        return {
            "messages": state.get("messages", []) + [ai_message],
            "response": response.content,
            "usage": usage,
            "tokens_used": usage.get('total_tokens') if usage else None,
            "finish_reason": "stop",
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
    except Exception as e:
        return {
            "messages": state.get("messages", []),
            "error": f"OpenAI API error: {str(e)}",
            "finish_reason": "error"
        } 