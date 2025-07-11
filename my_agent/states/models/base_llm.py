"""
Base LLM state definition.
"""

from typing import Optional, Dict, Any
from pydantic import Field
from langgraph.graph import MessagesState


class BaseLLMState(MessagesState):
    """
    Base state for LLM operations.
    
    Provides common fields for all LLM-based nodes.
    """
    
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the LLM model being used"
    )
    
    temperature: Optional[float] = Field(
        default=0.7,
        description="Temperature parameter for generation"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for the model"
    )
    
    model_kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional model-specific parameters"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    ) 