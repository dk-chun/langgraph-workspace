"""
OpenAI-specific state definition.
"""

from typing import Optional, List
from pydantic import Field
from gta.states.models.base_llm_state import BaseLLMState


class OpenAIState(BaseLLMState):
    """
    State for OpenAI LLM operations.
    
    Extends BaseLLMState with OpenAI-specific fields.
    """
    
    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID"
    )
    
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream responses"
    )
    
    functions: Optional[List[dict]] = Field(
        default_factory=list,
        description="Available functions for function calling"
    )
    
    function_call: Optional[str] = Field(
        default=None,
        description="Function call mode (auto, none, or function name)"
    )
    
    usage: Optional[dict] = Field(
        default=None,
        description="Token usage information from OpenAI"
    ) 