"""
Ollama-specific state definition.
"""

from typing import Optional, Dict, Any
from pydantic import Field
from gta.states.models.base_llm_state import BaseLLMState


class OllamaState(BaseLLMState):
    """
    State for Ollama LLM operations.
    
    Extends BaseLLMState with Ollama-specific fields.
    """
    
    base_url: Optional[str] = Field(
        default="http://localhost:11434",
        description="Ollama server base URL"
    )
    
    timeout: Optional[int] = Field(
        default=60,
        description="Request timeout in seconds"
    )
    
    keep_alive: Optional[str] = Field(
        default=None,
        description="Keep model loaded (e.g., '5m', '10s')"
    )
    
    format: Optional[str] = Field(
        default=None,
        description="Response format (e.g., 'json')"
    )
    
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional Ollama options"
    )
    
    num_ctx: Optional[int] = Field(
        default=None,
        description="Context window size"
    )
    
    num_predict: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to predict"
    )
    
    repeat_penalty: Optional[float] = Field(
        default=None,
        description="Penalty for repeating tokens"
    )
    
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k sampling parameter"
    )
    
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling parameter"
    ) 