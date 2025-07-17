"""
Model configuration schemas for runtime configuration.
"""

from typing import Optional, Dict, Any
from typing_extensions import TypedDict


class ModelConfig(TypedDict):
    """
    Base model configuration schema.
    
    Used for runtime configuration of LLM models.
    """
    
    model_name: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    system_prompt: Optional[str]
    model_kwargs: Optional[Dict[str, Any]]


class OllamaConfig(ModelConfig):
    """
    Ollama-specific configuration schema.
    
    Extends ModelConfig with Ollama-specific settings.
    """
    
    # Server configuration
    base_url: Optional[str]
    keep_alive: Optional[str]
    
    # Generation parameters
    num_ctx: Optional[int]
    num_predict: Optional[int]
    repeat_penalty: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float] 