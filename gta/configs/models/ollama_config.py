"""
Ollama-specific configuration schema.
"""

from typing import Optional
from gta.configs.models.base_model_config import BaseModelConfig


class OllamaConfig(BaseModelConfig):
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