"""
Base model configuration schema.
"""

from typing import Optional, Dict, Any
from typing_extensions import TypedDict


class BaseModelConfig(TypedDict):
    """
    Base model configuration schema.
    
    Used for runtime configuration of LLM models.
    """
    
    model_name: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    system_prompt: Optional[str]
    model_kwargs: Optional[Dict[str, Any]] 