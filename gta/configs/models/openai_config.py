"""
OpenAI-specific configuration schema.
"""

from typing import Optional, List
from gta.configs.models.base_model_config import BaseModelConfig


class OpenAIConfig(BaseModelConfig):
    """
    OpenAI-specific configuration schema.
    
    Extends ModelConfig with OpenAI-specific settings.
    """
    
    # API configuration
    api_key: Optional[str]
    organization: Optional[str]
    
    # Generation parameters
    stream: Optional[bool]
    functions: Optional[List[dict]]
    function_call: Optional[str] 