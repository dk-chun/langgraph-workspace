"""
Model configuration schemas.
"""

from gta.configs.models.base_model_config import BaseModelConfig
from gta.configs.models.ollama_config import OllamaConfig
from gta.configs.models.openai_config import OpenAIConfig

__all__ = [
    "BaseModelConfig",
    "OllamaConfig", 
    "OpenAIConfig"
] 