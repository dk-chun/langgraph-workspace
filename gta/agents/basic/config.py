"""
Basic Agent Configuration Schema.
"""

from typing import Optional, List, Tuple
from typing_extensions import TypedDict


class LLMConfig(TypedDict):
    """LLM provider configuration."""
    provider: str  # "ollama", "openai", "anthropic"
    model: str
    temperature: Optional[float]
    base_url: Optional[str]  # For Ollama
    api_key: Optional[str]   # For OpenAI/Anthropic


class BasicConfigSchema(TypedDict):
    """Complete Basic Agent configuration schema."""
    llm: LLMConfig
    system_prompt: Optional[str]


# Default configurations
DEFAULT_OLLAMA_LLM = LLMConfig(
    provider="ollama",
    model="qwen3:0.6b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

DEFAULT_BASIC_CONFIG = BasicConfigSchema(
    llm=DEFAULT_OLLAMA_LLM,
    system_prompt="You are a helpful assistant."
) 