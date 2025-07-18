"""
Unified chat configuration schema for all LLM providers.
"""

from typing import Optional, Dict, Any
from typing_extensions import TypedDict


class ChatConfig(TypedDict):
    """
    Configuration schema for chat/LLM functionality using Ollama.
    """
    
    # Provider selection (currently only Ollama supported)
    provider: Optional[str]            # "ollama" (default: "ollama")
    
    # Common LLM settings
    model_name: Optional[str]          # Model name for Ollama
    temperature: Optional[float]       # Generation temperature (0.0-1.0)
    max_tokens: Optional[int]          # Maximum tokens to generate
    system_prompt: Optional[str]       # System prompt for the conversation
    
    # Ollama-specific settings
    base_url: Optional[str]            # Ollama server URL (default: "http://localhost:11434")
    keep_alive: Optional[str]          # Keep model in memory duration
    num_ctx: Optional[int]             # Context window size
    num_predict: Optional[int]         # Number of tokens to predict
    repeat_penalty: Optional[float]    # Repetition penalty
    top_k: Optional[int]              # Top-k sampling
    top_p: Optional[float]            # Top-p sampling
    
    # Additional settings
    model_kwargs: Optional[Dict[str, Any]]  # Additional Ollama-specific parameters 