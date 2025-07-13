"""
Model-related node functions.
"""

from .openai import openai_node
from .ollama import ollama_node

__all__ = ["openai_node", "ollama_node"] 