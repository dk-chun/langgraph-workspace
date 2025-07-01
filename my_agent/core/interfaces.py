"""
Interface definitions for agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from langgraph.graph import StateGraph
from .base_state import BaseState


class AgentInterface(ABC):
    """
    Interface for all agents.
    """
    
    @abstractmethod
    def create_graph(self) -> StateGraph:
        """Create and return the agent graph."""
        pass
    
    @abstractmethod
    def get_state_class(self) -> type:
        """Return the state class for this agent."""
        pass
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        pass


class ConfigurableAgent(AgentInterface):
    """
    Base class for configurable agents.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default) 