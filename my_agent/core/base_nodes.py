"""
Base node functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from .base_state import BaseState


class BaseNode(ABC):
    """
    Abstract base class for all nodes.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, state: BaseState) -> Dict[str, Any]:
        """Execute the node logic."""
        pass
    
    def __call__(self, state: BaseState) -> Dict[str, Any]:
        """Make the node callable."""
        return self.execute(state)


class ProcessingNode(BaseNode):
    """
    Base class for processing nodes.
    """
    
    def pre_process(self, state: BaseState) -> BaseState:
        """Pre-processing hook."""
        return state
    
    def post_process(self, state: BaseState, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-processing hook."""
        return result
    
    def execute(self, state: BaseState) -> Dict[str, Any]:
        """Execute with pre/post processing."""
        state = self.pre_process(state)
        result = self.process(state)
        return self.post_process(state, result)
    
    @abstractmethod
    def process(self, state: BaseState) -> Dict[str, Any]:
        """Main processing logic."""
        pass 