"""
State definition for the OpenAI agent.
"""

from typing import List, Optional
from my_agent.core.base_state import BaseState


class OpenAIState(BaseState):
    """
    State for the OpenAI agent graph.
    
    Extends BaseState with OpenAI-specific state variables.
    """
    # OpenAI specific state
    current_step: Optional[str] = None
    context: Optional[dict] = None
    tool_calls: List[dict] = [] 