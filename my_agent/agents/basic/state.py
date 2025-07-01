"""
State definition for the basic prompting agent.
"""

from typing import Optional
from my_agent.core.base_state import BaseState


class BasicState(BaseState):
    """
    State for the basic prompting agent graph.
    
    Extends BaseState with basic prompting-specific state variables.
    """
    # Prompting configuration
    prompt_template: Optional[str] = None
    system_message: Optional[str] = None
    response_format: Optional[str] = None
    
    # Processing metadata
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None 