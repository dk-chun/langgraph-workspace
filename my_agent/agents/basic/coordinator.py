"""
Basic agent coordinator for managing multiple conversations.
"""

from typing import Dict, Any, Optional, List
from .basic_agent import BasicAgent


class BasicCoordinator:
    """
    Coordinates multiple basic agent instances.
    
    Manages sessions, routing, and conversation history.
    """
    
    def __init__(
        self,
        default_model: str = "qwen3:0.6b",
        default_temperature: float = 0.7,
        max_sessions: int = 100
    ):
        """
        Initialize the coordinator.
        
        Args:
            default_model: Default model for new agents
            default_temperature: Default temperature
            max_sessions: Maximum number of concurrent sessions
        """
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.max_sessions = max_sessions
        
        # Store active agents by session
        self.agents: Dict[str, BasicAgent] = {}
        self.session_configs: Dict[str, Dict[str, Any]] = {}
    
    def get_or_create_agent(
        self,
        session_id: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> BasicAgent:
        """
        Get existing agent or create new one for session.
        
        Args:
            session_id: Session identifier
            model_name: Model name override
            temperature: Temperature override
            system_prompt: System prompt override
            **kwargs: Additional agent parameters
            
        Returns:
            BasicAgent instance
        """
        
        if session_id not in self.agents:
            # Check session limit
            if len(self.agents) >= self.max_sessions:
                # Remove oldest session
                oldest_session = next(iter(self.agents))
                del self.agents[oldest_session]
                del self.session_configs[oldest_session]
            
            # Create new agent
            agent_config = {
                "model_name": model_name or self.default_model,
                "temperature": temperature or self.default_temperature,
                "system_prompt": system_prompt,
                **kwargs
            }
            
            self.agents[session_id] = BasicAgent(**agent_config)
            self.session_configs[session_id] = agent_config
        
        return self.agents[session_id]
    
    def chat(
        self,
        user_input: str,
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process chat message for a session.
        
        Args:
            user_input: User's message
            session_id: Session identifier
            **kwargs: Additional parameters
            
        Returns:
            Agent response
        """
        
        agent = self.get_or_create_agent(session_id, **kwargs)
        return agent.invoke(user_input, session_id, **kwargs)
    
    def stream_chat(
        self,
        user_input: str,
        session_id: str = "default",
        **kwargs
    ):
        """
        Stream chat response for a session.
        
        Args:
            user_input: User's message
            session_id: Session identifier
            **kwargs: Additional parameters
            
        Yields:
            Streaming updates
        """
        
        agent = self.get_or_create_agent(session_id, **kwargs)
        yield from agent.stream(user_input, session_id, **kwargs)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session configuration or None
        """
        return self.session_configs.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """
        List all active sessions.
        
        Returns:
            List of session IDs
        """
        return list(self.agents.keys())
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared
        """
        if session_id in self.agents:
            del self.agents[session_id]
            del self.session_configs[session_id]
            return True
        return False
    
    def clear_all_sessions(self):
        """Clear all sessions."""
        self.agents.clear()
        self.session_configs.clear() 