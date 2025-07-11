"""
Basic agent service for LangGraph server deployment.
"""

from typing import Dict, Any, Optional
from .basic_agent import BasicAgent
from .coordinator import BasicCoordinator


class BasicService:
    """
    Service wrapper for basic agent deployment.
    
    Provides standardized interface for LangGraph server.
    """
    
    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        base_url: Optional[str] = None,
        use_coordinator: bool = True,
        max_sessions: int = 100
    ):
        """
        Initialize the service.
        
        Args:
            model_name: Ollama model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt for the model
            base_url: Ollama server base URL
            use_coordinator: Whether to use coordinator for multi-session
            max_sessions: Maximum concurrent sessions
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.use_coordinator = use_coordinator
        
        if use_coordinator:
            self.coordinator = BasicCoordinator(
                default_model=model_name,
                default_temperature=temperature,
                max_sessions=max_sessions
            )
        else:
            self.agent = BasicAgent(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                base_url=base_url
            )
    
    def process_message(
        self,
        user_input: str,
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user message.
        
        Args:
            user_input: User's message
            session_id: Session identifier
            **kwargs: Additional parameters
            
        Returns:
            Agent response
        """
        
        if self.use_coordinator:
            return self.coordinator.chat(
                user_input=user_input,
                session_id=session_id,
                **kwargs
            )
        else:
            return self.agent.invoke(
                user_input=user_input,
                session_id=session_id,
                **kwargs
            )
    
    def stream_message(
        self,
        user_input: str,
        session_id: str = "default",
        **kwargs
    ):
        """
        Stream a user message response.
        
        Args:
            user_input: User's message
            session_id: Session identifier
            **kwargs: Additional parameters
            
        Yields:
            Streaming updates
        """
        
        if self.use_coordinator:
            yield from self.coordinator.stream_chat(
                user_input=user_input,
                session_id=session_id,
                **kwargs
            )
        else:
            yield from self.agent.stream(
                user_input=user_input,
                session_id=session_id,
                **kwargs
            )
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session info or None
        """
        if self.use_coordinator:
            return self.coordinator.get_session_info(session_id)
        else:
            return {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "system_prompt": self.system_prompt
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint.
        
        Returns:
            Service health status
        """
        return {
            "status": "healthy",
            "service": "basic_agent",
            "model": self.model_name,
            "coordinator_enabled": self.use_coordinator,
            "active_sessions": len(self.coordinator.agents) if self.use_coordinator else 1
        }


def create_basic_service(
    model_name: str = "qwen3:0.6b",
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    **kwargs
) -> BasicService:
    """
    Factory function to create a basic service.
    
    Args:
        model_name: Ollama model name
        temperature: Generation temperature
        system_prompt: System prompt
        **kwargs: Additional service parameters
        
    Returns:
        BasicService instance
    """
    return BasicService(
        model_name=model_name,
        temperature=temperature,
        system_prompt=system_prompt,
        **kwargs
    ) 