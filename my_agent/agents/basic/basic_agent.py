"""
Basic agent implementation using LangGraph.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from my_agent.states.chat.chat_input import ChatInputState
from my_agent.states.chat.chat_output import ChatOutputState
from my_agent.states.models.ollama import OllamaState
from my_agent.nodes.chat.chat_input import chat_input_node
from my_agent.nodes.chat.chat_output import chat_output_node
from my_agent.nodes.models.ollama import ollama_node


class BasicAgent:
    """
    Basic conversational agent using Ollama.
    
    Implements a simple 3-node graph:
    ChatInput -> Ollama -> ChatOutput
    """
    
    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the basic agent.
        
        Args:
            model_name: Ollama model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt for the model
            base_url: Ollama server base URL
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.base_url = base_url
        
        # Create the graph
        self.graph = self._create_graph()
        
        # Compile with memory
        self.app = self.graph.compile(checkpointer=MemorySaver())
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        
        # Define the overall state schema
        class BasicAgentState(ChatInputState, ChatOutputState, OllamaState):
            """Combined state for the basic agent."""
            pass
        
        # Create the graph
        workflow = StateGraph(BasicAgentState)
        
        # Add nodes
        workflow.add_node("chat_input", chat_input_node)
        workflow.add_node("ollama", ollama_node)
        workflow.add_node("chat_output", chat_output_node)
        
        # Define the flow
        workflow.set_entry_point("chat_input")
        workflow.add_edge("chat_input", "ollama")
        workflow.add_edge("ollama", "chat_output")
        workflow.add_edge("chat_output", END)
        
        return workflow
    
    def invoke(
        self,
        user_input: str,
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user input and return the response.
        
        Args:
            user_input: User's message
            session_id: Session identifier
            **kwargs: Additional parameters
            
        Returns:
            Agent response
        """
        
        # Prepare initial state
        initial_state = {
            "user_input": user_input,
            "session_id": session_id,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "base_url": self.base_url,
            "messages": []
        }
        
        # Add any additional kwargs
        initial_state.update(kwargs)
        
        # Execute the graph
        config = {"configurable": {"thread_id": session_id}}
        result = self.app.invoke(initial_state, config)
        
        return result
    
    def stream(
        self,
        user_input: str,
        session_id: str = "default",
        **kwargs
    ):
        """
        Stream the agent's response.
        
        Args:
            user_input: User's message
            session_id: Session identifier
            **kwargs: Additional parameters
            
        Yields:
            Streaming updates
        """
        
        # Prepare initial state
        initial_state = {
            "user_input": user_input,
            "session_id": session_id,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "base_url": self.base_url,
            "messages": []
        }
        
        # Add any additional kwargs
        initial_state.update(kwargs)
        
        # Stream the graph execution
        config = {"configurable": {"thread_id": session_id}}
        for update in self.app.stream(initial_state, config):
            yield update 