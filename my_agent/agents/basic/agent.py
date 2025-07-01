"""
Basic prompting agent implementation.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from my_agent.core.interfaces import ConfigurableAgent
from .state import BasicState
from .nodes import BasicPromptNode


class BasicAgent(ConfigurableAgent):
    """
    Basic prompting agent implementation.
    
    Simple prompt-response agent without tools or complex routing.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        # Initialize prompt node with configuration
        node_config = {
            "model_type": self.get_config("model_type", "ollama"),
            "model_name": self.get_config("model_name", "deepseek-r1:latest"),
            "template": self.get_config("template", "chat"),
            "system_message": self.get_config("system_message"),
            "temperature": self.get_config("temperature", 0.7)
        }
        
        # Remove None values
        node_config = {k: v for k, v in node_config.items() if v is not None}
        
        self.prompt_node = BasicPromptNode(**node_config)
    
    @property
    def agent_type(self) -> str:
        return "basic"
    
    def get_state_class(self) -> type:
        return BasicState
    
    def create_graph(self) -> StateGraph:
        """
        Create and configure the basic prompting agent graph.
        
        Simple linear flow: START → BasicPromptNode → END
        """
        # Create the graph
        workflow = StateGraph(BasicState)
        
        # Add single node
        workflow.add_node("prompt", self.prompt_node)
        
        # Add simple linear edges
        workflow.add_edge(START, "prompt")
        workflow.add_edge("prompt", END)
        
        # Add memory if configured (only for local use, not for LangGraph server)
        memory = self.get_config("memory")
        
        # Compile the graph
        if memory:
            return workflow.compile(checkpointer=memory)
        else:
            return workflow.compile()


def create_basic_agent(config: dict = None) -> StateGraph:
    """
    Factory function to create basic prompting agent.
    
    Args:
        config: Configuration dictionary with options:
            - model_type: "openai" or "ollama" (default: "ollama")
            - model_name: Specific model name
            - template: Template key (default: "chat")
            - system_message: Custom system message
            - temperature: Model temperature (default: 0.7)
            - memory: Custom memory implementation
    
    Returns:
        Compiled LangGraph agent
    
    Examples:
        # Basic usage
        agent = create_basic_agent()
        
        # With Ollama (default)
        agent = create_basic_agent({
            "model_type": "ollama",
            "model_name": "deepseek-r1:latest",
            "template": "summarize"
        })
        
        # With OpenAI
        agent = create_basic_agent({
            "model_type": "openai",
            "model_name": "gpt-4o-mini",
            "template": "code"
        })
        
        # Custom system message
        agent = create_basic_agent({
            "system_message": "You are a helpful coding assistant",
            "temperature": 0.3
        })
    """
    agent = BasicAgent(config)
    return agent.create_graph() 