"""
OpenAI agent implementation.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from my_agent.core.interfaces import ConfigurableAgent
from .state import OpenAIState
from .nodes import openai_agent_node, openai_tool_node, should_continue


class OpenAIAgent(ConfigurableAgent):
    """
    OpenAI-based agent implementation.
    """
    
    @property
    def agent_type(self) -> str:
        return "openai"
    
    def get_state_class(self) -> type:
        return OpenAIState
    
    def create_graph(self) -> StateGraph:
        """
        Create and configure the OpenAI agent graph.
        """
        # Create the graph
        workflow = StateGraph(OpenAIState)
        
        # Add nodes
        workflow.add_node("agent", openai_agent_node)
        workflow.add_node("tools", openai_tool_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Add memory if configured (only for local use, not for LangGraph server)
        memory = self.get_config("memory")
        
        # Compile the graph
        if memory:
            return workflow.compile(checkpointer=memory)
        else:
            return workflow.compile()


def create_openai_agent(config: dict = None) -> StateGraph:
    """
    Factory function to create OpenAI agent.
    """
    agent = OpenAIAgent(config)
    return agent.create_graph() 