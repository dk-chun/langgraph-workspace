"""
Node functions for the OpenAI agent.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from my_agent.core.base_nodes import ProcessingNode
from my_agent.core.base_state import add_metadata, set_agent_type
from .state import OpenAIState
from .tools import OPENAI_TOOLS


class OpenAIAgentNode(ProcessingNode):
    """
    Main agent node for OpenAI-based processing.
    """
    
    def __init__(self):
        super().__init__("openai_agent")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.llm_with_tools = self.llm.bind_tools(OPENAI_TOOLS)
    
    def pre_process(self, state: OpenAIState) -> OpenAIState:
        """Set agent type and add metadata before processing."""
        set_agent_type(state, "openai")
        add_metadata(state, "model", "gpt-4o-mini")
        return state
    
    def process(self, state: OpenAIState) -> Dict[str, Any]:
        """
        Process messages with OpenAI model.
        """
        messages = state["messages"]
        
        # Generate response
        response = self.llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "current_step": "agent_response"
        }
    
    def post_process(self, state: OpenAIState, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add post-processing metadata."""
        result["processing_metadata"] = {
            "node_name": self.name,
            "model": "gpt-4o-mini",
            "tools_available": len(OPENAI_TOOLS)
        }
        return result


class OpenAIToolNode(ProcessingNode):
    """
    Tool execution node for OpenAI agent.
    """
    
    def __init__(self):
        super().__init__("openai_tools")
        self.tool_node = ToolNode(OPENAI_TOOLS)
    
    def pre_process(self, state: OpenAIState) -> OpenAIState:
        """Add tool execution metadata."""
        add_metadata(state, "tools_executing", True)
        return state
    
    def process(self, state: OpenAIState) -> Dict[str, Any]:
        """
        Execute tools and return results.
        """
        # ToolNode is callable, invoke it directly
        return self.tool_node.invoke(state)
    
    def post_process(self, state: OpenAIState, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add tool execution results metadata."""
        result["tool_metadata"] = {
            "node_name": self.name,
            "available_tools": [tool.name for tool in OPENAI_TOOLS]
        }
        return result


def should_continue(state: OpenAIState) -> str:
    """
    Conditional edge function to determine next step.
    
    Args:
        state: Current state
        
    Returns:
        Next node name
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, go to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"


# Create node instances
openai_agent_node = OpenAIAgentNode()
openai_tool_node = OpenAIToolNode() 