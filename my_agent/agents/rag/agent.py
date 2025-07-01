"""
RAG agent implementation.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from my_agent.core.interfaces import ConfigurableAgent
from .state import RAGState
from .nodes import rag_retrieval_node, rag_generation_node
from .components import RAGComponents


class RAGAgent(ConfigurableAgent):
    """
    RAG-based agent implementation using Ollama.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        # Initialize RAG components with custom config if provided
        rag_config = self.get_config("rag_components", {})
        self.rag_components = RAGComponents(**rag_config)
    
    @property
    def agent_type(self) -> str:
        return "rag"
    
    def get_state_class(self) -> type:
        return RAGState
    
    def create_graph(self) -> StateGraph:
        """
        Create and configure the RAG agent graph.
        """
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieval", rag_retrieval_node)
        workflow.add_node("generation", rag_generation_node)
        
        # Add edges
        workflow.add_edge(START, "retrieval")
        workflow.add_edge("retrieval", "generation")
        workflow.add_edge("generation", END)
        
        # Add memory if configured (only for local use, not for LangGraph server)
        memory = self.get_config("memory")
        
        # Compile the graph
        if memory:
            return workflow.compile(checkpointer=memory)
        else:
            return workflow.compile()
    
    def add_documents(self, file_path: str):
        """
        Add documents to the RAG knowledge base.
        
        Args:
            file_path: Path to file or directory to add
        """
        self.rag_components.add_documents(file_path)


def create_rag_agent(config: dict = None) -> StateGraph:
    """
    Factory function to create RAG agent.
    """
    agent = RAGAgent(config)
    return agent.create_graph() 