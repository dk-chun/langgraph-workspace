"""
Node functions for the RAG agent.
"""

import os
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from my_agent.core.base_nodes import ProcessingNode
from my_agent.core.base_state import get_last_message_content, add_metadata, set_agent_type
from .state import RAGState
from .components import default_rag_components

# Load environment variables
load_dotenv()


class RAGRetrievalNode(ProcessingNode):
    """
    Document retrieval node for RAG agent.
    """
    
    def __init__(self, rag_components=None):
        super().__init__("rag_retrieval")
        self.rag_components = rag_components or default_rag_components
    
    def pre_process(self, state: RAGState) -> RAGState:
        """Set agent type before processing."""
        set_agent_type(state, "rag")
        return state
    
    def process(self, state: RAGState) -> Dict[str, Any]:
        """
        Retrieve relevant documents based on the user query.
        """
        # Use BaseState utility function to get last message content
        query = get_last_message_content(state)
        
        if not query:
            return {"retrieved_docs": [], "context": "", "num_docs_retrieved": 0}
        
        # Retrieve relevant documents
        retrieved_docs = self.rag_components.retrieve_documents(query, k=3)
        
        # Format context
        context = self.rag_components.format_context(retrieved_docs)
        
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "num_docs_retrieved": len(retrieved_docs),
            "retrieval_score": retrieved_docs[0]["score"] if retrieved_docs else 0.0
        }


class RAGGenerationNode(ProcessingNode):
    """
    Response generation node using Ollama model.
    """
    
    def __init__(self, 
                 base_url: str = None,
                 model_name: str = None,
                 temperature: float = 0.1):
        super().__init__("rag_generation")
        
        # Get configuration from environment or use defaults
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
        self.temperature = temperature
        
        # Initialize Ollama model
        self.llm = ChatOllama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=self.temperature
        )
    
    def pre_process(self, state: RAGState) -> RAGState:
        """Add metadata before processing."""
        add_metadata(state, "model", self.model_name)
        add_metadata(state, "temperature", self.temperature)
        return state
    
    def process(self, state: RAGState) -> Dict[str, Any]:
        """
        Generate response using Ollama model with retrieved context.
        """
        # Get query and context
        query = state.get("query", "")
        context = state.get("context", "")
        
        # Create prompt with context
        if context and context != "No relevant documents found.":
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer: Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please mention that."""
        else:
            prompt = f"""Question: {query}

Answer: I don't have specific context documents to reference for this question. I'll provide a general response based on my knowledge."""
        
        # Generate response
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "messages": [AIMessage(content=response.content)],
            "response": response.content
        }
    
    def post_process(self, state: RAGState, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add post-processing metadata."""
        result["generation_metadata"] = {
            "model": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature
        }
        return result


# Create node instances
rag_retrieval_node = RAGRetrievalNode()
rag_generation_node = RAGGenerationNode() 