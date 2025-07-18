"""
RAG (Retrieval-Augmented Generation) agent implementation.
Combines Qdrant vector search with LLM response generation.
"""

from typing import Any, Dict, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from gta.states.messages_state import MessagesState
from gta.states.vectorstores.qdrant_state import QdrantState
from gta.configs.vectorstores.qdrant_config import QdrantConfig
from gta.configs.chat_config import ChatConfig
from gta.nodes.vectorstores.qdrant_node import qdrant_node
from gta.nodes.chat_node import chat_node


class RAGState(MessagesState):
    """
    Enhanced state for RAG operations.
    Combines message handling with vector search capabilities.
    """
    
    # Vector search fields
    query: str = ""
    search_results: List[Dict[str, Any]] = []
    context: str = ""
    
    # Operation control
    use_search: bool = True
    search_top_k: int = 5
    context_max_length: int = 2000


def create_rag_agent() -> Any:
    """
    Create a RAG agent with Qdrant search and LLM response.
    
    Flow:
    1. Extract query from user message
    2. Search relevant documents in Qdrant
    3. Format context from search results
    4. Generate LLM response with context
    
    Returns:
        Compiled LangGraph application
    """
    
    # Create the graph with RAG state and combined config
    workflow = StateGraph(RAGState, config_schema=QdrantConfig)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("llm_response", llm_response_node)
    
    # Define the flow
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "format_context")
    workflow.add_edge("format_context", "llm_response")
    workflow.add_edge("llm_response", END)
    
    # Compile with memory
    return workflow.compile(name="rag_agent")


def search_node(state: RAGState, config) -> Dict[str, Any]:
    """
    Extract query from messages and perform vector search.
    """
    
    # Extract query from the last human message
    query = ""
    if state.messages:
        last_message = state.messages[-1]
        if hasattr(last_message, 'content'):
            query = last_message.content
    
    if not query or not state.use_search:
        return {
            "query": query,
            "search_results": [],
            "context": ""
        }
    
    # Create Qdrant state for search
    qdrant_state = QdrantState(
        operation="search",
        query=query
    )
    
    # Perform search using qdrant_node
    search_result = qdrant_node(qdrant_state, config)
    
    return {
        "query": query,
        "search_results": search_result.get("search_results", []),
        "context": ""  # Will be formatted in next node
    }


def format_context_node(state: RAGState, config) -> Dict[str, Any]:
    """
    Format search results into context for LLM.
    """
    
    if not state.search_results:
        return {"context": ""}
    
    # Format search results into context
    context_parts = []
    current_length = 0
    max_length = state.context_max_length
    
    for i, result in enumerate(state.search_results[:state.search_top_k]):
        doc = result.get("document", {})
        content = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        score = result.get("score", 0.0)
        
        # Format context entry
        context_entry = f"[Document {i+1}] (Score: {score:.3f})\n{content}\n"
        
        # Add metadata if available
        if metadata:
            context_entry += f"Metadata: {metadata}\n"
        
        context_entry += "\n"
        
        # Check length limit
        if current_length + len(context_entry) > max_length:
            break
            
        context_parts.append(context_entry)
        current_length += len(context_entry)
    
    context = "".join(context_parts)
    
    return {"context": context}


def llm_response_node(state: RAGState, config) -> Dict[str, Any]:
    """
    Generate LLM response using retrieved context.
    """
    
    # Prepare messages with context
    messages = list(state.messages)
    
    # Add system message with context if available
    if state.context:
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context from knowledge base:
{state.context}

Instructions:
- Answer the user's question using the provided context
- If the context doesn't contain relevant information, say so clearly
- Be accurate and cite specific information from the context when possible
- If asked about something not in the context, indicate that you don't have that information
"""
        
        # Insert system message at the beginning
        messages.insert(0, SystemMessage(content=system_prompt))
    
    # Create a temporary messages state for LLM
    llm_state = MessagesState(messages=messages)
    
    # Call LLM node
    llm_result = chat_node(llm_state, config)
    
    # Return the updated messages
    return {
        "messages": llm_result.get("messages", messages)
    }


def create_simple_rag_agent() -> Any:
    """
    Create a simpler RAG agent for basic use cases.
    
    This version uses a single node that combines search and response.
    """
    
    workflow = StateGraph(RAGState, config_schema=QdrantConfig)
    
    # Add single combined node
    workflow.add_node("rag", combined_rag_node)
    
    # Simple flow
    workflow.add_edge(START, "rag")
    workflow.add_edge("rag", END)
    
    return workflow.compile(name="simple_rag_agent")


def combined_rag_node(state: RAGState, config) -> Dict[str, Any]:
    """
    Combined RAG node that performs search and generates response.
    """
    
    # Extract query
    query = ""
    if state.messages:
        last_message = state.messages[-1]
        if hasattr(last_message, 'content'):
            query = last_message.content
    
    # Perform search if enabled
    context = ""
    if query and state.use_search:
        qdrant_state = QdrantState(operation="search", query=query)
        search_result = qdrant_node(qdrant_state, config)
        
        # Format context
        search_results = search_result.get("search_results", [])
        if search_results:
            context_parts = []
            for i, result in enumerate(search_results[:state.search_top_k]):
                doc = result.get("document", {})
                content = doc.get("page_content", "")
                score = result.get("score", 0.0)
                context_parts.append(f"[{i+1}] (Score: {score:.3f}) {content}")
            
            context = "\n\n".join(context_parts)
    
    # Prepare messages with context
    messages = list(state.messages)
    
    if context:
        system_prompt = f"""Answer the user's question using the following context:

{context}

Be accurate and cite the context when possible."""
        
        messages.insert(0, SystemMessage(content=system_prompt))
    
    # Generate response
    llm_state = MessagesState(messages=messages)
    llm_result = chat_node(llm_state, config)
    
    return {
        "messages": llm_result.get("messages", messages),
        "query": query,
        "context": context
    } 