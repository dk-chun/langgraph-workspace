"""
Common formatting utilities.
"""

from typing import List, Dict, Any
from langchain_core.messages import BaseMessage


def format_messages(messages: List[BaseMessage]) -> str:
    """
    Format messages for display or logging.
    
    Args:
        messages: List of messages
        
    Returns:
        Formatted string representation
    """
    formatted = []
    for i, msg in enumerate(messages):
        role = msg.__class__.__name__.replace("Message", "")
        content = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
        formatted.append(f"{i+1}. {role}: {content}")
    
    return "\n".join(formatted)


def format_context(retrieved_docs: List[Dict[str, Any]], max_length: int = 2000) -> str:
    """
    Format retrieved documents into context string with length limit.
    
    Args:
        retrieved_docs: List of retrieved documents
        max_length: Maximum length of formatted context
        
    Returns:
        Formatted context string
    """
    if not retrieved_docs:
        return "No relevant documents found."
    
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(retrieved_docs, 1):
        doc_text = f"Document {i}:\n{doc['content']}\n\n"
        
        if current_length + len(doc_text) > max_length:
            # Truncate if needed
            remaining = max_length - current_length
            if remaining > 50:  # Only add if there's meaningful space
                truncated = doc_text[:remaining-3] + "..."
                context_parts.append(truncated)
            break
        
        context_parts.append(doc_text)
        current_length += len(doc_text)
    
    return "".join(context_parts).strip()


def format_agent_response(agent_type: str, response: str, metadata: Dict[str, Any] = None) -> str:
    """
    Format agent response with metadata.
    
    Args:
        agent_type: Type of agent
        response: Agent response
        metadata: Additional metadata
        
    Returns:
        Formatted response string
    """
    formatted = f"[{agent_type.upper()}] {response}"
    
    if metadata:
        meta_parts = []
        for key, value in metadata.items():
            meta_parts.append(f"{key}: {value}")
        
        if meta_parts:
            formatted += f"\n(Metadata: {', '.join(meta_parts)})"
    
    return formatted 