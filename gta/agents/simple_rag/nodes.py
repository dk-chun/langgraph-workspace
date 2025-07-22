"""
RAG Agent Nodes with Runtime Configuration.
"""

import functools
from typing import Dict, Any, List, Tuple, Callable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from gta.agents.simple_rag.state import RAGState
from gta.agents.simple_rag.config import DEFAULT_SIMPLE_RAG_CONFIG
from gta.core.factories import create_llm, create_vectorstore


def _get_runtime_config(config: RunnableConfig) -> Dict[str, Any]:
    """Get runtime configuration with fallback to defaults."""
    if not config or "configurable" not in config:
        return DEFAULT_SIMPLE_RAG_CONFIG
    
    runtime_config = config["configurable"]
    
    # Merge with defaults for missing keys
    merged_config = {}
    merged_config.update(DEFAULT_SIMPLE_RAG_CONFIG)
    merged_config.update(runtime_config)
    
    return merged_config


def extract_query_node(state: RAGState) -> Dict[str, Any]:
    """Extract query from messages for search."""
    # Get the last user message as query
    question = ""
    if state.messages:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break
    
    return {"question": question}


def search_node(state: RAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Search documents and format context using runtime configuration."""
    query = state.question
    if not query:
        return {
            "documents": [],
            "context": "No question provided."
        }
    
    # Get configuration with defaults
    runtime_config = _get_runtime_config(config)
    vs_config = runtime_config["vectorstore"]
    top_k = runtime_config.get("top_k", 5)
    
    # Create vectorstore dynamically
    vectorstore = create_vectorstore(vs_config)
    
    # Search documents
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    # Format search results
    if not results:
        context = "No relevant documents found."
    else:
        context_parts = []
        for i, (doc, score) in enumerate(results):
            context_parts.append(f"[Document {i+1}] (Score: {score:.3f})\n{doc.page_content}")
        context = "\n\n".join(context_parts)
    
    return {
        "documents": results,
        "context": context
    }


def prompt_node(state: RAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Format prompt with context and question using runtime configuration."""
    variables = {
        "context": state.context,
        "question": state.question
    }
    
    # Get configuration with defaults
    runtime_config = _get_runtime_config(config)
    template_messages = runtime_config.get("template_messages", [
        ("system", "You are a helpful assistant. Use the following context to answer the user's question accurately and concisely."),
        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    
    # Format prompt template
    chat_template = ChatPromptTemplate(template_messages)
    prompt_value = chat_template.invoke(variables)
    messages = prompt_value.to_messages()
    
    return {"messages": messages}


def chat_node(state: RAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate response using LLM from runtime configuration."""
    messages = state.messages
    
    # Get configuration with defaults
    runtime_config = _get_runtime_config(config)
    llm_config = runtime_config["llm"]
    llm = create_llm(llm_config)
    
    response = llm.invoke(messages)
    
    return {"messages": [AIMessage(content=response.content)]}


# Node factory functions
def create_extract_query_node() -> Callable[[RAGState], Dict[str, Any]]:
    """Create query extraction node."""
    return extract_query_node


def create_search_node_with_config() -> Callable[[RAGState, RunnableConfig], Dict[str, Any]]:
    """Create search node that uses runtime configuration."""
    return search_node


def create_prompt_node_with_config() -> Callable[[RAGState, RunnableConfig], Dict[str, Any]]:
    """Create prompt formatting node that uses runtime configuration."""
    return prompt_node


def create_chat_node_with_config() -> Callable[[RAGState, RunnableConfig], Dict[str, Any]]:
    """Create chat node that uses runtime configuration."""
    return chat_node 