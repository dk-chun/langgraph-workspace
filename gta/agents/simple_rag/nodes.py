"""
RAG Agent Adapter Nodes.
"""

import functools
from typing import Dict, Any, List, Tuple, Callable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from gta.core.chat import invoke_llm
from gta.core.vectorstore import search_documents, format_search_results
from gta.core.prompt import format_prompt_template
from gta.agents.simple_rag.state import RAGState


def _extract_query_adapter(state: RAGState) -> Dict[str, Any]:
    """Extract query from messages for search."""
    # Get the last user message as query
    question = ""
    if state.messages:  # Fixed: Pydantic model access
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break
    
    return {"question": question}


def _search_adapter(state: RAGState, vectorstore: VectorStore, top_k: int = 5) -> Dict[str, Any]:
    """Search documents and format context."""
    query = state.question
    if not query:
        return {
            "documents": [],
            "context": "No question provided."
        }
    
    # Search documents
    results = search_documents(query, vectorstore, top_k)
    context = format_search_results(results)
    
    return {
        "documents": results,
        "context": context
    }


def _prompt_adapter(state: RAGState, template_messages: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Format prompt with context and question."""
    variables = {
        "context": state.context,
        "question": state.question
    }
    
    messages = format_prompt_template(template_messages, variables)
    
    return {"messages": messages}


def _chat_adapter(state: RAGState, llm: BaseChatModel) -> Dict[str, Any]:
    """Generate response using LLM."""
    messages = state.messages
    response_content = invoke_llm(messages, llm)
    
    return {"messages": [AIMessage(content=response_content)]}


# Node factory functions
def create_extract_query_node() -> Callable:
    """Create query extraction node."""
    return _extract_query_adapter


def create_search_node(vectorstore: VectorStore, top_k: int = 5) -> Callable:
    """Create search node with vectorstore."""
    return functools.partial(_search_adapter, vectorstore=vectorstore, top_k=top_k)


def create_prompt_node(template_messages: List[Tuple[str, str]]) -> Callable:
    """Create prompt formatting node."""
    return functools.partial(_prompt_adapter, template_messages=template_messages)


def create_chat_node(llm: BaseChatModel) -> Callable:
    """Create chat node with LLM."""
    return functools.partial(_chat_adapter, llm=llm) 