"""
RAG Agent Nodes.
"""

import functools
from typing import Dict, Any, List, Tuple, Callable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate

from gta.agents.simple_rag.state import RAGState


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


def search_node(state: RAGState, vectorstore: VectorStore, top_k: int = 5) -> Dict[str, Any]:
    """Search documents and format context."""
    query = state.question
    if not query:
        return {
            "documents": [],
            "context": "No question provided."
        }
    
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


def prompt_node(state: RAGState, template_messages: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Format prompt with context and question."""
    variables = {
        "context": state.context,
        "question": state.question
    }
    
    # Format prompt template
    chat_template = ChatPromptTemplate(template_messages)
    prompt_value = chat_template.invoke(variables)
    messages = prompt_value.to_messages()
    
    return {"messages": messages}


def chat_node(state: RAGState, llm: BaseChatModel) -> Dict[str, Any]:
    """Generate response using LLM."""
    messages = state.messages
    response = llm.invoke(messages)
    
    return {"messages": [AIMessage(content=response.content)]}


# Node factory functions
def create_extract_query_node() -> Callable:
    """Create query extraction node."""
    return extract_query_node


def create_search_node(vectorstore: VectorStore, top_k: int = 5) -> Callable:
    """Create search node with vectorstore."""
    return functools.partial(search_node, vectorstore=vectorstore, top_k=top_k)


def create_prompt_node(template_messages: List[Tuple[str, str]]) -> Callable:
    """Create prompt formatting node."""
    return functools.partial(prompt_node, template_messages=template_messages)


def create_chat_node(llm: BaseChatModel) -> Callable:
    """Create chat node with LLM."""
    return functools.partial(chat_node, llm=llm) 