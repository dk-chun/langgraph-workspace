"""
RAG Agent Graph Definition with Runtime Configuration.
"""

from typing import List, Tuple, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import AIMessage, SystemMessage
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate

from gta.agents.simple_rag.state import RAGState
from gta.agents.simple_rag.config import SimpleRAGConfigSchema
from gta.agents.simple_rag.nodes import (
    create_extract_query_node,
    create_search_node_with_config,
    create_prompt_node_with_config,
    create_chat_node_with_config
)


def create_rag_graph_with_runtime_config() -> CompiledStateGraph:
    """
    Create RAG graph with runtime configuration support.
    
    This graph uses LangGraph's runtime configuration feature to dynamically
    configure LLM, embeddings, and vectorstore at execution time.
    
    Returns:
        Compiled RAG graph with runtime configuration schema
        
    Example:
        Basic usage with Ollama + Qdrant:
        
        ```python
        from langchain_core.messages import HumanMessage
        
        config = {
            "configurable": {
                "llm": {"provider": "ollama", "model": "qwen3:0.6b"},
                "vectorstore": {
                    "provider": "qdrant", 
                    "collection_name": "documents",
                    "url": "http://localhost:6333",
                    "embedding": {"provider": "ollama", "model": "bge-m3:latest"}
                },
                "top_k": 3
            }
        }
        
        graph = create_rag_graph_with_runtime_config()
        result = graph.invoke({"messages": [HumanMessage("What is AI?")]}, config)
        ```
        
        Mixed providers:
        
        ```python
        config = {
            "configurable": {
                "llm": {"provider": "openai", "model": "gpt-4", "api_key": "sk-..."},
                "vectorstore": {
                    "provider": "pinecone",
                    "index_name": "knowledge-base", 
                    "api_key": "pinecone-key",
                    "collection_name": "docs",
                    "embedding": {"provider": "openai", "model": "text-embedding-3-small", "api_key": "sk-..."}
                }
            }
        }
        ```
    """
    
    # Create nodes that accept runtime configuration
    extract_query = create_extract_query_node()
    search = create_search_node_with_config()
    prompt = create_prompt_node_with_config()
    generate = create_chat_node_with_config()
    
    # Build graph with config schema
    builder = StateGraph(RAGState, config_schema=SimpleRAGConfigSchema)
    
    # Add nodes
    builder.add_node("extract_query", extract_query)
    builder.add_node("search", search)
    builder.add_node("prompt", prompt)
    builder.add_node("generate", generate)
    
    # Add edges
    builder.add_edge(START, "extract_query")
    builder.add_edge("extract_query", "search")
    builder.add_edge("search", "prompt")
    builder.add_edge("prompt", "generate")
    builder.add_edge("generate", END)
    
    # Compile and return graph
    return builder.compile()


# Legacy function (static configuration)
def create_rag_graph(
    llm: Optional[BaseChatModel] = None,
    vectorstore: Optional[VectorStore] = None,
    template_messages: Optional[List[Tuple[str, str]]] = None,
    top_k: int = 5
) -> CompiledStateGraph:
    """
    Create RAG graph with static configuration (legacy).
    
    Args:
        llm: Language model instance (default: ChatOllama)
        vectorstore: Vector store instance (default: QdrantVectorStore)
        template_messages: Prompt template messages (default: RAG template)
        top_k: Number of documents to retrieve (default: 5)
        
    Returns:
        Compiled RAG graph
    """
    
    # Default LLM
    if llm is None:
        llm = ChatOllama(model="qwen3:0.6b", temperature=0.3)
    
    # Default vectorstore
    if vectorstore is None:
        embeddings = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url="http://localhost:11434"
        )
        client = QdrantClient(url="http://localhost:6333")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="test_langgraph",
            embedding=embeddings
        )
    
    # Default template
    if template_messages is None:
        template_messages = [
            ("system", "You are a helpful assistant. Use the following context to answer the user's question accurately and concisely."),
            ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ]
    
    # Legacy nodes (static config)
    def legacy_extract_query_node(state: RAGState) -> Dict[str, Any]:
        question = ""
        if state.messages:
            for msg in reversed(state.messages):
                if isinstance(msg, SystemMessage) or isinstance(msg, AIMessage):
                    question = msg.content
                    break
        return {"question": question}
    
    def legacy_search_node(state: RAGState) -> Dict[str, Any]:
        query = state.question
        if not query:
            return {"documents": [], "context": "No question provided."}
        
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        
        if not results:
            context = "No relevant documents found."
        else:
            context_parts = []
            for i, (doc, score) in enumerate(results):
                context_parts.append(f"[Document {i+1}] (Score: {score:.3f})\n{doc.page_content}")
            context = "\n\n".join(context_parts)
        
        return {"documents": results, "context": context}
    
    def legacy_prompt_node(state: RAGState) -> Dict[str, Any]:
        variables = {"context": state.context, "question": state.question}
        chat_template = ChatPromptTemplate(template_messages)
        prompt_value = chat_template.invoke(variables)
        messages = prompt_value.to_messages()
        return {"messages": messages}
    
    def legacy_chat_node(state: RAGState) -> Dict[str, Any]:
        messages = state.messages
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}
    
    # Build graph
    builder = StateGraph(RAGState)
    
    # Add nodes
    builder.add_node("extract_query", legacy_extract_query_node)
    builder.add_node("search", legacy_search_node)
    builder.add_node("prompt", legacy_prompt_node)
    builder.add_node("generate", legacy_chat_node)
    
    # Add edges
    builder.add_edge(START, "extract_query")
    builder.add_edge("extract_query", "search")
    builder.add_edge("search", "prompt")
    builder.add_edge("prompt", "generate")
    builder.add_edge("generate", END)
    
    # Compile and return graph
    return builder.compile()


# Main graph instance (runtime configuration)
simple_rag_graph = create_rag_graph_with_runtime_config()

# Legacy graph instance for backward compatibility
simple_rag_graph_legacy = create_rag_graph() 