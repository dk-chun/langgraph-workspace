"""
Multi-VectorStore RAG Agent Nodes with Runtime Configuration.
"""

import functools
from typing import Dict, Any, List, Tuple, Callable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from gta.agents.multi_rag.state import MultiRAGState
from gta.agents.multi_rag.config import DEFAULT_MULTI_RAG_CONFIG
from gta.core.factories import create_llm, create_vectorstore


def _get_runtime_config(config: RunnableConfig) -> Dict[str, Any]:
    """Get runtime configuration with fallback to defaults."""
    if not config or "configurable" not in config:
        return DEFAULT_MULTI_RAG_CONFIG
    
    runtime_config = config["configurable"]
    
    # Merge with defaults for missing keys
    merged_config = {}
    merged_config.update(DEFAULT_MULTI_RAG_CONFIG)
    merged_config.update(runtime_config)
    
    return merged_config


def extract_query_node(state: MultiRAGState) -> Dict[str, Any]:
    """Extract query from messages for search."""
    question = ""
    if state.messages:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break
    
    return {"question": question}


def individual_vectorstore_search_node(
    state: MultiRAGState, 
    config: RunnableConfig,
    store_index: int
) -> Dict[str, Any]:
    """Individual vectorstore search node using runtime configuration."""
    query = state.question
    if not query:
        store_id = f"vs{store_index + 1}"
        return {f"{store_id}_documents": []}
    
    # Get configuration with defaults
    runtime_config = _get_runtime_config(config)
    vs_configs = runtime_config["vectorstores"]
    top_k = runtime_config.get("top_k_per_store", 3)
    
    # Create specific vectorstore
    if store_index >= len(vs_configs):
        store_id = f"vs{store_index + 1}"
        return {f"{store_id}_documents": []}
    
    vectorstore = create_vectorstore(vs_configs[store_index])
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    # Calculate search quality score
    search_score = 0.0
    if results:
        search_score = sum(score for _, score in results) / len(results)
    
    store_id = f"vs{store_index + 1}"
    return {
        f"{store_id}_documents": results,
        "search_scores": {store_id: search_score}
    }


def _normalize_scores(results: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    """Normalize scores using Min-Max normalization."""
    if not results:
        return []
    
    scores = [score for _, score in results]
    min_score, max_score = min(scores), max(scores)
    
    # Avoid division by zero
    if max_score == min_score:
        return [(doc, 1.0) for doc, _ in results]
    
    normalized = []
    for doc, score in results:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((doc, norm_score))
    
    return normalized


def _remove_duplicates(results: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    """Remove duplicate documents based on content similarity."""
    unique_results = []
    seen_contents = set()
    
    for doc, score in results:
        content = doc.page_content
        # Simple duplicate detection by exact content match
        if content not in seen_contents:
            unique_results.append((doc, score))
            seen_contents.add(content)
    
    return unique_results


def _format_search_results(results: List[Tuple[Any, float]]) -> str:
    """Format search results into context string."""
    if not results:
        return "No relevant documents found."
    
    context_parts = []
    for i, (doc, score) in enumerate(results):
        context_parts.append(f"[Document {i+1}] (Score: {score:.3f})\n{doc.page_content}")
    
    return "\n\n".join(context_parts)


def merge_results_node(state: MultiRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Merge results from all vectorstores using runtime configuration."""
    vs1_results = state.vs1_documents
    vs2_results = state.vs2_documents
    vs3_results = state.vs3_documents
    
    # Get configuration with defaults
    runtime_config = _get_runtime_config(config)
    strategy = runtime_config.get("merge_strategy", "simple")
    final_top_k = runtime_config.get("final_top_k", 5)
    
    if strategy == "simple":
        # Normalize scores for each vectorstore
        norm_vs1 = _normalize_scores(vs1_results)
        norm_vs2 = _normalize_scores(vs2_results)
        norm_vs3 = _normalize_scores(vs3_results)
        
        # Combine all results
        all_results = norm_vs1 + norm_vs2 + norm_vs3
        
        # Sort by normalized score
        sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        
        # Remove duplicates
        unique_results = _remove_duplicates(sorted_results)
        
        # Select top_k
        merged_documents = unique_results[:final_top_k]
        
    else:
        # Default to simple strategy
        merged_documents = []
    
    # Format final context
    final_context = _format_search_results(merged_documents)
    
    return {
        "merged_documents": merged_documents,
        "final_context": final_context,
        "merge_strategy": strategy
    }


def prompt_node(state: MultiRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Format prompt with merged context and question using runtime configuration."""
    variables = {
        "context": state.final_context,
        "question": state.question
    }
    
    # Get configuration with defaults
    runtime_config = _get_runtime_config(config)
    template_messages = runtime_config.get("template_messages", [
        ("system", "You are a helpful assistant. Use the following context from multiple sources to answer the user's question accurately and concisely."),
        ("user", "Context from multiple sources:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    
    # Format prompt template
    chat_template = ChatPromptTemplate(template_messages)
    prompt_value = chat_template.invoke(variables)
    messages = prompt_value.to_messages()
    
    return {"messages": messages}


def chat_node(state: MultiRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate response using LLM from runtime configuration."""
    messages = state.messages
    
    # Get configuration with defaults
    runtime_config = _get_runtime_config(config)
    llm_config = runtime_config["llm"]
    llm = create_llm(llm_config)
    
    response = llm.invoke(messages)
    
    return {"messages": [AIMessage(content=response.content)]}


# Node factory functions
def create_extract_query_node() -> Callable[[MultiRAGState], Dict[str, Any]]:
    """Create query extraction node."""
    return extract_query_node


def create_individual_search_node_with_config(store_index: int) -> Callable[[MultiRAGState, RunnableConfig], Dict[str, Any]]:
    """Create individual vectorstore search node that uses runtime configuration."""
    return functools.partial(individual_vectorstore_search_node, store_index=store_index)


def create_merge_results_node_with_config() -> Callable[[MultiRAGState, RunnableConfig], Dict[str, Any]]:
    """Create results merging node that uses runtime configuration."""
    return merge_results_node


def create_prompt_node_with_config() -> Callable[[MultiRAGState, RunnableConfig], Dict[str, Any]]:
    """Create prompt formatting node that uses runtime configuration."""
    return prompt_node


def create_chat_node_with_config() -> Callable[[MultiRAGState, RunnableConfig], Dict[str, Any]]:
    """Create chat node that uses runtime configuration."""
    return chat_node 