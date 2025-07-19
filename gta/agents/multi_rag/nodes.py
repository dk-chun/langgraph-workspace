"""
Multi-VectorStore RAG Agent Nodes.
"""

import functools
from typing import Dict, Any, List, Tuple, Callable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from gta.core.chat import invoke_llm
from gta.core.vectorstore import search_documents, format_search_results
from gta.core.prompt import format_prompt_template
from gta.agents.multi_rag.state import MultiRAGState


def _extract_query_adapter(state: MultiRAGState) -> Dict[str, Any]:
    """Extract query from messages for search."""
    question = ""
    if state.messages:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break
    
    return {"question": question}


def _parallel_search_adapter(
    state: MultiRAGState, 
    vectorstores: List[VectorStore], 
    top_k: int = 3
) -> Dict[str, Any]:
    """Search all vectorstores in parallel."""
    query = state.question
    if not query:
        return {
            "vs1_documents": [],
            "vs2_documents": [],
            "vs3_documents": [],
            "search_scores": {}
        }
    
    # Search each vectorstore
    vs1_results = search_documents(query, vectorstores[0], top_k) if len(vectorstores) > 0 else []
    vs2_results = search_documents(query, vectorstores[1], top_k) if len(vectorstores) > 1 else []
    vs3_results = search_documents(query, vectorstores[2], top_k) if len(vectorstores) > 2 else []
    
    # Calculate search quality scores (average similarity scores)
    search_scores = {}
    if vs1_results:
        search_scores["vs1"] = sum(score for _, score in vs1_results) / len(vs1_results)
    if vs2_results:
        search_scores["vs2"] = sum(score for _, score in vs2_results) / len(vs2_results)
    if vs3_results:
        search_scores["vs3"] = sum(score for _, score in vs3_results) / len(vs3_results)
    
    return {
        "vs1_documents": vs1_results,
        "vs2_documents": vs2_results,
        "vs3_documents": vs3_results,
        "search_scores": search_scores
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


def _merge_results_adapter(
    state: MultiRAGState, 
    strategy: str = "simple", 
    final_top_k: int = 5
) -> Dict[str, Any]:
    """Merge results from all vectorstores."""
    vs1_results = state.vs1_documents
    vs2_results = state.vs2_documents
    vs3_results = state.vs3_documents
    
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
    final_context = format_search_results(merged_documents)
    
    return {
        "merged_documents": merged_documents,
        "final_context": final_context,
        "merge_strategy": strategy
    }


def _prompt_adapter(state: MultiRAGState, template_messages: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Format prompt with merged context and question."""
    variables = {
        "context": state.final_context,
        "question": state.question
    }
    
    messages = format_prompt_template(template_messages, variables)
    
    return {"messages": messages}


def _chat_adapter(state: MultiRAGState, llm: BaseChatModel) -> Dict[str, Any]:
    """Generate response using LLM."""
    messages = state.messages
    response_content = invoke_llm(messages, llm)
    
    return {"messages": [AIMessage(content=response_content)]}


# Node factory functions
def create_extract_query_node() -> Callable[[MultiRAGState], Dict[str, Any]]:
    """Create query extraction node."""
    return _extract_query_adapter


def create_vectorstore_search_node(
    vectorstore: VectorStore, 
    store_id: str, 
    top_k: int = 3
) -> Callable[[MultiRAGState], Dict[str, Any]]:
    """
    Create individual vectorstore search node.
    
    Args:
        vectorstore: VectorStore instance to search
        store_id: Identifier for the vectorstore (vs1, vs2, vs3)
        top_k: Number of documents to retrieve
        
    Returns:
        Node function for individual vectorstore search
    """
    def _individual_search_adapter(state: MultiRAGState) -> Dict[str, Any]:
        query = state.question
        if not query:
            return {f"{store_id}_documents": []}
        
        results = search_documents(query, vectorstore, top_k)
        
        # Calculate search quality score
        search_score = 0.0
        if results:
            search_score = sum(score for _, score in results) / len(results)
        
        return {
            f"{store_id}_documents": results,
            "search_scores": {store_id: search_score}
        }
    
    return _individual_search_adapter


def create_conditional_search_node(
    vectorstore: VectorStore,
    store_id: str,
    condition_fn: Callable[[MultiRAGState], bool],
    top_k: int = 3
) -> Callable[[MultiRAGState], Dict[str, Any]]:
    """
    Create conditional vectorstore search node that only searches if condition is met.
    
    Args:
        vectorstore: VectorStore instance to search
        store_id: Identifier for the vectorstore
        condition_fn: Function that returns True if search should be performed
        top_k: Number of documents to retrieve
        
    Returns:
        Node function for conditional vectorstore search
    """
    def _conditional_search_adapter(state: MultiRAGState) -> Dict[str, Any]:
        if not condition_fn(state):
            return {
                f"{store_id}_documents": [],
                "search_scores": {store_id: 0.0}
            }
        
        query = state.question
        if not query:
            return {
                f"{store_id}_documents": [],
                "search_scores": {store_id: 0.0}
            }
        
        results = search_documents(query, vectorstore, top_k)
        
        # Calculate search quality score
        search_score = 0.0
        if results:
            search_score = sum(score for _, score in results) / len(results)
        
        return {
            f"{store_id}_documents": results,
            "search_scores": {store_id: search_score}
        }
    
    return _conditional_search_adapter


def create_parallel_search_node(vectorstores: List[VectorStore], top_k: int = 3) -> Callable[[MultiRAGState], Dict[str, Any]]:
    """Create parallel search node for multiple vectorstores."""
    return functools.partial(_parallel_search_adapter, vectorstores=vectorstores, top_k=top_k)


def create_merge_results_node(strategy: str = "simple", final_top_k: int = 5) -> Callable[[MultiRAGState], Dict[str, Any]]:
    """Create results merging node."""
    return functools.partial(_merge_results_adapter, strategy=strategy, final_top_k=final_top_k)


def create_prompt_node(template_messages: List[Tuple[str, str]]) -> Callable[[MultiRAGState], Dict[str, Any]]:
    """Create prompt formatting node."""
    return functools.partial(_prompt_adapter, template_messages=template_messages)


def create_chat_node(llm: BaseChatModel) -> Callable[[MultiRAGState], Dict[str, Any]]:
    """Create chat node with LLM."""
    return functools.partial(_chat_adapter, llm=llm) 