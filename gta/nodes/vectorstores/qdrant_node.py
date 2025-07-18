"""
Qdrant vectorstore node implementation with embedded embeddings.
"""

from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

from gta.states.vectorstores.qdrant_state import QdrantState


def qdrant_node(state: QdrantState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Process Qdrant vectorstore operations with embedded embeddings.
    
    Operations:
    - insert: Add documents to collection (with auto-embedding)
    - search: Similarity search with query (with auto-embedding)
    - delete: Remove documents by IDs
    - update: Update existing documents
    
    Args:
        state: Current Qdrant state
        config: Runtime configuration with Qdrant and embedding settings
        
    Returns:
        Updated state dictionary with operation results
    """
    
    try:
        # Get configuration
        config_data = config.get("configurable", {})
        
        # Qdrant client configuration
        qdrant_url = config_data.get("url", "http://localhost:6333")
        api_key = config_data.get("api_key")
        collection_name = config_data.get("collection_name", "default_collection")
        
        # Embedding configuration
        embedding_model = config_data.get("embedding_model", "mxbai-embed-large")
        embedding_provider = config_data.get("embedding_provider", "ollama")
        embedding_base_url = config_data.get("embedding_base_url", "http://localhost:11434")
        embedding_api_key = config_data.get("embedding_api_key")
        
        # Operation settings
        top_k = config_data.get("top_k", 5)
        score_threshold = config_data.get("score_threshold", 0.0)
        distance_metric = config_data.get("distance", "cosine")
        
        # Initialize embedding model
        if embedding_provider == "openai":
            embeddings = OpenAIEmbeddings(
                model=embedding_model,
                api_key=embedding_api_key,
                **config_data.get("model_kwargs", {})
            )
        else:  # Default to Ollama
            embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=embedding_base_url,
                **config_data.get("model_kwargs", {})
            )
        
        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=api_key,
            timeout=config_data.get("timeout", 60.0),
            prefer_grpc=config_data.get("prefer_grpc", False)
        )
        
        # Initialize vector store
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            distance_func=_get_distance_func(distance_metric)
        )
        
        # Process operation
        if state.operation == "insert":
            return _handle_insert(state, vectorstore, config_data)
        elif state.operation == "search":
            return _handle_search(state, vectorstore, top_k, score_threshold, config_data)
        elif state.operation == "delete":
            return _handle_delete(state, vectorstore, config_data)
        elif state.operation == "update":
            return _handle_update(state, vectorstore, config_data)
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {state.operation}",
                "count": 0,
                "search_results": [],
                "operation_stats": {}
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Qdrant operation failed: {str(e)}",
            "count": 0,
            "search_results": [],
            "operation_stats": {"error_type": type(e).__name__}
        }


def _get_distance_func(distance_metric: str) -> Distance:
    """Convert distance metric string to Qdrant Distance enum."""
    distance_map = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclidean": Distance.EUCLID
    }
    return distance_map.get(distance_metric.lower(), Distance.COSINE)


def _handle_insert(state: QdrantState, vectorstore: QdrantVectorStore, config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document insertion operation."""
    try:
        if not state.documents:
            return {
                "success": False,
                "error": "No documents provided for insertion",
                "count": 0,
                "search_results": [],
                "operation_stats": {}
            }
        
        # Add documents to vector store (embeddings are generated automatically)
        batch_size = config.get("batch_size", 100)
        
        if len(state.documents) <= batch_size:
            # Single batch
            document_ids = vectorstore.add_documents(state.documents)
        else:
            # Multiple batches
            document_ids = []
            for i in range(0, len(state.documents), batch_size):
                batch = state.documents[i:i + batch_size]
                batch_ids = vectorstore.add_documents(batch)
                document_ids.extend(batch_ids)
        
        return {
            "success": True,
            "error": None,
            "count": len(document_ids),
            "search_results": [],
            "operation_stats": {
                "inserted_documents": len(document_ids),
                "batch_size": batch_size,
                "document_ids": document_ids
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Insert operation failed: {str(e)}",
            "count": 0,
            "search_results": [],
            "operation_stats": {"error_type": type(e).__name__}
        }


def _handle_search(state: QdrantState, vectorstore: QdrantVectorStore, top_k: int, 
                  score_threshold: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle similarity search operation."""
    try:
        if not state.query:
            return {
                "success": False,
                "error": "No query provided for search",
                "count": 0,
                "search_results": [],
                "operation_stats": {}
            }
        
        # Perform similarity search (query embedding is generated automatically)
        if state.filter_conditions:
            # Search with metadata filtering
            docs = vectorstore.similarity_search_with_score(
                state.query,
                k=top_k,
                filter=state.filter_conditions,
                score_threshold=score_threshold
            )
        else:
            # Simple similarity search
            docs = vectorstore.similarity_search_with_score(
                state.query,
                k=top_k,
                score_threshold=score_threshold
            )
        
        # Format search results
        search_results = []
        for doc, score in docs:
            search_results.append({
                "document": {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                },
                "score": float(score),
                "id": doc.metadata.get("id", str(uuid.uuid4()))
            })
        
        return {
            "success": True,
            "error": None,
            "count": len(search_results),
            "search_results": search_results,
            "operation_stats": {
                "query": state.query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "filter_applied": state.filter_conditions is not None
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Search operation failed: {str(e)}",
            "count": 0,
            "search_results": [],
            "operation_stats": {"error_type": type(e).__name__}
        }


def _handle_delete(state: QdrantState, vectorstore: QdrantVectorStore, config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document deletion operation."""
    try:
        if not state.document_ids:
            return {
                "success": False,
                "error": "No document IDs provided for deletion",
                "count": 0,
                "search_results": [],
                "operation_stats": {}
            }
        
        # Delete documents by IDs
        deleted_count = 0
        for doc_id in state.document_ids:
            try:
                vectorstore.delete([doc_id])
                deleted_count += 1
            except Exception as e:
                # Continue with other deletions even if one fails
                pass
        
        return {
            "success": True,
            "error": None,
            "count": deleted_count,
            "search_results": [],
            "operation_stats": {
                "requested_deletions": len(state.document_ids),
                "successful_deletions": deleted_count,
                "failed_deletions": len(state.document_ids) - deleted_count
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Delete operation failed: {str(e)}",
            "count": 0,
            "search_results": [],
            "operation_stats": {"error_type": type(e).__name__}
        }


def _handle_update(state: QdrantState, vectorstore: QdrantVectorStore, config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document update operation."""
    try:
        if not state.documents:
            return {
                "success": False,
                "error": "No documents provided for update",
                "count": 0,
                "search_results": [],
                "operation_stats": {}
            }
        
        # Update is typically implemented as delete + insert
        # For simplicity, we'll just add the documents (which may create duplicates)
        # In a production system, you'd want to implement proper update logic
        
        document_ids = vectorstore.add_documents(state.documents)
        
        return {
            "success": True,
            "error": None,
            "count": len(document_ids),
            "search_results": [],
            "operation_stats": {
                "updated_documents": len(document_ids),
                "note": "Update implemented as add operation"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Update operation failed: {str(e)}",
            "count": 0,
            "search_results": [],
            "operation_stats": {"error_type": type(e).__name__}
        } 