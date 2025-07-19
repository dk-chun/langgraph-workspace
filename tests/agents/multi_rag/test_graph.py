"""
Test Multi-RAG Graph.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel

from gta.agents.multi_rag.graph import (
    create_multi_rag_graph, 
    create_multi_rag_graph_with_individual_nodes,
    create_search_condition_by_score_threshold,
    create_search_condition_by_query_length,
    create_conditional_multi_rag_example
)
from gta.agents.multi_rag.state import MultiRAGState


def test_create_multi_rag_graph_defaults():
    """Test multi-RAG graph creation with defaults."""
    with patch('gta.agents.multi_rag.graph.ChatOllama') as mock_llm, \
         patch('gta.agents.multi_rag.graph.QdrantVectorStore') as mock_vs, \
         patch('gta.agents.multi_rag.graph.QdrantClient') as mock_client, \
         patch('gta.agents.multi_rag.graph.OllamaEmbeddings') as mock_embeddings:
        
        # Mock the vectorstore constructor
        mock_vs_instance = Mock(spec=VectorStore)
        mock_vs.return_value = mock_vs_instance
        
        graph = create_multi_rag_graph()
        
        assert graph is not None
        # Should have created 3 vectorstores
        assert mock_vs.call_count == 3


def test_create_multi_rag_graph_custom_vectorstores(mock_vectorstores):
    """Test multi-RAG graph with custom vectorstores."""
    with patch('gta.agents.multi_rag.graph.ChatOllama') as mock_llm:
        graph = create_multi_rag_graph(vectorstores=mock_vectorstores)
        
        assert graph is not None


def test_create_multi_rag_graph_wrong_vectorstore_count():
    """Test error when not exactly 3 vectorstores provided."""
    vs1 = Mock(spec=VectorStore)
    vs2 = Mock(spec=VectorStore)
    
    with pytest.raises(ValueError, match="Exactly 3 vectorstores are required"):
        create_multi_rag_graph(vectorstores=[vs1, vs2])  # Only 2 vectorstores


def test_create_multi_rag_graph_custom_params(mock_vectorstores, mock_llm):
    """Test multi-RAG graph with custom parameters."""
    custom_template = [
        ("system", "Custom system message"),
        ("user", "Custom user template: {context} {question}")
    ]
    
    graph = create_multi_rag_graph(
        llm=mock_llm,
        vectorstores=mock_vectorstores,
        template_messages=custom_template,
        top_k_per_store=5,
        final_top_k=8,
        merge_strategy="weighted"
    )
    
    assert graph is not None


@pytest.mark.integration
def test_multi_rag_graph_execution(mock_vectorstores, mock_llm):
    """Integration test for multi-RAG graph execution."""
    # Mock document objects with different content
    mock_doc1 = Mock()
    mock_doc1.page_content = "AI is artificial intelligence."
    mock_doc2 = Mock()
    mock_doc2.page_content = "Machine learning is a subset of AI."
    mock_doc3 = Mock()
    mock_doc3.page_content = "Deep learning uses neural networks."
    
    # Override mock search results
    mock_vectorstores[0].similarity_search_with_score.return_value = [(mock_doc1, 0.9)]
    mock_vectorstores[1].similarity_search_with_score.return_value = [(mock_doc2, 0.8)]
    mock_vectorstores[2].similarity_search_with_score.return_value = [(mock_doc3, 0.7)]
    
    # Mock LLM response
    with patch('gta.core.chat.invoke_llm') as mock_invoke:
        mock_invoke.return_value = "AI is a broad field that includes machine learning."
        
        # Create and test graph
        graph = create_multi_rag_graph(
            llm=mock_llm,
            vectorstores=mock_vectorstores,
            top_k_per_store=1,
            final_top_k=3
        )
        
        # Execute graph
        initial_state = MultiRAGState(
            messages=[HumanMessage(content="What is AI?")]
        )
        
        result = graph.invoke(initial_state)
        
        # Verify results
        assert result["question"] == "What is AI?"
        assert len(result["vs1_documents"]) == 1
        assert len(result["vs2_documents"]) == 1
        assert len(result["vs3_documents"]) == 1
        assert len(result["merged_documents"]) <= 3
        assert result["final_context"] != ""
        assert len(result["messages"]) >= 2  # Original + AI response


def test_multi_rag_graph_nodes(mock_vectorstores):
    """Test that multi-RAG graph has correct nodes."""
    with patch('gta.agents.multi_rag.graph.ChatOllama') as mock_llm:
        graph = create_multi_rag_graph(vectorstores=mock_vectorstores)
        
        # Check that graph has expected nodes
        nodes = list(graph.nodes.keys())
        expected_nodes = ["extract_query", "parallel_search", "merge_results", "prompt", "generate"]
        
        for node in expected_nodes:
            assert node in nodes


# Individual nodes tests
def test_create_multi_rag_graph_with_individual_nodes_defaults():
    """Test individual nodes multi-RAG graph creation with defaults."""
    with patch('gta.agents.multi_rag.graph.ChatOllama') as mock_llm, \
         patch('gta.agents.multi_rag.graph.QdrantVectorStore') as mock_vs, \
         patch('gta.agents.multi_rag.graph.QdrantClient') as mock_client, \
         patch('gta.agents.multi_rag.graph.OllamaEmbeddings') as mock_embeddings:
        
        # Mock the vectorstore constructor
        mock_vs_instance = Mock(spec=VectorStore)
        mock_vs.return_value = mock_vs_instance
        
        graph = create_multi_rag_graph_with_individual_nodes()
        
        assert graph is not None
        # Should have created 3 vectorstores
        assert mock_vs.call_count == 3


def test_create_multi_rag_graph_with_individual_nodes_custom(mock_vectorstores, mock_llm):
    """Test individual nodes multi-RAG graph with custom parameters."""
    custom_template = [
        ("system", "Custom system message"),
        ("user", "Custom user template: {context} {question}")
    ]
    
    graph = create_multi_rag_graph_with_individual_nodes(
        llm=mock_llm,
        vectorstores=mock_vectorstores,
        template_messages=custom_template,
        top_k_per_store=5,
        final_top_k=8,
        merge_strategy="weighted"
    )
    
    assert graph is not None


def test_individual_nodes_graph_structure(mock_vectorstores):
    """Test that individual nodes graph has correct structure."""
    with patch('gta.agents.multi_rag.graph.ChatOllama') as mock_llm:
        graph = create_multi_rag_graph_with_individual_nodes(vectorstores=mock_vectorstores)
        
        # Check that graph has expected nodes
        nodes = list(graph.nodes.keys())
        expected_nodes = ["extract_query", "vs1_search", "vs2_search", "vs3_search", "merge_results", "prompt", "generate"]
        
        for node in expected_nodes:
            assert node in nodes


def test_conditional_search_nodes(mock_vectorstores, mock_llm):
    """Test conditional search nodes functionality."""
    # Create condition functions
    conditions = [
        lambda state: True,  # vs1 always searches
        create_search_condition_by_query_length(10),  # vs2 for longer queries
        create_search_condition_by_score_threshold(0.5)  # vs3 for low scores
    ]
    
    graph = create_multi_rag_graph_with_individual_nodes(
        llm=mock_llm,
        vectorstores=mock_vectorstores,
        use_conditional_search=True,
        search_conditions=conditions
    )
    
    assert graph is not None


@pytest.mark.integration
def test_individual_nodes_graph_execution(mock_vectorstores, mock_llm):
    """Integration test for individual nodes multi-RAG graph execution."""
    # Mock document objects with different content
    mock_doc1 = Mock()
    mock_doc1.page_content = "AI is artificial intelligence."
    mock_doc2 = Mock()
    mock_doc2.page_content = "Machine learning is a subset of AI."
    mock_doc3 = Mock()
    mock_doc3.page_content = "Deep learning uses neural networks."
    
    # Override mock search results
    mock_vectorstores[0].similarity_search_with_score.return_value = [(mock_doc1, 0.9)]
    mock_vectorstores[1].similarity_search_with_score.return_value = [(mock_doc2, 0.8)]
    mock_vectorstores[2].similarity_search_with_score.return_value = [(mock_doc3, 0.7)]
    
    # Mock LLM response
    with patch('gta.core.chat.invoke_llm') as mock_invoke:
        mock_invoke.return_value = "AI is a broad field that includes machine learning."
        
        # Create and test graph
        graph = create_multi_rag_graph_with_individual_nodes(
            llm=mock_llm,
            vectorstores=mock_vectorstores,
            top_k_per_store=1,
            final_top_k=3
        )
        
        # Execute graph
        initial_state = MultiRAGState(
            messages=[HumanMessage(content="What is AI?")]
        )
        
        result = graph.invoke(initial_state)
        
        # Verify results
        assert result["question"] == "What is AI?"
        assert len(result["vs1_documents"]) == 1
        assert len(result["vs2_documents"]) == 1
        assert len(result["vs3_documents"]) == 1
        assert len(result["merged_documents"]) <= 3
        assert result["final_context"] != ""
        assert len(result["messages"]) >= 2  # Original + AI response


def test_search_condition_functions():
    """Test search condition utility functions."""
    # Test query length condition
    length_condition = create_search_condition_by_query_length(10)
    
    short_state = MultiRAGState(question="Short")
    long_state = MultiRAGState(question="This is a much longer question")
    
    assert not length_condition(short_state)
    assert length_condition(long_state)
    
    # Test score threshold condition
    score_condition = create_search_condition_by_score_threshold(0.7)
    
    no_scores_state = MultiRAGState()
    high_scores_state = MultiRAGState(search_scores={"vs1": 0.9, "vs2": 0.8})
    low_scores_state = MultiRAGState(search_scores={"vs1": 0.5, "vs2": 0.4})
    
    assert score_condition(no_scores_state)  # Should search when no scores
    assert not score_condition(high_scores_state)  # Should not search for high scores
    assert score_condition(low_scores_state)  # Should search for low scores


def test_conditional_multi_rag_example():
    """Test the conditional multi-rag example function."""
    with patch('gta.agents.multi_rag.graph.ChatOllama') as mock_llm, \
         patch('gta.agents.multi_rag.graph.QdrantVectorStore') as mock_vs, \
         patch('gta.agents.multi_rag.graph.QdrantClient') as mock_client, \
         patch('gta.agents.multi_rag.graph.OllamaEmbeddings') as mock_embeddings:
        
        mock_vs_instance = Mock(spec=VectorStore)
        mock_vs.return_value = mock_vs_instance
        
        graph = create_conditional_multi_rag_example()
        
        assert graph is not None
        
        # Check that graph has expected nodes for individual search
        nodes = list(graph.nodes.keys())
        expected_nodes = ["extract_query", "vs1_search", "vs2_search", "vs3_search", "merge_results", "prompt", "generate"]
        
        for node in expected_nodes:
            assert node in nodes 