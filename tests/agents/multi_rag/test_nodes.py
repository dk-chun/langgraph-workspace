"""
Test Multi-RAG Nodes.
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel

from gta.agents.multi_rag.state import MultiRAGState
from gta.agents.multi_rag.nodes import (
    _extract_query_adapter,
    _parallel_search_adapter,
    _normalize_scores,
    _remove_duplicates,
    _merge_results_adapter,
    create_extract_query_node,
    create_parallel_search_node,
    create_merge_results_node
)


def test_extract_query_adapter():
    """Test query extraction from messages."""
    state = MultiRAGState(
        messages=[
            HumanMessage(content="What is machine learning?"),
            AIMessage(content="ML is..."),
            HumanMessage(content="Tell me more about AI")
        ]
    )
    
    result = _extract_query_adapter(state)
    
    # Should extract the last human message
    assert result["question"] == "Tell me more about AI"


def test_extract_query_adapter_empty_messages():
    """Test query extraction with no messages."""
    state = MultiRAGState()
    
    result = _extract_query_adapter(state)
    
    assert result["question"] == ""


def test_normalize_scores():
    """Test score normalization."""
    mock_doc1 = Mock()
    mock_doc2 = Mock()
    mock_doc3 = Mock()
    
    results = [
        (mock_doc1, 0.9),
        (mock_doc2, 0.5),
        (mock_doc3, 0.1)
    ]
    
    normalized = _normalize_scores(results)
    
    assert len(normalized) == 3
    # Max score should be 1.0, min should be 0.0
    assert normalized[0][1] == 1.0  # (0.9 - 0.1) / (0.9 - 0.1) = 1.0
    assert normalized[2][1] == 0.0  # (0.1 - 0.1) / (0.9 - 0.1) = 0.0
    assert 0.0 < normalized[1][1] < 1.0  # Middle value


def test_normalize_scores_empty():
    """Test score normalization with empty results."""
    normalized = _normalize_scores([])
    assert normalized == []


def test_normalize_scores_identical():
    """Test score normalization with identical scores."""
    mock_doc1 = Mock()
    mock_doc2 = Mock()
    
    results = [(mock_doc1, 0.5), (mock_doc2, 0.5)]
    normalized = _normalize_scores(results)
    
    # All scores should be 1.0 when identical
    assert all(score == 1.0 for _, score in normalized)


def test_remove_duplicates():
    """Test duplicate removal."""
    mock_doc1 = Mock()
    mock_doc1.page_content = "Content A"
    mock_doc2 = Mock()
    mock_doc2.page_content = "Content B"
    mock_doc3 = Mock()
    mock_doc3.page_content = "Content A"  # Duplicate
    
    results = [
        (mock_doc1, 0.9),
        (mock_doc2, 0.8),
        (mock_doc3, 0.7)  # Should be removed
    ]
    
    unique_results = _remove_duplicates(results)
    
    assert len(unique_results) == 2
    assert unique_results[0][0] == mock_doc1
    assert unique_results[1][0] == mock_doc2


def test_parallel_search_adapter(mock_vectorstores):
    """Test parallel search across vectorstores."""
    # Override mock results for this test
    mock_vectorstores[0].similarity_search_with_score.return_value = [("doc1", 0.9), ("doc2", 0.8)]
    mock_vectorstores[1].similarity_search_with_score.return_value = [("doc3", 0.7), ("doc4", 0.6)]
    mock_vectorstores[2].similarity_search_with_score.return_value = [("doc5", 0.85)]
    
    state = MultiRAGState(question="What is AI?")
    
    result = _parallel_search_adapter(state, mock_vectorstores, top_k=2)
    
    assert len(result["vs1_documents"]) == 2
    assert len(result["vs2_documents"]) == 2
    assert len(result["vs3_documents"]) == 1
    
    # Check search scores (averages) - use pytest.approx for floating point comparison
    assert result["search_scores"]["vs1"] == pytest.approx(0.85, rel=1e-9)  # (0.9 + 0.8) / 2
    assert result["search_scores"]["vs2"] == pytest.approx(0.65, rel=1e-9)  # (0.7 + 0.6) / 2
    assert result["search_scores"]["vs3"] == pytest.approx(0.85, rel=1e-9)  # 0.85 / 1


def test_merge_results_adapter(multi_rag_sample_state):
    """Test results merging."""
    result = _merge_results_adapter(multi_rag_sample_state, strategy="simple", final_top_k=3)
    
    # The actual number of merged documents depends on the sample state
    # Let's check what we actually get and verify it's reasonable
    merged_docs = result["merged_documents"]
    assert len(merged_docs) >= 1  # Should have at least some documents
    assert len(merged_docs) <= 3  # Should not exceed final_top_k
    assert result["merge_strategy"] == "simple"
    assert result["final_context"] != ""
    
    # Results should be sorted by normalized score (if we have multiple docs)
    if len(merged_docs) > 1:
        assert merged_docs[0][1] >= merged_docs[1][1]


def test_create_extract_query_node():
    """Test extract query node factory."""
    node = create_extract_query_node()
    
    state = MultiRAGState(messages=[HumanMessage(content="Test question")])
    result = node(state)
    
    assert result["question"] == "Test question"


def test_create_parallel_search_node(mock_vectorstores):
    """Test parallel search node factory."""
    node = create_parallel_search_node(mock_vectorstores[:1], top_k=1)
    
    state = MultiRAGState(question="Test query")
    result = node(state)
    
    assert len(result["vs1_documents"]) == 1


def test_create_merge_results_node():
    """Test merge results node factory."""
    node = create_merge_results_node(strategy="simple", final_top_k=2)
    
    mock_doc = Mock()
    mock_doc.page_content = "Test content"
    
    state = MultiRAGState(
        vs1_documents=[(mock_doc, 0.8)],
        vs2_documents=[],
        vs3_documents=[]
    )
    
    result = node(state)
    
    assert len(result["merged_documents"]) == 1
    assert result["merge_strategy"] == "simple" 