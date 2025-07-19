"""
Test Multi-RAG State.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from gta.agents.multi_rag.state import MultiRAGState


def test_multi_rag_state_initialization():
    """Test MultiRAGState initialization with defaults."""
    state = MultiRAGState()
    
    assert state.question == ""
    assert state.vs1_documents == []
    assert state.vs2_documents == []
    assert state.vs3_documents == []
    assert state.merged_documents == []
    assert state.final_context == ""
    assert state.search_scores == {}
    assert state.merge_strategy == "simple"
    assert state.messages == []


def test_multi_rag_state_with_data():
    """Test MultiRAGState with sample data."""
    sample_docs = [("doc1", 0.8), ("doc2", 0.7)]
    sample_scores = {"vs1": 0.75, "vs2": 0.65, "vs3": 0.80}
    
    state = MultiRAGState(
        question="What is AI?",
        vs1_documents=sample_docs,
        search_scores=sample_scores,
        merge_strategy="weighted"
    )
    
    assert state.question == "What is AI?"
    assert state.vs1_documents == sample_docs
    assert state.search_scores == sample_scores
    assert state.merge_strategy == "weighted"


def test_multi_rag_state_messages():
    """Test MultiRAGState message handling."""
    state = MultiRAGState()
    
    # Add messages
    human_msg = HumanMessage(content="Hello")
    ai_msg = AIMessage(content="Hi there!")
    
    state.messages = [human_msg, ai_msg]
    
    assert len(state.messages) == 2
    assert isinstance(state.messages[0], HumanMessage)
    assert isinstance(state.messages[1], AIMessage)
    assert state.messages[0].content == "Hello"
    assert state.messages[1].content == "Hi there!" 