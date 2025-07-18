"""
Tests for simple_rag agent state.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from gta.agents.simple_rag.state import RAGState


class TestRAGState:
    """Test cases for RAGState."""
    
    def test_rag_state_initialization(self):
        """Test RAGState can be initialized empty."""
        state = RAGState()
        assert state.question == ""
        assert state.context == ""
        assert state.documents == []
        assert state.messages == []
    
    def test_rag_state_with_parameters(self):
        """Test RAGState with initial parameters."""
        messages = [HumanMessage(content="What is AI?")]
        documents = [{"content": "AI is artificial intelligence"}]
        
        state = RAGState(
            question="What is AI?",
            context="AI is artificial intelligence",
            documents=documents,
            messages=messages
        )
        
        assert state.question == "What is AI?"
        assert state.context == "AI is artificial intelligence"
        assert len(state.documents) == 1
        assert len(state.messages) == 1
    
    def test_rag_state_field_validation(self):
        """Test that RAGState validates field types correctly."""
        # Test with correct types
        state = RAGState(
            question="test question",
            context="test context", 
            documents=[{"content": "test"}],
            messages=[HumanMessage(content="test")]
        )
        
        assert isinstance(state.question, str)
        assert isinstance(state.context, str)
        assert isinstance(state.documents, list)
        assert isinstance(state.messages, list)
    
    def test_message_reducer_functionality(self):
        """Test that messages are properly added using the reducer."""
        state = RAGState()
        new_messages = [
            HumanMessage(content="Question"),
            AIMessage(content="Answer")
        ]
        state.messages.extend(new_messages)
        
        assert len(state.messages) == 2
        assert state.messages[0].content == "Question"
        assert state.messages[1].content == "Answer"
    
    def test_state_serialization(self):
        """Test RAGState can be serialized to dict."""
        state = RAGState(
            question="Test question",
            context="Test context",
            documents=[{"content": "test"}],
            messages=[HumanMessage(content="Test")]
        )
        
        state_dict = state.model_dump()
        
        assert state_dict["question"] == "Test question"
        assert state_dict["context"] == "Test context"
        assert len(state_dict["documents"]) == 1
        assert len(state_dict["messages"]) == 1
    
    def test_documents_list_handling(self):
        """Test documents list can handle various document types."""
        documents = [
            {"content": "doc1", "metadata": {"source": "file1"}},
            {"content": "doc2", "metadata": {"source": "file2"}}
        ]
        
        state = RAGState(documents=documents)
        
        assert len(state.documents) == 2
        assert state.documents[0]["content"] == "doc1"
        assert state.documents[1]["content"] == "doc2" 