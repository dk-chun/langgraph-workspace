"""
Tests for simple_rag agent nodes.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from gta.agents.simple_rag.nodes import (
    _extract_query_adapter,
    _search_adapter,
    create_extract_query_node,
    create_search_node
)
from gta.agents.simple_rag.state import RAGState


class TestRAGNodes:
    """Test cases for RAG agent nodes."""
    
    def test_extract_query_adapter_with_messages(self):
        """Test query extraction from messages."""
        messages = [
            HumanMessage(content="What is machine learning?"),
            AIMessage(content="Previous response"),
            HumanMessage(content="Tell me about AI")
        ]
        state = RAGState(messages=messages)
        
        result = _extract_query_adapter(state)
        
        assert "question" in result
        assert result["question"] == "Tell me about AI"  # Should get last human message
    
    def test_extract_query_adapter_empty_messages(self):
        """Test query extraction with empty messages."""
        state = RAGState()
        
        result = _extract_query_adapter(state)
        
        assert "question" in result
        assert result["question"] == ""
    
    def test_extract_query_adapter_no_human_messages(self):
        """Test query extraction with no human messages."""
        messages = [AIMessage(content="AI response")]
        state = RAGState(messages=messages)
        
        result = _extract_query_adapter(state)
        
        assert "question" in result
        assert result["question"] == ""
    
    def test_search_adapter_with_query(self, mock_vectorstore):
        """Test search adapter with valid query."""
        state = RAGState(question="What is AI?")
        
        with patch('gta.agents.simple_rag.nodes.search_documents') as mock_search, \
             patch('gta.agents.simple_rag.nodes.format_search_results') as mock_format:
            
            mock_docs = [Mock(page_content="AI is artificial intelligence")]
            mock_search.return_value = mock_docs
            mock_format.return_value = "Formatted context"
            
            result = _search_adapter(state, mock_vectorstore)
            
            assert "documents" in result
            assert "context" in result
            assert result["context"] == "Formatted context"
            mock_search.assert_called_once_with("What is AI?", mock_vectorstore, 5)
            mock_format.assert_called_once_with(mock_docs)
    
    def test_search_adapter_empty_query(self, mock_vectorstore):
        """Test search adapter with empty query."""
        state = RAGState(question="")
        
        result = _search_adapter(state, mock_vectorstore)
        
        assert "documents" in result
        assert "context" in result
        assert result["documents"] == []
        assert result["context"] == "No question provided."
    
    def test_search_adapter_custom_top_k(self, mock_vectorstore):
        """Test search adapter with custom top_k parameter."""
        state = RAGState(question="Test query")
        
        with patch('gta.agents.simple_rag.nodes.search_documents') as mock_search, \
             patch('gta.agents.simple_rag.nodes.format_search_results') as mock_format:
            
            mock_search.return_value = []
            mock_format.return_value = "Context"
            
            result = _search_adapter(state, mock_vectorstore, top_k=3)
            
            mock_search.assert_called_once_with("Test query", mock_vectorstore, 3)
    
    def test_create_extract_query_node_factory(self):
        """Test extract query node factory function."""
        node = create_extract_query_node()
        
        # Test the created node
        messages = [HumanMessage(content="Test question")]
        state = RAGState(messages=messages)
        result = node(state)
        
        assert "question" in result
        assert result["question"] == "Test question"
    
    def test_create_search_node_factory(self, mock_vectorstore):
        """Test search node factory function."""
        node = create_search_node(mock_vectorstore)
        
        with patch('gta.agents.simple_rag.nodes.search_documents') as mock_search, \
             patch('gta.agents.simple_rag.nodes.format_search_results') as mock_format:
            
            mock_search.return_value = []
            mock_format.return_value = "Context"
            
            state = RAGState(question="Test")
            result = node(state)
            
            assert "documents" in result
            assert "context" in result
    
    def test_create_search_node_with_custom_top_k(self, mock_vectorstore):
        """Test search node factory with custom top_k."""
        node = create_search_node(mock_vectorstore, top_k=10)
        
        with patch('gta.agents.simple_rag.nodes.search_documents') as mock_search, \
             patch('gta.agents.simple_rag.nodes.format_search_results') as mock_format:
            
            mock_search.return_value = []
            mock_format.return_value = "Context"
            
            state = RAGState(question="Test")
            result = node(state)
            
            mock_search.assert_called_once_with("Test", mock_vectorstore, 10) 