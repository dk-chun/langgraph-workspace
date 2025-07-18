"""
Tests for core vectorstore functionality.
"""

import pytest
from unittest.mock import Mock
from langchain_core.vectorstores import VectorStore

from gta.core.vectorstore import search_documents, format_search_results


class TestVectorstoreFunctions:
    """Test cases for core vectorstore functions."""
    
    def test_search_documents_basic(self, mock_vectorstore):
        """Test basic document search."""
        query = "What is AI?"
        mock_results = [
            (Mock(page_content="AI is artificial intelligence"), 0.95),
            (Mock(page_content="Machine learning is a subset of AI"), 0.85)
        ]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        
        result = search_documents(query, mock_vectorstore)
        
        assert result == mock_results
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(query, k=5)
    
    def test_search_documents_custom_top_k(self, mock_vectorstore):
        """Test document search with custom top_k."""
        query = "Python programming"
        top_k = 3
        mock_results = [
            (Mock(page_content="Python is a programming language"), 0.9)
        ]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        
        result = search_documents(query, mock_vectorstore, top_k=top_k)
        
        assert result == mock_results
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(query, k=top_k)
    
    def test_search_documents_empty_query(self, mock_vectorstore):
        """Test document search with empty query."""
        query = ""
        mock_results = []
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        
        result = search_documents(query, mock_vectorstore)
        
        assert result == mock_results
        mock_vectorstore.similarity_search_with_score.assert_called_once_with("", k=5)
    
    def test_search_documents_no_results(self, mock_vectorstore):
        """Test document search with no results."""
        query = "nonexistent topic"
        mock_results = []
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        
        result = search_documents(query, mock_vectorstore)
        
        assert result == []
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(query, k=5)
    
    def test_format_search_results_basic(self):
        """Test basic search results formatting."""
        mock_doc1 = Mock()
        mock_doc1.page_content = "First document content"
        mock_doc2 = Mock()
        mock_doc2.page_content = "Second document content"
        
        results = [
            (mock_doc1, 0.95),
            (mock_doc2, 0.85)
        ]
        
        formatted = format_search_results(results)
        
        expected = (
            "[Document 1] (Score: 0.950)\n"
            "First document content\n\n"
            "[Document 2] (Score: 0.850)\n"
            "Second document content"
        )
        assert formatted == expected
    
    def test_format_search_results_single_document(self):
        """Test formatting single search result."""
        mock_doc = Mock()
        mock_doc.page_content = "Single document content"
        
        results = [(mock_doc, 0.9)]
        
        formatted = format_search_results(results)
        
        expected = "[Document 1] (Score: 0.900)\nSingle document content"
        assert formatted == expected
    
    def test_format_search_results_empty_results(self):
        """Test formatting empty search results."""
        results = []
        
        formatted = format_search_results(results)
        
        assert formatted == "No relevant documents found."
    
    def test_format_search_results_with_low_scores(self):
        """Test formatting results with low similarity scores."""
        mock_doc1 = Mock()
        mock_doc1.page_content = "Low score document"
        mock_doc2 = Mock()
        mock_doc2.page_content = "Very low score document"
        
        results = [
            (mock_doc1, 0.123),
            (mock_doc2, 0.001)
        ]
        
        formatted = format_search_results(results)
        
        expected = (
            "[Document 1] (Score: 0.123)\n"
            "Low score document\n\n"
            "[Document 2] (Score: 0.001)\n"
            "Very low score document"
        )
        assert formatted == expected
    
    def test_format_search_results_with_multiline_content(self):
        """Test formatting results with multiline document content."""
        mock_doc = Mock()
        mock_doc.page_content = "First line\nSecond line\nThird line"
        
        results = [(mock_doc, 0.8)]
        
        formatted = format_search_results(results)
        
        expected = "[Document 1] (Score: 0.800)\nFirst line\nSecond line\nThird line"
        assert formatted == expected
    
    def test_format_search_results_with_empty_content(self):
        """Test formatting results with empty document content."""
        mock_doc = Mock()
        mock_doc.page_content = ""
        
        results = [(mock_doc, 0.7)]
        
        formatted = format_search_results(results)
        
        expected = "[Document 1] (Score: 0.700)\n"
        assert formatted == expected
    
    def test_format_search_results_score_precision(self):
        """Test that scores are formatted with correct precision."""
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        
        results = [(mock_doc, 0.123456789)]
        
        formatted = format_search_results(results)
        
        # Should be rounded to 3 decimal places
        assert "[Document 1] (Score: 0.123)" in formatted
    
    def test_format_search_results_multiple_documents_numbering(self):
        """Test that documents are numbered correctly."""
        docs = []
        results = []
        
        for i in range(5):
            mock_doc = Mock()
            mock_doc.page_content = f"Document {i+1} content"
            docs.append(mock_doc)
            results.append((mock_doc, 0.9 - i * 0.1))
        
        formatted = format_search_results(results)
        
        for i in range(5):
            assert f"[Document {i+1}]" in formatted
            assert f"Document {i+1} content" in formatted 