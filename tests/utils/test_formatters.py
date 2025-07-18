"""
Tests for utils formatters functionality.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from gta.utils.formatters import format_messages, format_context, format_agent_response


class TestFormatters:
    """Test cases for formatter functions."""
    
    def test_format_messages_basic(self):
        """Test basic message formatting."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        
        result = format_messages(messages)
        
        expected = "1. Human: Hello\n2. AI: Hi there!"
        assert result == expected
    
    def test_format_messages_with_system_message(self):
        """Test formatting with system message."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!")
        ]
        
        result = format_messages(messages)
        
        expected = "1. System: You are a helpful assistant\n2. Human: Hello\n3. AI: Hi!"
        assert result == expected
    
    def test_format_messages_long_content_truncation(self):
        """Test message formatting with long content truncation."""
        long_content = "This is a very long message " * 10  # Over 100 characters
        messages = [HumanMessage(content=long_content)]
        
        result = format_messages(messages)
        
        assert "..." in result
        assert len(result.split(": ", 1)[1]) <= 103  # 100 chars + "..."
    
    def test_format_messages_empty_list(self):
        """Test formatting empty message list."""
        messages = []
        
        result = format_messages(messages)
        
        assert result == ""
    
    def test_format_messages_single_message(self):
        """Test formatting single message."""
        messages = [HumanMessage(content="Single message")]
        
        result = format_messages(messages)
        
        expected = "1. Human: Single message"
        assert result == expected
    
    def test_format_messages_numbering(self):
        """Test that messages are numbered correctly."""
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third"),
            AIMessage(content="Fourth")
        ]
        
        result = format_messages(messages)
        
        lines = result.split("\n")
        assert len(lines) == 4
        assert lines[0].startswith("1. ")
        assert lines[1].startswith("2. ")
        assert lines[2].startswith("3. ")
        assert lines[3].startswith("4. ")
    
    def test_format_context_basic(self):
        """Test basic context formatting."""
        docs = [
            {"content": "First document content"},
            {"content": "Second document content"}
        ]
        
        result = format_context(docs)
        
        expected = "Document 1:\nFirst document content\n\nDocument 2:\nSecond document content"
        assert result == expected
    
    def test_format_context_empty_docs(self):
        """Test formatting empty document list."""
        docs = []
        
        result = format_context(docs)
        
        assert result == "No relevant documents found."
    
    def test_format_context_single_document(self):
        """Test formatting single document."""
        docs = [{"content": "Single document content"}]
        
        result = format_context(docs)
        
        expected = "Document 1:\nSingle document content"
        assert result == expected
    
    def test_format_context_with_max_length(self):
        """Test context formatting with max length limit."""
        docs = [
            {"content": "First document with some content"},
            {"content": "Second document with more content"},
            {"content": "Third document that might be truncated"}
        ]
        max_length = 100
        
        result = format_context(docs, max_length=max_length)
        
        assert len(result) <= max_length
        assert "Document 1:" in result
    
    def test_format_context_truncation_with_ellipsis(self):
        """Test that truncation adds ellipsis."""
        long_content = "This is a very long document content " * 20
        docs = [{"content": long_content}]
        max_length = 100
        
        result = format_context(docs, max_length=max_length)
        
        assert len(result) <= max_length
        assert result.endswith("...")
    
    def test_format_context_no_truncation_needed(self):
        """Test context formatting when no truncation is needed."""
        docs = [{"content": "Short content"}]
        max_length = 1000
        
        result = format_context(docs, max_length=max_length)
        
        expected = "Document 1:\nShort content"
        assert result == expected
        assert "..." not in result
    
    def test_format_context_multiple_documents_numbering(self):
        """Test that documents are numbered correctly."""
        docs = [
            {"content": f"Document {i+1} content"} for i in range(5)
        ]
        
        result = format_context(docs)
        
        for i in range(5):
            assert f"Document {i+1}:" in result
            assert f"Document {i+1} content" in result
    
    def test_format_agent_response_basic(self):
        """Test basic agent response formatting."""
        agent_type = "basic"
        response = "Hello, how can I help you?"
        
        result = format_agent_response(agent_type, response)
        
        expected = "[BASIC] Hello, how can I help you?"
        assert result == expected
    
    def test_format_agent_response_with_metadata(self):
        """Test agent response formatting with metadata."""
        agent_type = "rag"
        response = "Based on the documents, AI is artificial intelligence."
        metadata = {"docs_found": 3, "confidence": 0.95}
        
        result = format_agent_response(agent_type, response, metadata)
        
        expected = "[RAG] Based on the documents, AI is artificial intelligence.\n(Metadata: docs_found: 3, confidence: 0.95)"
        assert result == expected
    
    def test_format_agent_response_empty_metadata(self):
        """Test agent response formatting with empty metadata."""
        agent_type = "basic"
        response = "Test response"
        metadata = {}
        
        result = format_agent_response(agent_type, response, metadata)
        
        expected = "[BASIC] Test response"
        assert result == expected
    
    def test_format_agent_response_none_metadata(self):
        """Test agent response formatting with None metadata."""
        agent_type = "basic"
        response = "Test response"
        metadata = None
        
        result = format_agent_response(agent_type, response, metadata)
        
        expected = "[BASIC] Test response"
        assert result == expected
    
    def test_format_agent_response_complex_metadata(self):
        """Test agent response formatting with complex metadata."""
        agent_type = "advanced"
        response = "Complex response"
        metadata = {
            "processing_time": 1.5,
            "model": "gpt-4",
            "tokens_used": 150,
            "temperature": 0.7
        }
        
        result = format_agent_response(agent_type, response, metadata)
        
        assert "[ADVANCED] Complex response" in result
        assert "processing_time: 1.5" in result
        assert "model: gpt-4" in result
        assert "tokens_used: 150" in result
        assert "temperature: 0.7" in result
    
    def test_format_agent_response_case_conversion(self):
        """Test that agent type is converted to uppercase."""
        agent_type = "lowercase_agent"
        response = "Test response"
        
        result = format_agent_response(agent_type, response)
        
        assert "[LOWERCASE_AGENT]" in result
    
    def test_format_context_edge_case_small_remaining_space(self):
        """Test context formatting when remaining space is too small."""
        docs = [
            {"content": "First short doc"},
            {"content": "This is a much longer document that will exceed the limit"}
        ]
        max_length = 50  # Very small limit
        
        result = format_context(docs, max_length=max_length)
        
        # Should only include first document since second would exceed limit
        assert "Document 1:" in result
        assert "Document 2:" not in result
        assert len(result) <= max_length 