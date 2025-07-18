"""
Tests for core chat functionality.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from gta.core.chat import invoke_llm, add_system_message


class TestChatFunctions:
    """Test cases for core chat functions."""
    
    def test_invoke_llm_basic(self, mock_llm):
        """Test basic LLM invocation."""
        messages = [HumanMessage(content="Hello")]
        
        result = invoke_llm(messages, mock_llm)
        
        assert result == "Mock response"
        mock_llm.invoke.assert_called_once_with(messages)
    
    def test_invoke_llm_with_multiple_messages(self, mock_llm):
        """Test LLM invocation with multiple messages."""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="First response"),
            HumanMessage(content="Second message")
        ]
        
        result = invoke_llm(messages, mock_llm)
        
        assert result == "Mock response"
        mock_llm.invoke.assert_called_once_with(messages)
    
    def test_invoke_llm_empty_messages(self, mock_llm):
        """Test LLM invocation with empty message list."""
        messages = []
        
        result = invoke_llm(messages, mock_llm)
        
        assert result == "Mock response"
        mock_llm.invoke.assert_called_once_with(messages)
    
    def test_invoke_llm_with_system_message(self, mock_llm):
        """Test LLM invocation with system message included."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello")
        ]
        
        result = invoke_llm(messages, mock_llm)
        
        assert result == "Mock response"
        mock_llm.invoke.assert_called_once_with(messages)
    
    def test_add_system_message_to_empty_list(self):
        """Test adding system message to empty message list."""
        messages = []
        system_prompt = "You are a helpful assistant."
        
        result = add_system_message(messages, system_prompt)
        
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == system_prompt
    
    def test_add_system_message_to_existing_messages(self):
        """Test adding system message to existing messages."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there")
        ]
        system_prompt = "You are a helpful assistant."
        
        result = add_system_message(messages, system_prompt)
        
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == system_prompt
        assert isinstance(result[1], HumanMessage)
        assert result[1].content == "Hello"
        assert isinstance(result[2], AIMessage)
        assert result[2].content == "Hi there"
    
    def test_add_system_message_preserves_original_list(self):
        """Test that adding system message doesn't modify original list."""
        original_messages = [HumanMessage(content="Hello")]
        system_prompt = "You are a helpful assistant."
        
        result = add_system_message(original_messages, system_prompt)
        
        # Original list should be unchanged
        assert len(original_messages) == 1
        assert isinstance(original_messages[0], HumanMessage)
        
        # Result should have system message added
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
    
    def test_add_system_message_with_existing_system_message(self):
        """Test adding system message when one already exists."""
        messages = [
            SystemMessage(content="Existing system message"),
            HumanMessage(content="Hello")
        ]
        system_prompt = "New system message"
        
        result = add_system_message(messages, system_prompt)
        
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == system_prompt
        assert isinstance(result[1], SystemMessage)
        assert result[1].content == "Existing system message"
        assert isinstance(result[2], HumanMessage)
    
    def test_add_system_message_empty_prompt(self):
        """Test adding empty system message."""
        messages = [HumanMessage(content="Hello")]
        system_prompt = ""
        
        result = add_system_message(messages, system_prompt)
        
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "" 