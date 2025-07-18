"""
Tests for basic agent nodes.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from gta.agents.basic.nodes import create_chat_node, _chat_adapter
from gta.agents.basic.state import BasicState


class TestBasicNodes:
    """Test cases for basic agent nodes."""
    
    def test_chat_adapter_basic_functionality(self, mock_llm):
        """Test basic chat adapter functionality."""
        # Setup
        state = BasicState(messages=[HumanMessage(content="Hello")])
        
        # Execute
        result = _chat_adapter(state, mock_llm)
        
        # Verify
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Mock response"
        mock_llm.invoke.assert_called_once()
    
    def test_chat_adapter_with_system_prompt(self, mock_llm):
        """Test chat adapter with system prompt."""
        # Setup
        state = BasicState(messages=[HumanMessage(content="Hello")])
        system_prompt = "You are a helpful assistant."
        
        # Execute
        result = _chat_adapter(state, mock_llm, system_prompt)
        
        # Verify
        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        mock_llm.invoke.assert_called_once()
        # Check that system message was added to the call
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 2  # system + human message
    
    def test_create_chat_node_factory(self, mock_llm):
        """Test chat node factory function."""
        # Create node
        chat_node = create_chat_node(mock_llm)
        
        # Test the created node
        state = BasicState(messages=[HumanMessage(content="Test")])
        result = chat_node(state)
        
        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        mock_llm.invoke.assert_called_once()
    
    def test_create_chat_node_with_system_prompt(self, mock_llm):
        """Test chat node factory with system prompt."""
        system_prompt = "You are a test assistant."
        chat_node = create_chat_node(mock_llm, system_prompt)
        
        state = BasicState(messages=[HumanMessage(content="Test")])
        result = chat_node(state)
        
        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        mock_llm.invoke.assert_called_once()
    
    def test_chat_adapter_empty_messages(self, mock_llm):
        """Test chat adapter with empty message list."""
        state = BasicState()
        
        result = _chat_adapter(state, mock_llm)
        
        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        mock_llm.invoke.assert_called_once_with([]) 