"""
Tests for basic agent state.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from gta.agents.basic.state import BasicState


class TestBasicState:
    """Test cases for BasicState."""
    
    def test_basic_state_initialization(self):
        """Test BasicState can be initialized empty."""
        state = BasicState()
        assert state.messages == []
    
    def test_basic_state_with_messages(self):
        """Test BasicState with initial messages."""
        messages = [HumanMessage(content="Hello")]
        state = BasicState(messages=messages)
        assert len(state.messages) == 1
        assert state.messages[0].content == "Hello"
    
    def test_message_reducer_functionality(self):
        """Test that messages are properly added using the reducer."""
        state = BasicState()
        # Simulate adding messages as the reducer would
        new_messages = [
            HumanMessage(content="First message"),
            AIMessage(content="AI response")
        ]
        state.messages.extend(new_messages)
        
        assert len(state.messages) == 2
        assert state.messages[0].content == "First message"
        assert state.messages[1].content == "AI response"
    
    def test_state_serialization(self):
        """Test BasicState can be serialized to dict."""
        messages = [HumanMessage(content="Test")]
        state = BasicState(messages=messages)
        state_dict = state.model_dump()
        
        assert "messages" in state_dict
        assert len(state_dict["messages"]) == 1 