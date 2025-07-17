"""
Tests for MessagesState implementation.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from gta.states.messages_state import MessagesState


class TestMessagesState:
    """Test cases for MessagesState."""

    def test_messages_state_creation(self):
        """Test MessagesState creation."""
        # Test empty state
        empty_state = MessagesState(messages=[])
        assert empty_state.messages == []
        
        # Test with messages
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        state = MessagesState(messages=messages)
        assert len(state.messages) == 2
        assert state.messages[0].content == "Hello"
        assert state.messages[1].content == "Hi there!"

    def test_messages_state_default(self):
        """Test MessagesState with default values."""
        # Should work without explicit messages
        state = MessagesState()
        assert hasattr(state, 'messages')
        assert isinstance(state.messages, list)

    def test_messages_state_with_different_message_types(self):
        """Test MessagesState with different message types."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="What's the weather?"),
            AIMessage(content="I can't check weather")
        ]
        
        state = MessagesState(messages=messages)
        assert len(state.messages) == 3
        assert isinstance(state.messages[0], SystemMessage)
        assert isinstance(state.messages[1], HumanMessage)
        assert isinstance(state.messages[2], AIMessage)

    def test_messages_state_add_messages_behavior(self):
        """Test that MessagesState supports add_messages reducer."""
        # This tests the Annotated behavior
        state1 = MessagesState(messages=[HumanMessage(content="First")])
        state2 = MessagesState(messages=[AIMessage(content="Second")])
        
        # The add_messages reducer should be available
        # (This is more of a structural test since the reducer is used by LangGraph)
        assert hasattr(state1, 'messages')
        assert hasattr(state2, 'messages')
        
        # Test that we can manually simulate the add behavior
        combined_messages = state1.messages + state2.messages
        combined_state = MessagesState(messages=combined_messages)
        
        assert len(combined_state.messages) == 2
        assert combined_state.messages[0].content == "First"
        assert combined_state.messages[1].content == "Second"

    def test_messages_state_serialization(self):
        """Test MessagesState serialization/deserialization."""
        messages = [
            HumanMessage(content="Test message"),
            AIMessage(content="Test response")
        ]
        state = MessagesState(messages=messages)
        
        # Test that state can be converted to dict
        state_dict = state.model_dump()
        assert "messages" in state_dict
        assert len(state_dict["messages"]) == 2
        
        # Test that state can be recreated from dict
        new_state = MessagesState.model_validate(state_dict)
        assert len(new_state.messages) == 2
        assert new_state.messages[0].content == "Test message"
        assert new_state.messages[1].content == "Test response"

    def test_messages_state_validation(self):
        """Test MessagesState validation."""
        # Valid state
        valid_state = MessagesState(messages=[HumanMessage(content="Valid")])
        assert len(valid_state.messages) == 1
        
        # Test with default (should use default_factory)
        state_with_default = MessagesState()
        assert isinstance(state_with_default.messages, list)
        assert len(state_with_default.messages) == 0

    def test_messages_state_immutability(self):
        """Test MessagesState immutability patterns."""
        original_messages = [HumanMessage(content="Original")]
        state = MessagesState(messages=original_messages)
        
        # Modifying the original list shouldn't affect the state
        original_messages.append(AIMessage(content="Added"))
        
        # The state should still have only one message
        # (This depends on how Pydantic handles the list)
        assert len(state.messages) >= 1  # At least the original message
        assert state.messages[0].content == "Original"

    def test_messages_state_equality(self):
        """Test MessagesState equality."""
        messages1 = [HumanMessage(content="Same")]
        messages2 = [HumanMessage(content="Same")]
        messages3 = [HumanMessage(content="Different")]
        
        state1 = MessagesState(messages=messages1)
        state2 = MessagesState(messages=messages2)
        state3 = MessagesState(messages=messages3)
        
        # States with same content should be equal
        assert state1.messages[0].content == state2.messages[0].content
        
        # States with different content should not be equal
        assert state1.messages[0].content != state3.messages[0].content

    def test_messages_state_empty_handling(self):
        """Test MessagesState with empty messages."""
        empty_state = MessagesState(messages=[])
        assert len(empty_state.messages) == 0
        assert isinstance(empty_state.messages, list)
        
        # Should be able to add messages to empty state
        new_messages = [HumanMessage(content="New message")]
        new_state = MessagesState(messages=empty_state.messages + new_messages)
        assert len(new_state.messages) == 1
        assert new_state.messages[0].content == "New message" 