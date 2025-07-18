"""
Tests for core prompt functionality.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from gta.core.prompt import format_prompt_template


class TestPromptFunctions:
    """Test cases for core prompt functions."""
    
    def test_format_prompt_template_basic(self):
        """Test basic prompt template formatting."""
        template_messages = [
            ("system", "You are a helpful assistant."),
            ("user", "Hello, {name}!")
        ]
        variables = {"name": "Alice"}
        
        result = format_prompt_template(template_messages, variables)
        
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are a helpful assistant."
        assert isinstance(result[1], HumanMessage)
        assert result[1].content == "Hello, Alice!"
    
    def test_format_prompt_template_multiple_variables(self):
        """Test prompt template with multiple variables."""
        template_messages = [
            ("system", "You are a {role}."),
            ("user", "My name is {name} and I am {age} years old.")
        ]
        variables = {
            "role": "helpful assistant",
            "name": "Bob",
            "age": "25"
        }
        
        result = format_prompt_template(template_messages, variables)
        
        assert len(result) == 2
        assert result[0].content == "You are a helpful assistant."
        assert result[1].content == "My name is Bob and I am 25 years old."
    
    def test_format_prompt_template_no_variables(self):
        """Test prompt template without variables."""
        template_messages = [
            ("system", "You are a helpful assistant."),
            ("user", "Hello!")
        ]
        variables = {}
        
        result = format_prompt_template(template_messages, variables)
        
        assert len(result) == 2
        assert result[0].content == "You are a helpful assistant."
        assert result[1].content == "Hello!"
    
    def test_format_prompt_template_single_message(self):
        """Test prompt template with single message."""
        template_messages = [("user", "What is {topic}?")]
        variables = {"topic": "artificial intelligence"}
        
        result = format_prompt_template(template_messages, variables)
        
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "What is artificial intelligence?"
    
    def test_format_prompt_template_complex_template(self):
        """Test prompt template with complex formatting."""
        template_messages = [
            ("system", "You are an expert in {domain}."),
            ("user", "Context: {context}\n\nQuestion: {question}\n\nPlease provide a detailed answer.")
        ]
        variables = {
            "domain": "machine learning",
            "context": "Deep learning is a subset of machine learning.",
            "question": "What is deep learning?"
        }
        
        result = format_prompt_template(template_messages, variables)
        
        assert len(result) == 2
        assert result[0].content == "You are an expert in machine learning."
        expected_user_content = (
            "Context: Deep learning is a subset of machine learning.\n\n"
            "Question: What is deep learning?\n\n"
            "Please provide a detailed answer."
        )
        assert result[1].content == expected_user_content
    
    def test_format_prompt_template_empty_template(self):
        """Test prompt template with empty template list."""
        template_messages = []
        variables = {}
        
        result = format_prompt_template(template_messages, variables)
        
        assert len(result) == 0
    
    def test_format_prompt_template_assistant_role(self):
        """Test prompt template with assistant role."""
        template_messages = [
            ("user", "Hello"),
            ("assistant", "Hi there! How can I help you today?"),
            ("user", "What is {topic}?")
        ]
        variables = {"topic": "Python"}
        
        result = format_prompt_template(template_messages, variables)
        
        assert len(result) == 3
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there! How can I help you today?"
        assert result[2].content == "What is Python?"
    
    def test_format_prompt_template_missing_variable(self):
        """Test prompt template with missing variable should raise error."""
        template_messages = [("user", "Hello {name}!")]
        variables = {}  # Missing 'name' variable
        
        with pytest.raises(KeyError):
            format_prompt_template(template_messages, variables)
    
    def test_format_prompt_template_extra_variables(self):
        """Test prompt template with extra variables (should be ignored)."""
        template_messages = [("user", "Hello {name}!")]
        variables = {
            "name": "Alice",
            "extra": "unused"
        }
        
        result = format_prompt_template(template_messages, variables)
        
        assert len(result) == 1
        assert result[0].content == "Hello Alice!"
    
    @patch('gta.core.prompt.ChatPromptTemplate')
    def test_format_prompt_template_uses_chat_prompt_template(self, mock_chat_prompt):
        """Test that function uses ChatPromptTemplate correctly."""
        mock_template = Mock()
        mock_prompt_value = Mock()
        mock_messages = [HumanMessage(content="Test")]
        
        mock_chat_prompt.return_value = mock_template
        mock_template.invoke.return_value = mock_prompt_value
        mock_prompt_value.to_messages.return_value = mock_messages
        
        template_messages = [("user", "Test {var}")]
        variables = {"var": "value"}
        
        result = format_prompt_template(template_messages, variables)
        
        mock_chat_prompt.assert_called_once_with(template_messages)
        mock_template.invoke.assert_called_once_with(variables)
        mock_prompt_value.to_messages.assert_called_once()
        assert result == mock_messages 