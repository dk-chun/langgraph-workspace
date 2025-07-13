"""
Tests for Ollama node implementation.
"""

import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from gta.nodes.models.ollama import ollama_node
from gta.states.messages import MessagesState


class TestOllamaNode:
    """Test cases for Ollama node."""

    @patch('gta.nodes.models.ollama.ChatOllama')
    def test_ollama_node_basic(self, mock_chat_ollama, empty_messages_state, runnable_config):
        """Test basic Ollama node functionality."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="테스트 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Call node
        result = ollama_node(empty_messages_state, runnable_config)
        
        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "테스트 응답"
        
        # Verify ChatOllama was configured correctly
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["base_url"] == "http://localhost:11434"
        assert call_args[1]["model"] == "qwen3:0.6b"
        assert call_args[1]["temperature"] == 0.7

    @patch('gta.nodes.models.ollama.ChatOllama')
    def test_ollama_node_with_existing_messages(self, mock_chat_ollama, sample_messages_state, runnable_config):
        """Test Ollama node with existing messages."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="기존 메시지 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Call node
        result = ollama_node(sample_messages_state, runnable_config)
        
        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "기존 메시지 응답"
        
        # Verify existing messages were passed to LLM
        mock_instance.invoke.assert_called_once()
        call_args = mock_instance.invoke.call_args[0][0]  # messages argument
        assert len(call_args) == 3  # System message + original messages
        assert call_args[0].content == "You are a helpful assistant."  # System message
        assert call_args[1].content == "안녕하세요"
        assert call_args[2].content == "안녕하세요! 무엇을 도와드릴까요?"

    @patch('gta.nodes.models.ollama.ChatOllama')
    def test_ollama_node_with_system_prompt(self, mock_chat_ollama, empty_messages_state, basic_ollama_config):
        """Test Ollama node with system prompt."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="시스템 프롬프트 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Create config with system prompt
        config = RunnableConfig(configurable=basic_ollama_config["configurable"])
        
        # Call node
        result = ollama_node(empty_messages_state, config)
        
        # Verify system message was added
        mock_instance.invoke.assert_called_once()
        call_args = mock_instance.invoke.call_args[0][0]  # messages argument
        assert len(call_args) == 1  # Only system message (no user messages)
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "You are a helpful assistant."

    @patch('gta.nodes.models.ollama.ChatOllama')
    def test_ollama_node_with_custom_options(self, mock_chat_ollama, empty_messages_state):
        """Test Ollama node with custom options."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="커스텀 옵션 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Create config with custom options
        config = RunnableConfig(configurable={
            "thread_id": "test",
            "model_name": "custom-model",
            "temperature": 0.5,
            "base_url": "http://custom:11434",
            "num_ctx": 8192,
            "num_predict": 256,
            "top_k": 20,
            "top_p": 0.8,
            "repeat_penalty": 1.2,
            "timeout": 60,
            "keep_alive": "5m"
        })
        
        # Call node
        result = ollama_node(empty_messages_state, config)
        
        # Verify custom configuration was used
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["base_url"] == "http://custom:11434"
        assert call_args[1]["model"] == "custom-model"
        assert call_args[1]["temperature"] == 0.5
        assert call_args[1]["timeout"] == 60
        assert call_args[1]["keep_alive"] == "5m"
        assert call_args[1]["num_ctx"] == 8192
        assert call_args[1]["num_predict"] == 256
        assert call_args[1]["top_k"] == 20
        assert call_args[1]["top_p"] == 0.8
        assert call_args[1]["repeat_penalty"] == 1.2

    @patch('gta.nodes.models.ollama.ChatOllama')
    def test_ollama_node_with_minimal_config(self, mock_chat_ollama, empty_messages_state):
        """Test Ollama node with minimal configuration."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="최소 설정 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Create minimal config
        config = RunnableConfig(configurable={"thread_id": "test"})
        
        # Call node
        result = ollama_node(empty_messages_state, config)
        
        # Verify default values were used
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["base_url"] == "http://localhost:11434"  # default
        assert call_args[1]["model"] == "qwen3:0.6b"  # default
        assert call_args[1]["temperature"] == 0.7  # default

    @patch('gta.nodes.models.ollama.ChatOllama')
    def test_ollama_node_with_model_kwargs(self, mock_chat_ollama, empty_messages_state):
        """Test Ollama node with additional model kwargs."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="모델 kwargs 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Create config with model kwargs
        config = RunnableConfig(configurable={
            "thread_id": "test",
            "model_kwargs": {
                "custom_param": "custom_value",
                "another_param": 42
            }
        })
        
        # Call node
        result = ollama_node(empty_messages_state, config)
        
        # Verify model kwargs were passed
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["custom_param"] == "custom_value"
        assert call_args[1]["another_param"] == 42

    @patch('gta.nodes.models.ollama.ChatOllama')
    def test_ollama_node_error_handling(self, mock_chat_ollama, empty_messages_state, runnable_config):
        """Test Ollama node error handling."""
        # Setup mock to raise exception
        mock_instance = Mock()
        mock_instance.invoke.side_effect = Exception("연결 오류")
        mock_chat_ollama.return_value = mock_instance
        
        # Call node
        result = ollama_node(empty_messages_state, runnable_config)
        
        # Verify error was handled gracefully
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "죄송합니다. 오류가 발생했습니다" in result["messages"][0].content
        assert "연결 오류" in result["messages"][0].content

    def test_ollama_node_empty_config(self, empty_messages_state):
        """Test Ollama node with empty config."""
        # Create empty config
        config = RunnableConfig(configurable={})
        
        # This should work with default values
        with patch('gta.nodes.models.ollama.ChatOllama') as mock_chat_ollama:
            mock_instance = Mock()
            mock_instance.invoke.return_value = Mock(content="빈 설정 응답")
            mock_chat_ollama.return_value = mock_instance
            
            result = ollama_node(empty_messages_state, config)
            
            # Verify it worked with defaults
            assert "messages" in result
            assert result["messages"][0].content == "빈 설정 응답" 