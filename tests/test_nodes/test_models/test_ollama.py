"""
Tests for Ollama node implementation.
"""

import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from gta.nodes.models.ollama_node import ollama_node
from gta.states.messages_state import MessagesState


class TestOllamaNode:
    """Test cases for Ollama node."""

    @patch('gta.nodes.models.ollama_node.ChatOllama')
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

    @patch('gta.nodes.models.ollama_node.ChatOllama')
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
        assert len(result["messages"]) == 1  # Only new AI message
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "기존 메시지 응답"

    @patch('gta.nodes.models.ollama_node.ChatOllama')
    def test_ollama_node_with_system_prompt(self, mock_chat_ollama, empty_messages_state, runnable_config_with_system):
        """Test Ollama node with system prompt."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="시스템 프롬프트 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Call node
        result = ollama_node(empty_messages_state, runnable_config_with_system)
        
        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "시스템 프롬프트 응답"
        
        # Verify invoke was called with system message
        mock_instance.invoke.assert_called_once()
        invoke_messages = mock_instance.invoke.call_args[0][0]
        assert len(invoke_messages) == 1
        assert invoke_messages[0].content == "당신은 도움이 되는 AI 어시스턴트입니다."

    @patch('gta.nodes.models.ollama_node.ChatOllama')
    def test_ollama_node_with_custom_options(self, mock_chat_ollama, empty_messages_state, runnable_config_with_options):
        """Test Ollama node with custom options."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="커스텀 옵션 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Call node
        result = ollama_node(empty_messages_state, runnable_config_with_options)
        
        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "커스텀 옵션 응답"
        
        # Verify ChatOllama was configured with custom options
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["base_url"] == "http://custom:11434"
        assert call_args[1]["model"] == "custom-model"
        assert call_args[1]["temperature"] == 0.5
        assert call_args[1]["num_predict"] == 100
        assert call_args[1]["keep_alive"] == "10m"
        assert call_args[1]["num_ctx"] == 4096
        assert call_args[1]["repeat_penalty"] == 1.1
        assert call_args[1]["top_k"] == 50
        assert call_args[1]["top_p"] == 0.9

    @patch('gta.nodes.models.ollama_node.ChatOllama')
    def test_ollama_node_with_minimal_config(self, mock_chat_ollama, empty_messages_state, minimal_runnable_config):
        """Test Ollama node with minimal configuration."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="최소 설정 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Call node
        result = ollama_node(empty_messages_state, minimal_runnable_config)
        
        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "최소 설정 응답"
        
        # Verify ChatOllama was configured with defaults
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["base_url"] == "http://localhost:11434"
        assert call_args[1]["model"] == "qwen3:0.6b"
        assert call_args[1]["temperature"] == 0.7

    @patch('gta.nodes.models.ollama_node.ChatOllama')
    def test_ollama_node_with_model_kwargs(self, mock_chat_ollama, empty_messages_state, runnable_config_with_kwargs):
        """Test Ollama node with model kwargs."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="모델 kwargs 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Call node
        result = ollama_node(empty_messages_state, runnable_config_with_kwargs)
        
        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "모델 kwargs 응답"
        
        # Verify ChatOllama was configured with model kwargs
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["custom_param"] == "custom_value"
        assert call_args[1]["another_param"] == 42

    @patch('gta.nodes.models.ollama_node.ChatOllama')
    def test_ollama_node_error_handling(self, mock_chat_ollama, empty_messages_state, runnable_config):
        """Test Ollama node error handling."""
        # Setup mock to raise exception
        mock_instance = Mock()
        mock_instance.invoke.side_effect = Exception("연결 실패")
        mock_chat_ollama.return_value = mock_instance
        
        # Call node
        result = ollama_node(empty_messages_state, runnable_config)
        
        # Verify error handling
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "연결 실패" in result["messages"][0].content
        assert "죄송합니다. 오류가 발생했습니다" in result["messages"][0].content

    def test_ollama_node_empty_config(self, empty_messages_state):
        """Test Ollama node with empty config."""
        empty_config = RunnableConfig(configurable={})
        
        # Mock ChatOllama directly in the test
        with patch('gta.nodes.models.ollama_node.ChatOllama') as mock_chat_ollama:
            mock_instance = Mock()
            mock_instance.invoke.return_value = Mock(content="빈 설정 응답")
            mock_chat_ollama.return_value = mock_instance
            
            # Call node
            result = ollama_node(empty_messages_state, empty_config)
            
            # Verify result
            assert "messages" in result
            assert len(result["messages"]) == 1
            assert isinstance(result["messages"][0], AIMessage)
            assert result["messages"][0].content == "빈 설정 응답"
            
            # Verify ChatOllama was configured with defaults
            mock_chat_ollama.assert_called_once()
            call_args = mock_chat_ollama.call_args
            assert call_args[1]["base_url"] == "http://localhost:11434"
            assert call_args[1]["model"] == "qwen3:0.6b"
            assert call_args[1]["temperature"] == 0.7 
