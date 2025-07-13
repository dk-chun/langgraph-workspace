"""
Tests for basic agent implementation.
"""

import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage, AIMessage

from my_agent.agents.basic import create_basic_agent
from my_agent.states.messages import MessagesState


class TestBasicAgent:
    """Test cases for basic agent."""

    def test_create_basic_agent(self):
        """Test basic agent creation."""
        agent = create_basic_agent()
        
        assert agent is not None
        assert hasattr(agent, 'invoke')
        assert hasattr(agent, 'get_graph')

    def test_agent_graph_structure(self):
        """Test agent graph structure."""
        agent = create_basic_agent()
        graph = agent.get_graph()
        
        # Check nodes - nodes is a dict[str, Node]
        node_names = list(graph.nodes.keys())
        assert "ollama" in node_names
        
        # Check edges
        edges = [(edge.source, edge.target) for edge in graph.edges]
        assert ("__start__", "ollama") in edges
        assert ("ollama", "__end__") in edges

    @patch('my_agent.nodes.models.ollama.ChatOllama')
    def test_agent_invoke_basic(self, mock_chat_ollama, basic_ollama_config):
        """Test basic agent invocation with OllamaConfig."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="테스트 응답입니다.")
        mock_chat_ollama.return_value = mock_instance
        
        # Create agent and invoke
        agent = create_basic_agent()
        result = agent.invoke(
            {"messages": [HumanMessage(content="안녕하세요")]},
            basic_ollama_config
        )
        
        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 2  # Human + AI message
        assert isinstance(result["messages"][0], HumanMessage)
        assert isinstance(result["messages"][1], AIMessage)
        assert result["messages"][1].content == "테스트 응답입니다."
        
        # Verify ChatOllama was called with correct config
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["base_url"] == "http://localhost:11434"
        assert call_args[1]["model"] == "qwen3:0.6b"
        assert call_args[1]["temperature"] == 0.7

    @patch('my_agent.nodes.models.ollama.ChatOllama')
    def test_agent_invoke_with_system_prompt(self, mock_chat_ollama, basic_ollama_config):
        """Test agent invocation with system prompt."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="시스템 프롬프트 적용된 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Modify config to include system prompt
        config_with_system = basic_ollama_config.copy()
        config_with_system["configurable"]["system_prompt"] = "You are a helpful assistant."
        
        # Create agent and invoke
        agent = create_basic_agent()
        result = agent.invoke(
            {"messages": [HumanMessage(content="안녕하세요")]},
            config_with_system
        )
        
        # Verify mock was called with system message
        mock_instance.invoke.assert_called_once()
        call_args = mock_instance.invoke.call_args[0][0]  # messages argument
        
        # Should have system message + human message
        assert len(call_args) == 2
        assert call_args[0].content == "You are a helpful assistant."
        assert call_args[1].content == "안녕하세요"

    @patch('my_agent.nodes.models.ollama.ChatOllama')
    def test_agent_invoke_with_minimal_config(self, mock_chat_ollama, minimal_ollama_config):
        """Test agent invocation with minimal configuration."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="기본 설정 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Create agent and invoke
        agent = create_basic_agent()
        result = agent.invoke(
            {"messages": [HumanMessage(content="테스트")]},
            minimal_ollama_config
        )
        
        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "기본 설정 응답"
        
        # Verify default values were used
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["base_url"] == "http://localhost:11434"  # default
        assert call_args[1]["model"] == "qwen3:0.6b"  # default
        assert call_args[1]["temperature"] == 0.7  # default

    @patch('my_agent.nodes.models.ollama.ChatOllama')
    def test_agent_invoke_with_custom_options(self, mock_chat_ollama, basic_ollama_config):
        """Test agent invocation with custom Ollama options."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="커스텀 옵션 응답")
        mock_chat_ollama.return_value = mock_instance
        
        # Add custom options to config
        config_with_options = basic_ollama_config.copy()
        config_with_options["configurable"].update({
            "num_ctx": 4096,
            "num_predict": 512,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        })
        
        # Create agent and invoke
        agent = create_basic_agent()
        result = agent.invoke(
            {"messages": [HumanMessage(content="테스트")]},
            config_with_options
        )
        
        # Verify custom options were passed
        mock_chat_ollama.assert_called_once()
        call_args = mock_chat_ollama.call_args
        assert call_args[1]["num_ctx"] == 4096
        assert call_args[1]["num_predict"] == 512
        assert call_args[1]["top_k"] == 40
        assert call_args[1]["top_p"] == 0.9
        assert call_args[1]["repeat_penalty"] == 1.1

    def test_agent_invoke_empty_messages(self, basic_ollama_config):
        """Test agent invocation with empty messages."""
        agent = create_basic_agent()
        
        # This should work with empty messages
        result = agent.invoke(
            {"messages": []},
            basic_ollama_config
        )
        
        # Should still have messages (at least the AI response)
        assert "messages" in result 