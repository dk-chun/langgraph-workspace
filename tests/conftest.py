"""
Pytest configuration and common fixtures.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from my_agent.agents.basic import create_basic_agent
from my_agent.states.messages import MessagesState


@pytest.fixture
def sample_human_message():
    """Sample human message for testing."""
    return HumanMessage(content="안녕하세요")


@pytest.fixture
def sample_ai_message():
    """Sample AI message for testing."""
    return AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")


@pytest.fixture
def sample_system_message():
    """Sample system message for testing."""
    return SystemMessage(content="You are a helpful assistant.")


@pytest.fixture
def sample_messages_state():
    """Sample MessagesState for testing."""
    return MessagesState(
        messages=[
            HumanMessage(content="안녕하세요"),
            AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")
        ]
    )


@pytest.fixture
def empty_messages_state():
    """Empty MessagesState for testing."""
    return MessagesState(messages=[])


@pytest.fixture
def basic_ollama_config():
    """Basic Ollama configuration for testing."""
    return {
        "configurable": {
            "thread_id": "test-thread",
            "model_name": "qwen3:0.6b",
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant.",
            "base_url": "http://localhost:11434",
            "max_tokens": 1000,
            "timeout": 30
        }
    }


@pytest.fixture
def minimal_ollama_config():
    """Minimal Ollama configuration for testing."""
    return {
        "configurable": {
            "thread_id": "test-minimal"
        }
    }


@pytest.fixture
def runnable_config(basic_ollama_config):
    """RunnableConfig instance for testing."""
    return RunnableConfig(**basic_ollama_config)


@pytest.fixture
def basic_agent():
    """Basic agent instance for testing."""
    return create_basic_agent()


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama response for testing."""
    mock_response = Mock()
    mock_response.content = "테스트 응답입니다."
    return mock_response


@pytest.fixture
def mock_chat_ollama():
    """Mock ChatOllama instance for testing."""
    with patch('langchain_ollama.ChatOllama') as mock:
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(content="테스트 응답입니다.")
        mock.return_value = mock_instance
        yield mock_instance 