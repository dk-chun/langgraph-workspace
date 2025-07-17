"""
Pytest configuration and common fixtures.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from gta.agents.basic_agent import create_basic_agent
from gta.states.messages_state import MessagesState


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
            "keep_alive": "5m"
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


@pytest.fixture
def runnable_config_with_system():
    """RunnableConfig with system prompt for testing."""
    return RunnableConfig(configurable={
        "thread_id": "test-thread",
        "model_name": "qwen3:0.6b",
        "temperature": 0.7,
        "system_prompt": "당신은 도움이 되는 AI 어시스턴트입니다.",
        "base_url": "http://localhost:11434"
    })


@pytest.fixture
def runnable_config_with_options():
    """RunnableConfig with custom options for testing."""
    return RunnableConfig(configurable={
        "thread_id": "test-thread",
        "model_name": "custom-model",
        "temperature": 0.5,
        "base_url": "http://custom:11434",
        "num_predict": 100,
        "keep_alive": "10m",
        "num_ctx": 4096,
        "repeat_penalty": 1.1,
        "top_k": 50,
        "top_p": 0.9
    })


@pytest.fixture
def minimal_runnable_config():
    """Minimal RunnableConfig for testing."""
    return RunnableConfig(configurable={
        "thread_id": "test-minimal"
    })


@pytest.fixture
def runnable_config_with_kwargs():
    """RunnableConfig with model kwargs for testing."""
    return RunnableConfig(configurable={
        "thread_id": "test-thread",
        "model_kwargs": {
            "custom_param": "custom_value",
            "another_param": 42
        }
    }) 