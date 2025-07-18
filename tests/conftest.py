"""
Pytest configuration and common fixtures.
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import VectorStore


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock(spec=BaseChatModel)
    llm.invoke.return_value = AIMessage(content="Mock response")
    return llm


@pytest.fixture
def mock_vectorstore():
    """Mock vector store for testing."""
    vectorstore = Mock(spec=VectorStore)
    vectorstore.similarity_search.return_value = [
        Mock(page_content="Mock document 1"),
        Mock(page_content="Mock document 2")
    ]
    return vectorstore


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!")
    ]


@pytest.fixture
def sample_state():
    """Sample state for testing."""
    return {
        "messages": [HumanMessage(content="Test message")],
        "query": "test query",
        "context": "test context"
    } 