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


@pytest.fixture
def mock_vectorstores():
    """Mock vector stores for multi-RAG testing."""
    vs1 = Mock(spec=VectorStore)
    vs2 = Mock(spec=VectorStore) 
    vs3 = Mock(spec=VectorStore)
    
    # Mock document objects
    mock_doc1 = Mock()
    mock_doc1.page_content = "Document from vectorstore 1"
    mock_doc2 = Mock()
    mock_doc2.page_content = "Document from vectorstore 2"
    mock_doc3 = Mock()
    mock_doc3.page_content = "Document from vectorstore 3"
    
    # Mock search results with scores
    vs1.similarity_search_with_score.return_value = [(mock_doc1, 0.9)]
    vs2.similarity_search_with_score.return_value = [(mock_doc2, 0.8)]
    vs3.similarity_search_with_score.return_value = [(mock_doc3, 0.7)]
    
    return [vs1, vs2, vs3]


@pytest.fixture
def multi_rag_sample_state():
    """Sample MultiRAG state for testing."""
    from gta.agents.multi_rag.state import MultiRAGState
    
    mock_doc = Mock()
    mock_doc.page_content = "Sample document content"
    
    return MultiRAGState(
        question="What is AI?",
        vs1_documents=[(mock_doc, 0.9)],
        vs2_documents=[(mock_doc, 0.8)],
        vs3_documents=[(mock_doc, 0.7)],
        search_scores={"vs1": 0.9, "vs2": 0.8, "vs3": 0.7},
        messages=[HumanMessage(content="What is AI?")]
    ) 