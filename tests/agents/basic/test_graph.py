"""
Tests for basic agent graph.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph

from gta.agents.basic.graph import create_basic_graph
from gta.agents.basic.state import BasicState


class TestBasicGraph:
    """Test cases for basic agent graph."""
    
    def test_create_basic_graph_default(self, mock_llm):
        """Test creating basic graph with default parameters."""
        graph = create_basic_graph(llm=mock_llm)
        
        assert graph is not None
        assert hasattr(graph, 'invoke')
    
    def test_create_basic_graph_with_system_prompt(self, mock_llm):
        """Test creating basic graph with system prompt."""
        system_prompt = "You are a helpful assistant."
        graph = create_basic_graph(llm=mock_llm, system_prompt=system_prompt)
        
        assert graph is not None
        assert hasattr(graph, 'invoke')
    
    @patch('gta.agents.basic.graph.ChatOllama')
    def test_create_basic_graph_without_llm(self, mock_chat_ollama):
        """Test creating basic graph without providing LLM (should use default)."""
        mock_ollama_instance = Mock()
        mock_chat_ollama.return_value = mock_ollama_instance
        
        graph = create_basic_graph()
        
        assert graph is not None
        mock_chat_ollama.assert_called_once()
    
    def test_basic_graph_execution(self, mock_llm):
        """Test basic graph execution with a message."""
        graph = create_basic_graph(llm=mock_llm)
        
        # Test input
        initial_state = {"messages": [HumanMessage(content="Hello")]}
        
        # Execute graph
        result = graph.invoke(initial_state)
        
        # Verify
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response
        assert isinstance(result["messages"][1], AIMessage)
        mock_llm.invoke.assert_called_once()
    
    def test_basic_graph_with_system_prompt_execution(self, mock_llm):
        """Test basic graph execution with system prompt."""
        system_prompt = "You are a test assistant."
        graph = create_basic_graph(llm=mock_llm, system_prompt=system_prompt)
        
        initial_state = {"messages": [HumanMessage(content="Hello")]}
        result = graph.invoke(initial_state)
        
        assert "messages" in result
        assert len(result["messages"]) == 2
        mock_llm.invoke.assert_called_once()
    
    def test_basic_graph_multiple_messages(self, mock_llm):
        """Test basic graph with multiple messages."""
        graph = create_basic_graph(llm=mock_llm)
        
        initial_state = {
            "messages": [
                HumanMessage(content="First message"),
                AIMessage(content="First response"),
                HumanMessage(content="Second message")
            ]
        }
        
        result = graph.invoke(initial_state)
        
        assert "messages" in result
        assert len(result["messages"]) == 4  # 3 original + 1 new AI response
        mock_llm.invoke.assert_called_once() 