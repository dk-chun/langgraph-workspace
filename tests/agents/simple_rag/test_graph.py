"""
Tests for simple_rag agent graph.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from gta.agents.simple_rag.graph import create_rag_graph
from gta.agents.simple_rag.state import RAGState


class TestRAGGraph:
    """Test cases for RAG agent graph."""
    
    def test_create_rag_graph_with_parameters(self, mock_llm, mock_vectorstore):
        """Test creating RAG graph with all parameters."""
        template_messages = [("system", "You are a helpful assistant.")]
        
        graph = create_rag_graph(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            template_messages=template_messages
        )
        
        assert graph is not None
        assert hasattr(graph, 'invoke')
    
    @patch('gta.agents.simple_rag.graph.ChatOllama')
    @patch('gta.agents.simple_rag.graph.OllamaEmbeddings')
    @patch('gta.agents.simple_rag.graph.QdrantVectorStore')
    def test_create_rag_graph_with_defaults(self, mock_vectorstore_class, mock_embeddings, mock_chat_ollama):
        """Test creating RAG graph with default parameters."""
        mock_ollama_instance = Mock()
        mock_chat_ollama.return_value = mock_ollama_instance
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore_instance = Mock()
        mock_vectorstore_class.return_value = mock_vectorstore_instance
        
        graph = create_rag_graph()
        
        assert graph is not None
        mock_chat_ollama.assert_called_once()
        mock_embeddings.assert_called_once()
    
    def test_create_rag_graph_custom_llm_only(self, mock_llm):
        """Test creating RAG graph with custom LLM only."""
        with patch('gta.agents.simple_rag.graph.OllamaEmbeddings') as mock_embeddings, \
             patch('gta.agents.simple_rag.graph.QdrantVectorStore') as mock_vectorstore_class:
            
            mock_embeddings_instance = Mock()
            mock_embeddings.return_value = mock_embeddings_instance
            
            mock_vectorstore_instance = Mock()
            mock_vectorstore_class.return_value = mock_vectorstore_instance
            
            graph = create_rag_graph(llm=mock_llm)
            
            assert graph is not None
            mock_embeddings.assert_called_once()
    
    def test_create_rag_graph_custom_vectorstore_only(self, mock_vectorstore):
        """Test creating RAG graph with custom vectorstore only."""
        with patch('gta.agents.simple_rag.graph.ChatOllama') as mock_chat_ollama:
            mock_ollama_instance = Mock()
            mock_chat_ollama.return_value = mock_ollama_instance
            
            graph = create_rag_graph(vectorstore=mock_vectorstore)
            
            assert graph is not None
            mock_chat_ollama.assert_called_once()
    
    def test_rag_graph_execution_flow(self, mock_llm, mock_vectorstore):
        """Test RAG graph execution with complete flow."""
        # Setup mocks
        with patch('gta.agents.simple_rag.nodes.search_documents') as mock_search, \
             patch('gta.agents.simple_rag.nodes.format_search_results') as mock_format, \
             patch('gta.agents.simple_rag.nodes.format_prompt_template') as mock_prompt:
            
            mock_docs = [Mock(page_content="AI is artificial intelligence")]
            mock_search.return_value = mock_docs
            mock_format.return_value = "Formatted context"
            mock_prompt.return_value = [HumanMessage(content="Formatted prompt")]
            
            graph = create_rag_graph(llm=mock_llm, vectorstore=mock_vectorstore)
            
            # Test input
            initial_state = {"messages": [HumanMessage(content="What is AI?")]}
            
            # Execute graph
            result = graph.invoke(initial_state)
            
            # Verify
            assert "messages" in result
            assert "question" in result
            assert "context" in result
            assert "documents" in result
            
            # Should have original message + AI response
            assert len(result["messages"]) >= 2
            mock_llm.invoke.assert_called()
    
    def test_rag_graph_with_template_messages(self, mock_llm, mock_vectorstore):
        """Test RAG graph with custom template messages."""
        template_messages = [
            ("system", "You are an expert assistant."),
            ("user", "Context: {context}\nQuestion: {question}")
        ]
        
        with patch('gta.agents.simple_rag.nodes.search_documents') as mock_search, \
             patch('gta.agents.simple_rag.nodes.format_search_results') as mock_format, \
             patch('gta.agents.simple_rag.nodes.format_prompt_template') as mock_prompt:
            
            mock_search.return_value = []
            mock_format.return_value = "Context"
            mock_prompt.return_value = [HumanMessage(content="Formatted prompt")]
            
            graph = create_rag_graph(
                llm=mock_llm,
                vectorstore=mock_vectorstore,
                template_messages=template_messages
            )
            
            initial_state = {"messages": [HumanMessage(content="Test question")]}
            result = graph.invoke(initial_state)
            
            assert "messages" in result
            mock_prompt.assert_called()
    
    def test_rag_graph_empty_messages(self, mock_llm, mock_vectorstore):
        """Test RAG graph with empty initial messages."""
        with patch('gta.agents.simple_rag.nodes.search_documents') as mock_search, \
             patch('gta.agents.simple_rag.nodes.format_search_results') as mock_format:
            
            mock_search.return_value = []
            mock_format.return_value = "No context"
            
            graph = create_rag_graph(llm=mock_llm, vectorstore=mock_vectorstore)
            
            initial_state = {"messages": []}
            result = graph.invoke(initial_state)
            
            assert "messages" in result
            assert "question" in result
            assert result["question"] == ""  # Should extract empty question 