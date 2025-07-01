"""
RAG components for document processing and retrieval.
"""

import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGComponents:
    """
    RAG components for document processing and retrieval.
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 embedding_model_type: str = None,
                 embedding_model_name: str = None,
                 embedding_base_url: str = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize RAG components.
        
        Args:
            persist_directory: Directory to persist vector database
            embedding_model_type: Type of embedding model ('ollama' or 'huggingface')
            embedding_model_name: Name of the embedding model
            embedding_base_url: Base URL for embedding service (for Ollama)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        
        # Get embedding configuration from environment or use defaults
        self.embedding_model_type = embedding_model_type or os.getenv("EMBEDDING_MODEL_TYPE", "huggingface")
        self.embedding_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")
        
        # Initialize embeddings based on type
        self.embeddings = self._initialize_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_embeddings(self):
        """Initialize embedding model based on configuration."""
        if self.embedding_model_type.lower() == "ollama":
            print(f"Using Ollama embeddings: {self.embedding_model_name}")
            return OllamaEmbeddings(
                base_url=self.embedding_base_url,
                model=self.embedding_model_name
            )
        else:
            print(f"Using HuggingFace embeddings: {self.embedding_model_name}")
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store."""
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            # Create empty vectorstore
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents from file or directory.
        
        Args:
            file_path: Path to file or directory
            
        Returns:
            List of loaded documents
        """
        if os.path.isdir(file_path):
            loader = DirectoryLoader(
                file_path,
                glob="**/*.{txt,pdf}",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
        
        return loader.load()
    
    def add_documents(self, file_path: str):
        """
        Add documents to vector store.
        
        Args:
            file_path: Path to file or directory to add
        """
        documents = self.load_documents(file_path)
        chunks = self.text_splitter.split_documents(documents)
        
        if chunks:
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()
            print(f"Added {len(chunks)} document chunks to vector store.")
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        if not self.vectorstore:
            return []
        
        # Get documents with similarity scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        retrieved_docs = []
        for doc, score in docs_with_scores:
            retrieved_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")
        
        return "\n\n".join(context_parts)
    
    def clear_vectorstore(self):
        """Clear all documents from vector store."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            self._initialize_vectorstore()


# Default RAG components instance
default_rag_components = RAGComponents() 