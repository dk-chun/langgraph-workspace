"""
pytest configuration and shared fixtures.
"""

import os
import pytest
import subprocess
import time
import requests
from typing import Generator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def check_ollama():
    """Check if Ollama is available and has required models."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
    
    try:
        # Check if Ollama server is running
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code != 200:
            pytest.skip("Ollama server not available")
        
        # Check if required model is available
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        
        if model_name not in model_names:
            pytest.skip(f"Required model {model_name} not found in Ollama")
            
        return True
        
    except requests.RequestException:
        pytest.skip("Ollama server not reachable")


@pytest.fixture(scope="session")
def langgraph_server(check_ollama) -> Generator[str, None, None]:
    """Check if LangGraph dev server is running (does not start server automatically)."""
    server_url = "http://localhost:7000"  # Expected server URL
    
    # Check if server is already running
    try:
        response = requests.get(f"{server_url}/ok", timeout=5)
        if response.status_code == 200:
            print(f"✅ LangGraph server found at {server_url}")
            yield server_url
            return
    except requests.RequestException:
        pass
    
    # Try alternative ports
    for port in [8000, 2024, 8080]:
        alt_url = f"http://localhost:{port}"
        try:
            response = requests.get(f"{alt_url}/ok", timeout=2)
            if response.status_code == 200:
                print(f"✅ LangGraph server found at {alt_url}")
                yield alt_url
                return
        except requests.RequestException:
            continue
    
    # Server not found
    pytest.skip(
        "LangGraph server not running. Please start it manually:\n"
        "  uv run langgraph dev --port 7000\n"
        "Then run the tests again."
    )


@pytest.fixture
def basic_agent_config():
    """Default configuration for basic agent testing."""
    return {
        "model_type": "ollama",
        "model_name": os.getenv("OLLAMA_MODEL", "deepseek-r1:latest"),
        "temperature": 0.1  # Low temperature for consistent testing
    }


@pytest.fixture
def test_messages():
    """Test messages for different templates."""
    return {
        "chat": "Hello! How are you?",
        "summarize": "Please summarize this text: The quick brown fox jumps over the lazy dog. This is a common English pangram used for testing.",
        "translate": "Translate to Korean: Hello, how are you today?",
        "explain": "Explain how photosynthesis works in simple terms.",
        "code": "Write a Python function to calculate factorial.",
        "creative": "Write a short poem about coding.",
        "analyze": "Analyze the pros and cons of remote work."
    } 