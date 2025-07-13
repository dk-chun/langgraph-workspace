"""
Tests for model configuration schemas.
"""

import pytest
from typing import get_type_hints

from gta.configs.model_config import ModelConfig, OllamaConfig


class TestModelConfig:
    """Test cases for ModelConfig."""

    def test_model_config_structure(self):
        """Test ModelConfig structure."""
        # Test that ModelConfig has expected fields
        config = ModelConfig()
        
        # Should be able to create empty config
        assert isinstance(config, dict)
        
        # Test with all fields
        full_config = ModelConfig(
            model_name="test-model",
            temperature=0.8,
            max_tokens=500,
            system_prompt="Test prompt",
            model_kwargs={"param": "value"}
        )
        
        assert full_config["model_name"] == "test-model"
        assert full_config["temperature"] == 0.8
        assert full_config["max_tokens"] == 500
        assert full_config["system_prompt"] == "Test prompt"
        assert full_config["model_kwargs"] == {"param": "value"}

    def test_model_config_partial(self):
        """Test ModelConfig with partial fields."""
        # Should work with only some fields
        partial_config = ModelConfig(
            model_name="partial-model",
            temperature=0.5
        )
        
        assert partial_config["model_name"] == "partial-model"
        assert partial_config["temperature"] == 0.5
        assert "max_tokens" not in partial_config
        assert "system_prompt" not in partial_config


class TestOllamaConfig:
    """Test cases for OllamaConfig."""

    def test_ollama_config_structure(self):
        """Test OllamaConfig structure."""
        # Test that OllamaConfig inherits from ModelConfig
        config = OllamaConfig()
        assert isinstance(config, dict)
        
        # Test with all fields
        full_config = OllamaConfig(
            # ModelConfig fields
            model_name="qwen3:0.6b",
            temperature=0.7,
            max_tokens=1000,
            system_prompt="You are helpful",
            model_kwargs={"custom": "value"},
            
            # OllamaConfig specific fields
            base_url="http://localhost:11434",
            timeout=30,
            keep_alive="5m",
            num_ctx=4096,
            num_predict=512,
            repeat_penalty=1.1,
            top_k=40,
            top_p=0.9
        )
        
        # Verify all fields are present
        assert full_config["model_name"] == "qwen3:0.6b"
        assert full_config["temperature"] == 0.7
        assert full_config["base_url"] == "http://localhost:11434"
        assert full_config["timeout"] == 30
        assert full_config["keep_alive"] == "5m"
        assert full_config["num_ctx"] == 4096
        assert full_config["num_predict"] == 512
        assert full_config["repeat_penalty"] == 1.1
        assert full_config["top_k"] == 40
        assert full_config["top_p"] == 0.9

    def test_ollama_config_minimal(self):
        """Test OllamaConfig with minimal fields."""
        # Should work with no fields
        minimal_config = OllamaConfig()
        assert isinstance(minimal_config, dict)
        assert len(minimal_config) == 0
        
        # Should work with only base fields
        base_config = OllamaConfig(
            model_name="test-model",
            base_url="http://test:11434"
        )
        
        assert base_config["model_name"] == "test-model"
        assert base_config["base_url"] == "http://test:11434"
        assert "temperature" not in base_config
        assert "timeout" not in base_config

    def test_ollama_config_server_settings(self):
        """Test OllamaConfig server-specific settings."""
        server_config = OllamaConfig(
            base_url="http://custom-server:8080",
            timeout=60,
            keep_alive="10m"
        )
        
        assert server_config["base_url"] == "http://custom-server:8080"
        assert server_config["timeout"] == 60
        assert server_config["keep_alive"] == "10m"

    def test_ollama_config_generation_params(self):
        """Test OllamaConfig generation parameters."""
        gen_config = OllamaConfig(
            num_ctx=8192,
            num_predict=256,
            repeat_penalty=1.2,
            top_k=20,
            top_p=0.8
        )
        
        assert gen_config["num_ctx"] == 8192
        assert gen_config["num_predict"] == 256
        assert gen_config["repeat_penalty"] == 1.2
        assert gen_config["top_k"] == 20
        assert gen_config["top_p"] == 0.8

    def test_ollama_config_with_model_kwargs(self):
        """Test OllamaConfig with model_kwargs (inherited from ModelConfig)."""
        kwargs_config = OllamaConfig(
            model_kwargs={
                "custom_param": "custom_value",
                "numeric_param": 42,
                "boolean_param": True
            }
        )
        
        assert kwargs_config["model_kwargs"]["custom_param"] == "custom_value"
        assert kwargs_config["model_kwargs"]["numeric_param"] == 42
        assert kwargs_config["model_kwargs"]["boolean_param"] is True

    def test_ollama_config_inheritance(self):
        """Test that OllamaConfig properly inherits from ModelConfig."""
        # Create config with both base and extended fields
        config = OllamaConfig(
            model_name="inherited-model",  # from ModelConfig
            temperature=0.3,               # from ModelConfig
            base_url="http://inherited:11434",  # from OllamaConfig
            num_ctx=2048                   # from OllamaConfig
        )
        
        # Should have both types of fields
        assert config["model_name"] == "inherited-model"
        assert config["temperature"] == 0.3
        assert config["base_url"] == "http://inherited:11434"
        assert config["num_ctx"] == 2048

    def test_config_type_annotations(self):
        """Test that configurations have proper type annotations."""
        # This is more of a static check, but we can verify the structure exists
        model_hints = get_type_hints(ModelConfig)
        ollama_hints = get_type_hints(OllamaConfig)
        
        # ModelConfig should have basic fields
        assert 'model_name' in model_hints
        assert 'temperature' in model_hints
        assert 'max_tokens' in model_hints
        
        # OllamaConfig should have additional fields
        assert 'base_url' in ollama_hints
        assert 'timeout' in ollama_hints
        assert 'num_ctx' in ollama_hints 