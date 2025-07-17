"""
My Agent Package - Modular LangGraph Agent Services

This package provides independent agent services that can be deployed separately:
- Basic Agent: General purpose conversational service using runtime configuration

Each service is designed to work independently and can be deployed on LangGraph server.
"""

from gta.agents.basic_agent import create_basic_agent

# Service factories
__all__ = [
    "create_basic_agent"
]

# Version information
__version__ = "2.0.0"
__author__ = "LangGraph Agent Services" 