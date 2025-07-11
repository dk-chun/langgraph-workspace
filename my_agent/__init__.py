"""
My Agent Package - Modular LangGraph Agent Services

This package provides independent agent services that can be deployed separately:
- Basic Agent: General purpose conversational service using Private State pattern

Each service is designed to work independently and can be deployed on LangGraph server.
"""

from my_agent.basic_agent import create_basic_agent, create_basic_coordinator, create_basic_service

# Service factories
__all__ = [
    "create_basic_agent",
    "create_basic_coordinator", 
    "create_basic_service"
]

# Version information
__version__ = "2.0.0"
__author__ = "LangGraph Agent Services" 