"""
Agent Services Package

Contains independent agent services that can be deployed separately:
- basic: General purpose conversational service using Private State pattern

Each service is self-contained and can be deployed independently on LangGraph server.
"""

from .basic import create_basic_agent

__all__ = [
    "create_basic_agent"
] 