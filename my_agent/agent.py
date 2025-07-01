"""
Main OpenAI agent implementation using LangGraph.
"""

from my_agent.agents.openai.agent import create_openai_agent

# Create the OpenAI agent instance with default configuration
agent = create_openai_agent() 