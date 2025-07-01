"""
Tools for the OpenAI agent.
"""

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query
        
    Returns:
        Search results (placeholder implementation)
    """
    return f"Search results for '{query}': This is a placeholder implementation."


# List of available tools for OpenAI agent
OPENAI_TOOLS = [calculator, search_web] 