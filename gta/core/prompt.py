"""
Pure prompt functions - Framework independent.
"""

from typing import List, Tuple, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage


def format_prompt_template(template_messages: List[Tuple[str, str]], variables: Dict[str, Any]) -> List[AnyMessage]:
    """
    Pure prompt template formatting function.
    
    Args:
        template_messages: List of (role, template) tuples
        variables: Variables for template substitution
        
    Returns:
        List of formatted messages
    """
    chat_template = ChatPromptTemplate(template_messages)
    prompt_value = chat_template.invoke(variables)
    return prompt_value.to_messages() 