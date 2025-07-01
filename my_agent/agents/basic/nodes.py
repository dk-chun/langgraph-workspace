"""
Node functions for the basic prompting agent.
"""

import os
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from my_agent.core.base_nodes import ProcessingNode
from my_agent.core.base_state import get_last_message_content, add_metadata, set_agent_type
from .state import BasicState

# Load environment variables
load_dotenv()


class BasicPromptNode(ProcessingNode):
    """
    Basic prompting node for simple prompt-response processing.
    """
    
    # Built-in prompt templates
    TEMPLATES = {
        "chat": "You are a helpful assistant.",
        "summarize": "You are an expert summarizer. Provide a clear and concise summary of the given text.",
        "translate": "You are a professional translator. Translate the given text accurately.",
        "explain": "You are an expert teacher. Explain the given concept in simple terms.",
        "code": "You are a coding expert. Help with programming questions and provide clean code.",
        "creative": "You are a creative writer. Help with creative writing tasks.",
        "analyze": "You are an analytical expert. Analyze the given information thoroughly.",
    }
    
    def __init__(self, 
                 model_type: str = "openai",
                 model_name: Optional[str] = None,
                 template: str = "chat",
                 system_message: Optional[str] = None,
                 temperature: float = 0.7):
        """
        Initialize basic prompt node.
        
        Args:
            model_type: Type of model ('openai' or 'ollama')
            model_name: Specific model name
            template: Template key from TEMPLATES
            system_message: Custom system message (overrides template)
            temperature: Model temperature
        """
        super().__init__("basic_prompt")
        
        self.model_type = model_type.lower()
        self.template = template
        self.custom_system_message = system_message
        self.temperature = temperature
        
        # Set default model names
        if model_name is None:
            if self.model_type == "openai":
                self.model_name = "gpt-4o-mini"
            else:  # ollama
                self.model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
        else:
            self.model_name = model_name
        
        # Initialize LLM
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        if self.model_type == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature
            )
        else:  # ollama
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return ChatOllama(
                base_url=base_url,
                model=self.model_name,
                temperature=self.temperature
            )
    
    def _get_system_message(self) -> str:
        """Get the system message to use."""
        if self.custom_system_message:
            return self.custom_system_message
        
        return self.TEMPLATES.get(self.template, self.TEMPLATES["chat"])
    
    def _format_prompt(self, user_message: str) -> str:
        """Format the user message with template if needed."""
        # For basic templates, we just use the message as-is
        # Templates are applied via system message
        return user_message
    
    def pre_process(self, state: BasicState) -> BasicState:
        """Set agent type and configuration before processing."""
        set_agent_type(state, "basic")
        add_metadata(state, "model_type", self.model_type)
        add_metadata(state, "model_name", self.model_name)
        add_metadata(state, "template", self.template)
        add_metadata(state, "temperature", self.temperature)
        return state
    
    def process(self, state: BasicState) -> Dict[str, Any]:
        """
        Process the user message with basic prompting.
        """
        # Get the user message
        user_message = get_last_message_content(state)
        
        if not user_message:
            return {
                "messages": [AIMessage(content="I didn't receive any message to process.")],
                "response": "No input received"
            }
        
        # Format the prompt
        formatted_message = self._format_prompt(user_message)
        
        # Create message list with system message
        messages = [
            SystemMessage(content=self._get_system_message()),
            HumanMessage(content=formatted_message)
        ]
        
        # Add conversation history if available
        if "messages" in state and len(state["messages"]) > 1:
            # Include previous conversation context
            conversation_messages = []
            for msg in state["messages"][:-1]:  # Exclude the last message (current)
                conversation_messages.append(msg)
            
            # Insert conversation history before the current message
            messages = [messages[0]] + conversation_messages + [messages[1]]
        
        # Generate response
        response = self.llm.invoke(messages)
        
        return {
            "messages": [AIMessage(content=response.content)],
            "response": response.content,
            "prompt_template": self.template,
            "system_message": self._get_system_message(),
            "model_type": self.model_type,
            "model_name": self.model_name
        }
    
    def post_process(self, state: BasicState, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add post-processing metadata."""
        result["processing_metadata"] = {
            "node_name": self.name,
            "template_used": self.template,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "system_message_length": len(self._get_system_message())
        }
        return result


# Create default node instances for different use cases
basic_chat_node = BasicPromptNode(template="chat")
basic_summarize_node = BasicPromptNode(template="summarize")
basic_translate_node = BasicPromptNode(template="translate")
basic_explain_node = BasicPromptNode(template="explain")
basic_code_node = BasicPromptNode(template="code")
basic_creative_node = BasicPromptNode(template="creative")
basic_analyze_node = BasicPromptNode(template="analyze") 