from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMService(ABC):
    """Base class for LLM services"""
    
    @abstractmethod
    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    async def analyze_grammar(self, text: str, rules_context: str = "") -> Dict[str, Any]:
        """
        Analyze grammar in the given text
        
        Args:
            text (str): Text to analyze
            rules_context (str): Optional context about grammar rules
            
        Returns:
            Dict[str, Any]: Analysis results in JSON format
        """
        pass
