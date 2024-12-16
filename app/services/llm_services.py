from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import google.generativeai as genai
import json
from app.services.llm_base import LLMService
from app.core.config import get_settings
from app.core.prompts import create_grammar_analysis_prompt
from app.models.grammar import GrammarAnalysis
import os
from dotenv import load_dotenv
from json_repair import repair_json
from pydantic import ValidationError

# Load environment variables from .env file
load_dotenv()

settings = get_settings()

# Models that support JSON response format
JSON_SUPPORTED_MODELS = {
    'gpt-4-1106-preview',
    'gpt-3.5-turbo-1106',
    'gpt-4o-mini'
}

class OpenAIService(LLMService):
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.model = self.settings.OPENAI_CHAT_MODEL

    async def generate_response(self, prompt: str) -> str:
        """Generate a response using OpenAI"""
        try:
            # Base parameters
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            
            # Add response_format if supported
            if self.model in JSON_SUPPORTED_MODELS:
                params["response_format"] = {"type": "json_object"}
            
            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def analyze_grammar(self, text: str, rules_context: str = "") -> Dict[str, Any]:
        """Analyze grammar using OpenAI"""
        try:
            prompt = create_grammar_analysis_prompt(text, rules_context)
            
            # Base parameters
            params = {
                "model": self.model,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": 0.7,
            }
            
            # Add response_format if supported
            if self.model in JSON_SUPPORTED_MODELS:
                params["response_format"] = {"type": "json_object"}
            
            response = await self.client.chat.completions.create(**params)
            
            # Parse and validate response
            json_response = json.loads(response.choices[0].message.content)
            validated_response = GrammarAnalysis(**json_response)
            return validated_response.model_dump()
            
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Response validation failed: {str(e)}")
            return {
                "error": str(e),
                "corrections": [],
                "improved_version": text,
                "overview": "Error analyzing grammar"
            }
        except Exception as e:
            print(f"OpenAI Analysis Error: {str(e)}")
            return {
                "error": str(e),
                "corrections": [],
                "improved_version": text,
                "overview": "Error analyzing grammar"
            }

class GeminiService(LLMService):
    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(self.settings.GEMINI_MODEL)

    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Gemini"""
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def analyze_grammar(self, text: str, rules_context: str = "") -> Dict[str, Any]:
        """Analyze grammar using Gemini"""
        try:
            prompt = create_grammar_analysis_prompt(text, rules_context)
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7
                )
            )
            
            try:
                # Use json_repair to handle malformed JSON
                repaired_json = repair_json(response.text)
                json_response = json.loads(repaired_json)
                
                # Validate against Pydantic model
                validated_response = GrammarAnalysis(**json_response)
                return validated_response.model_dump()
                
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Response validation failed: {str(e)}")
                return {
                    "error": str(e),
                    "corrections": [],
                    "improved_version": text,
                    "overview": "Error analyzing grammar due to response format"
                }
                
        except Exception as e:
            print(f"Gemini Analysis Error: {str(e)}")
            return {
                "error": str(e),
                "corrections": [],
                "improved_version": text,
                "overview": "Error analyzing grammar"
            }

def get_llm_service() -> LLMService:
    """Factory function to get the configured LLM service"""
    if settings.LLM_PROVIDER == "openai":
        return OpenAIService()
    elif settings.LLM_PROVIDER == "gemini":
        return GeminiService()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

import asyncio

async def test_both_services():
    """Test both OpenAI and Gemini services with the same input"""
    # Initialize both services
    openai_service = OpenAIService()
    gemini_service = GeminiService()
    
    # Test text with intentional grammar mistakes
    test_text = """
    1. She dont like ice cream
    2. The cats is sleeping
    3. He have went to the store
    """
    
    print("\n=== Testing Grammar Analysis ===")
    print("\nInput text:")
    print(test_text)
    
    # Test OpenAI
    print("\n🔵 OpenAI Analysis:")
    try:
        openai_result = await openai_service.analyze_grammar(test_text)
        print(json.dumps(openai_result, indent=2))
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
    
    # Test Gemini
    print("\n🟢 Gemini Analysis:")
    try:
        gemini_result = await gemini_service.analyze_grammar(test_text)
        print(json.dumps(gemini_result, indent=2))
    except Exception as e:
        print(f"Gemini Error: {str(e)}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_both_services())
