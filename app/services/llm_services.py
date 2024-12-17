from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import google.generativeai as genai
import json
from app.services.llm_base import LLMService
from app.core.config import get_settings, LLMProvider
from app.core.prompts import create_grammar_analysis_prompt
from app.models.grammar import GrammarAnalysis
import os
from dotenv import load_dotenv
from json_repair import repair_json
from pydantic import ValidationError
import asyncio
import pandas as pd

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
    def __init__(self, model: str = None):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.model = model or self.settings.OPENAI_CHAT_MODEL
        # Load grammar rules for mistake class mapping
        self.grammar_rules = pd.read_csv("data/Updated_Grammar_Mistakes_with_Fixed_Classifications.csv")
        self.title_to_class_map = dict(zip(self.grammar_rules['Mistake Title'], self.grammar_rules['Class']))

    async def generate_response(self, prompt: str) -> str:
        """
        Generate a response using OpenAI
        
        Args:
            prompt: Input prompt for the model
        """
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
            
            # Parse response
            json_response = json.loads(response.choices[0].message.content)
            
            # Add mistake classes based on titles
            if 'corrections' in json_response:
                for correction in json_response['corrections']:
                    mistake_title = correction.get('mistake_title')
                    correction['mistake_class'] = self.title_to_class_map.get(mistake_title, "Unknown")
            
            # Validate response structure
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
    def __init__(self, model: str = None):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model or self.settings.GEMINI_MODEL)
        # Load grammar rules for mistake class mapping
        self.grammar_rules = pd.read_csv("data/Updated_Grammar_Mistakes_with_Fixed_Classifications.csv")
        self.title_to_class_map = dict(zip(self.grammar_rules['Mistake Title'], self.grammar_rules['Class']))

    async def generate_response(self, prompt: str) -> str:
        """
        Generate a response using Gemini
        
        Args:
            prompt: Input prompt for the model
        """
        try:
            response = await asyncio.to_thread(
                lambda: self.model.generate_content(prompt).text
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def analyze_grammar(self, text: str, rules_context: str = "") -> Dict:
        """Analyze grammar using Gemini"""
        try:
            prompt = create_grammar_analysis_prompt(text, rules_context)
            print("prompt :\n", prompt)
            response = await asyncio.to_thread(
                lambda: self.model.generate_content(prompt, generation_config={"temperature": 0.7}).text
            )
            
            try:
                # Try to parse the JSON response directly
                json_response = json.loads(response)
                
                # Add mistake classes based on titles
                if 'corrections' in json_response:
                    for correction in json_response['corrections']:
                        mistake_title = correction.get('mistake_title')
                        correction['mistake_class'] = self.title_to_class_map.get(mistake_title, "Unknown")
                
                # Validate against Pydantic model
                analysis = GrammarAnalysis(**json_response)
                return analysis.model_dump()
            except (json.JSONDecodeError, ValidationError):
                # Try to repair and parse the JSON
                try:
                    repaired_json = repair_json(response)
                    json_response = json.loads(repaired_json)
                    
                    # Add mistake classes based on titles
                    if 'corrections' in json_response:
                        for correction in json_response['corrections']:
                            mistake_title = correction.get('mistake_title')
                            correction['mistake_class'] = self.title_to_class_map.get(mistake_title, "Unknown")
                    
                    # Validate against Pydantic model
                    analysis = GrammarAnalysis(**json_response)
                    return analysis.model_dump()
                except:
                    return {
                        "corrections": [],
                        "improved_version": text,
                        "overview": "Error analyzing grammar"
                    }
        except Exception as e:
            print(f"Grammar Analysis Error: {str(e)}")
            return {
                "corrections": [],
                "improved_version": text,
                "overview": "Error analyzing grammar"
            }

def get_llm_service(provider: Optional[str] = None, model: Optional[str] = None) -> LLMService:
    """
    Factory function to get the configured LLM service
    
    Args:
        provider: Optional provider override (defaults to settings)
        model: Optional model name override (defaults to settings)
    """
    provider = provider or settings.LLM_PROVIDER
    
    if provider == LLMProvider.OPENAI:
        return OpenAIService(model=model)
    elif provider == LLMProvider.GEMINI:
        return GeminiService(model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

async def test_both_services():
    """Test both OpenAI and Gemini services with the same input"""
    settings = get_settings()
    openai_service = OpenAIService()
    gemini_service = GeminiService()
    
    # Test text with intentional grammar mistakes
    test_text = """
    Hi everybody, welcome back to the channel. Today we are going to solve another five spark question and this question was asking like you can expect this question in almost every MNCs and this is the most ideal question for a fresher or a person who is having experience around one or two years of experience in data engineering. So this is generally to test your knowledge in basics of five spark in this part basically. 

So let's see the question. So the question is you need to return the details of employee whose location is not in Bangalore. So if you see the employee ID name location column, you have two employee with the locations Pune, Bangalore, Hyderabad and Mumbai, Bangalore, Pune. 

So you need to return the detail of employee whose location is not in Bangalore. So first you need to unpack this list of location and have it as an individual row and then you need to filter out that location should not be equal to Bangalore. So this is how we are going to solve this question. 

You see I have already imported a spark session and you can have your spark SQL functions import. Then I'm creating a spark session with a spark session dot builder dot get or create. First we will create a data frame.

So the data I'm taking is we are using with column to add location column and we are doing explode, which is like a splitting the locations into a different individual rows using a split splitting based on the column. Then we are displaying our final data frame in that we are selecting the columns that we want in this case employee ID name and location and we are doing the filter where the location is not equal to Bangalore. So this is our final output. 

If you like this video, please subscribe to the channel and follow for more content like this. Till then see you in the next one. Bye.
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
