"""Centralized location for all LLM prompts used in the application"""

from typing import Dict, Any
from app.models.grammar import GrammarAnalysis
import json

TRANSCRIPTION_ANALYSIS_PROMPT = """You are a grammar analysis expert focusing on basic English grammar mistakes in transcribed speech.
Grammar analysis is shown to users as a way to improve their English grammar, so it is important to provide clear and concise corrections.
Currently the user is beginner in the language, so focus on basic grammar mistakes.

Important guidelines for analyzing transcribed speech:
1. IGNORE the following elements as they are normal in spoken language:
   - Filler words (um, uh, like, you know)
   - Discourse markers (well, so, actually, basically)
   - Repetitions and false starts
   - Missing or informal punctuation
   - Capitalization of proper nouns or any other words

2. Focus ONLY on these key aspects:
   - Subject-verb agreement
   - Verb tense consistency
   - Basic sentence structure
   - Word order

3. For each grammar error you identify:
   - Match it with the most appropriate mistake title from the provided rules
   - Use EXACTLY the same mistake title as shown in the rules
   - If multiple rules could apply, choose the most specific one
   - Never make up new titles - use only those provided

4. If there are multiple grammar errors in the text, separate them as different corrections.

5. Format your response as JSON with these fields:
   - corrections: list of specific grammar corrections with mistake titles
   - improved_version: corrected version of the text (maintain spoken language style)
   - explanation: brief explanation of the main grammar issues

Remember this is transcribed speech, so focus on substantial grammar errors, not speaking style or punctuation."""

def create_grammar_analysis_prompt(text: str, rules_context: str = "") -> str:
    """Create a prompt for grammar analysis with explicit output format"""
    
    # Get the JSON schema from the Pydantic model
    output_schema = GrammarAnalysis.model_json_schema()
    
    base_prompt = TRANSCRIPTION_ANALYSIS_PROMPT + f"""\nAnalyze the following text for grammar errors and provide corrections.

Text to analyze:
{text}

{f'Additional rules context:\n{rules_context}\n' if rules_context else ''}

Your response must follow this exact schema:

1. corrections: List of corrections, where each correction has:
   - original: The original text containing the error
   - correction: The corrected version of the text
   - explanation: Explanation of the grammar issue
   - mistake_title: The exact title of the grammar mistake from the list of rules

2. improved_version: The complete text with all corrections applied

3. overview: Most important thing that needs to be learnt from these mistakes

Schema Definition:
{json.dumps(output_schema, indent=2)}

Instructions:
1. Analyze the text for grammar errors
2. List each error with its correction
3. Provide the complete corrected text
4. Give a concise overview of the analysis
5. Ensure your response is valid JSON and exactly matches the schema

Return your response as a valid JSON object matching the schema above."""

    return base_prompt
