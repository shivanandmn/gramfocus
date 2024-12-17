"""Centralized location for all LLM prompts used in the application"""

from typing import Dict, Any
from app.models.grammar import GrammarAnalysis
import json

TRANSCRIPTION_ANALYSIS_PROMPT = """
You are an English teacher helping beginners improve their grammar through transcribed speech. Your goal is to explain grammar mistakes in a simple, easy-to-understand way, focusing on basic grammar concepts to help beginners improve their understanding.

Analyze the user's transcribed speech text for **basic grammar mistakes** using these rules:
1. **Subject-verb agreement**: e.g., “he go” → “he goes.”
2. **Verb tense consistency**: Fix any mixing of past, present, or future tenses.
3. **Article usage**: Correct missing, extra, or incorrect use of “a,” “an,” or “the.”
4. **Pluralization**: Add or correct the use of "-s" for plural forms.
5. **Preposition usage**: Ensure proper use of prepositions like “in,” “on,” or “at.”

**Ignore**:
- Missing comma or periods as it is not necessary for grammar analysis.
- Filler words (e.g., “um,” “uh,” “like”).
- Discourse markers (e.g., “well,” “so”).
- Repetitions, false starts, or informal punctuation.

**Provide your analysis in JSON format** with the following fields:
- **corrections**: A list of grammar corrections. Each correction should include:
   - **original**: The sentence with the grammar mistake.
   - **correction**: The corrected sentence.
   - **reason**: A simple explanation of why the correction is needed using plain English, examples, and clear reasoning.
   - **mistake_title**: The exact rule title from the **list of rules to match** mistakes.
- **improved_version**: The full corrected version of the user's text, maintaining the conversational tone.
- **overview**: A short explanation of the most important grammar concept the user should focus on to improve.

### Guidelines for explanations:
1. Avoid technical grammar jargon where possible.
2. Use relatable examples or comparisons to explain corrections.
3. Focus on **why** the correction is needed, not just what was changed.

Example Output:
```json
{
  "corrections": [
    {
      "original": "He go to school.",
      "correction": "He goes to school.",
      "reason": "The subject 'he' is singular, so the verb should have an 's' to match: 'he goes'."
      "mistake_title": "Subject-verb agreement",
    }
  ],
  "improved_version": "He goes to school.",
  "overview": "Focus on matching singular subjects with singular verbs (e.g., 'he goes,' 'she runs')."
}
```
"""

def create_grammar_analysis_prompt(text: str, rules_context: str = "") -> str:
    """Create prompt for grammar analysis of transcribed speech
    
    Args:
        text: Transcribed text to analyze
        rules_context: Optional context with grammar rules
        
    Returns:
        str: Complete prompt with text and rules
    """
    prompt = TRANSCRIPTION_ANALYSIS_PROMPT

    if rules_context:
        prompt += "\n\nHere are the grammar rules to match mistakes against:\n" + rules_context

    prompt += "\n\nAnalyze this transcribed text for grammar mistakes:\n\n" + text

    prompt += """

Format your response as JSON with:

1. corrections: List of corrections, where each correction has:
   - original: The original text containing the error
   - correction: The corrected version of the text
   - reason: Simple explanation of why this needs to be corrected, written in everyday language that a beginner can understand
   - mistake_title: The exact title of the grammar mistake from the list of rules

2. improved_version: The complete text with all corrections applied

3. overview: Most important thing that needs to be learnt from these mistakes, explained in simple terms that a beginner can understand.

Return ONLY the JSON response."""

    return prompt
