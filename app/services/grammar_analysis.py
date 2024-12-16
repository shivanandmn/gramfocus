import os
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
from .llm_services import get_llm_service

load_dotenv()

class GrammarAnalysisService:
    def __init__(self):
        self.grammar_rules = self._load_grammar_rules()
        self.llm_service = get_llm_service()

    def _load_grammar_rules(self) -> pd.DataFrame:
        """Load and process grammar rules from CSV file"""
        csv_path = "data/Updated_Grammar_Mistakes_with_Levels.csv"
        df = pd.read_csv(csv_path)
        # Filter for basic level rules
        return df[df['Grammar Level'] == 'Basic']

    def _create_rules_context(self) -> str:
        """Create a context string from the basic grammar rules"""
        rules = []
        for _, row in self.grammar_rules.iterrows():
            rules.append(f"- {row['Mistake Title']}: {row['Why it Happens']}")
        return "\n".join(rules)

    async def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for grammar mistakes using the configured LLM
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Analysis results containing mistakes and corrections
        """
        try:
            rules_context = self._create_rules_context()
            return await self.llm_service.analyze_grammar(text, rules_context)
        except Exception as e:
            print(f"Error analyzing grammar: {str(e)}")
            return {
                "error": "Failed to analyze grammar",
                "corrections": [],
                "improved_version": text,
                "explanation": "Error occurred during analysis"
            }
