import os
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
from app.services.llm_services import get_llm_service

load_dotenv()

class GrammarAnalysisService:
    def __init__(self):
        self.grammar_rules = self._load_grammar_rules()
        self.title_to_class_map = self._create_title_class_map()
        self.llm_service = get_llm_service()

    def _load_grammar_rules(self) -> pd.DataFrame:
        """Load and process grammar rules from CSV file"""
        csv_path = "data/Updated_Grammar_Mistakes_with_Fixed_Classifications.csv"
        df = pd.read_csv(csv_path)
        # Filter for basic level rules
        return df[df['Grammar Level'] == 'Basic']

    def _create_title_class_map(self) -> Dict[str, str]:
        """Create a mapping of mistake titles to their corresponding classes
        
        Returns:
            Dict[str, str]: Dictionary mapping mistake titles to their classes
        """
        return dict(zip(self.grammar_rules['Mistake Title'], self.grammar_rules['Class']))

    def _create_rules_context(self) -> str:
        """Create a context string from the basic grammar rules
        
        Returns:
            str: A formatted string containing grammar rules with their titles
        """
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
            analysis = await self.llm_service.analyze_grammar(text, rules_context)
            
            # Add mistake classes based on titles
            if 'corrections' in analysis:
                for correction in analysis['corrections']:
                    mistake_title = correction.get('mistake_title')
                    if mistake_title in self.title_to_class_map:
                        correction['mistake_class'] = self.title_to_class_map[mistake_title]
                    else:
                        correction['mistake_class'] = "Unknown"  # Fallback if title not found
            
            return analysis
        except Exception as e:
            print(f"Error analyzing grammar: {str(e)}")
            return {
                "error": "Failed to analyze grammar",
                "corrections": [],
                "improved_version": text,
                "explanation": "Error occurred during analysis"
            }

async def test_grammar_analysis():
    """Test the GrammarAnalysisService with real-world text"""
    try:
        # Initialize service
        grammar_service = GrammarAnalysisService()
        
        # Real-world text sample with natural grammar mistakes
        test_text = """
Hi everybody, welcome back to the channel. Today we are going to solve another five spark question and this question was asking like you can expect this question in almost every MNCs and this is the most ideal question for a fresher or a person who is having experience around one or two years of experience in data engineering. So this is generally to test your knowledge in basics of five spark in this part basically. 

So let's see the question. So the question is you need to return the details of employee whose location is not in Bangalore. So if you see the employee ID name location column, you have two employee with the locations Pune, Bangalore, Hyderabad and Mumbai, Bangalore, Pune. 

So you need to return the detail of employee whose location is not in Bangalore. So first you need to unpack this list of location and have it as an individual row and then you need to filter out that location should not be equal to Bangalore. So this is how we are going to solve this question. 

You see I have already imported a spark session and you can have your spark SQL functions import. Then I'm creating a spark session with a spark session dot builder dot get or create. First we will create a data frame.

So the data I'm taking is we are using with column to add location column and we are doing explode, which is like a splitting the locations into a different individual rows using a split splitting based on the column. Then we are displaying our final data frame in that we are selecting the columns that we want in this case employee ID name and location and we are doing"""
        
        print("\n=== Starting Grammar Analysis Test ===\n")
        print("Input text sample:")
        print(test_text[:200] + "...\n")  # Show first 200 chars for preview
        
        # Analyze text
        result = await grammar_service.analyze_text(test_text)
        
        # Print results
        print("\nAnalysis Results:")
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
            
        print("\nGrammar Corrections:")
        for i, correction in enumerate(result['corrections'], 1):
            print(f"\n{i}. Correction:")
            print(f"   Original: {correction['original']}")
            print(f"   Corrected: {correction['correction']}")
            print(f"   Explanation: {correction['explanation']}")
            print(f"   Mistake Title: {correction['mistake_title']}")
            print(f"   Mistake Class: {correction['mistake_class']}")
        
        print("\nImproved Version Preview:")
        print(result['improved_version'][:200] + "...")  # Show first 200 chars
        
        print("\nOverview of Issues:")
        print(result['overview'])
        
        print("\n=== Grammar Analysis Test Completed ===")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    # Run the test
    import asyncio
    asyncio.run(test_grammar_analysis())
