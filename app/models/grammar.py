from pydantic import BaseModel, Field
from typing import List
from app.core.config import LLMProvider


class GrammarCorrection(BaseModel):
    """Model for individual grammar corrections"""

    original: str = Field(..., description="The original text with grammar error")
    correction: str = Field(..., description="The corrected version of the text")
    reason: str = Field(..., description="Reason why the correction is needed by comparing original and corrected text")
    mistake_title: str = Field(..., description="The exact title of the grammar mistake from the list of rules")


class GrammarAnalysis(BaseModel):
    """Model for complete grammar analysis response"""

    corrections: List[GrammarCorrection] = Field(
        ..., description="List of grammar corrections"
    )
    improved_version: str = Field(
        ..., description="Complete text with all corrections applied"
    )
    overview: str = Field(
        ...,
        description="What is the most important that needs to be learnt from these mistakes, explain",
    )


class TranscriptAnalysisRequest(BaseModel):
    """Model for transcript analysis request"""
    
    transcript: str = Field(..., description="The transcript text to analyze")
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use for grammar analysis (openai/gemini)"
    )
    model_name: str | None = Field(
        None, 
        description="Specific model name to use (e.g., 'gpt-4-mini' for OpenAI or 'gemini-pro' for Gemini)"
    )
