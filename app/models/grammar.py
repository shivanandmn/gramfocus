from pydantic import BaseModel, Field
from typing import List


class GrammarCorrection(BaseModel):
    """Model for individual grammar corrections"""

    original: str = Field(..., description="The original text with grammar error")
    correction: str = Field(..., description="The corrected version of the text")
    explanation: str = Field(..., description="Explanation of the grammar issue")
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
