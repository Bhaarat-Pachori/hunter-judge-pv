"""
schemas.py
"""

from typing import TypedDict, Optional, Dict
from pydantic import BaseModel, Field


class ExtractedEntities(BaseModel):
    """Pydantic model for the entities extracted from the clinical text."""
    drug: Optional[str] = Field(description="The suspected drug causing the adverse event.")
    reaction: Optional[str] = Field(description="The adverse reaction or event observed.")


class CausalityAssessment(BaseModel):
    """Pydantic model for the final structured output, including Naranjo causality scoring."""
    drug: str = Field(description="The suspected drug.")
    reaction: str = Field(description="The adverse reaction.")
    causality: str = Field(
        description="The assessed causality category based on Naranjo score (e.g., 'Definite', 'Probable', 'Possible', 'Doubtful')."
    )
    naranjo_score: int = Field(description="The total calculated Naranjo score (from -4 to 13).")
    naranjo_answers: Dict[str, int] = Field(description="A dictionary of the scores for each of the 10 Naranjo questions.")
    reasoning_chain: str = Field(description="A step-by-step explanation of how each Naranjo question was scored based on the provided evidence.")



# This is the state that will be passed between nodes in our LangGraph agent.
class AgentState(TypedDict):
    """
    Represents the state of the agent's workflow.

    Attributes:
        text: The original input clinical text.
        entities: The entities extracted by the NLP tool.
        assessment: The final causality assessment.
        retries: A counter to prevent infinite loops if entity extraction fails.
    """
    text: str

    # normalized text from slang
    entities: Optional[ExtractedEntities]
    normalized_text: Optional[str]
    
    # NEW: Track what we are actually searching for
    current_drug_query: Optional[str] 
    current_reaction_query: Optional[str]
    
    verification_evidence: Optional[str] 
    assessment: Optional[CausalityAssessment]
    
    retries: int # We will repurpose this for the Reflection Loop count
    error: Optional[str]
