"""
Models for critic scoring system.

Defines structured outputs for the SingleCriticAgent that evaluates answers
based on coverage, detail, and style rubric (not factual correctness).
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, model_validator


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""
    dimension_name: str = Field(description="Name of the scoring dimension")
    score: int = Field(ge=0, le=100, description="Score from 0-100 for this dimension")
    tier: str = Field(description="Quality tier: excellent/good/adequate/poor/inadequate")
    justification: str = Field(description="Brief explanation of the score for this dimension")
    weight: float = Field(ge=0, le=100, description="Weight percentage for this dimension")
    weighted_score: float = Field(default=0.0, description="Score * weight / 100")

    @validator('weighted_score', always=True)
    def calculate_weighted_score(cls, v, values):
        """Calculate weighted score automatically."""
        score = values.get('score', 0)
        weight = values.get('weight', 0)
        return round((score * weight / 100), 2)


class CriticScore(BaseModel):
    """
    Complete scoring output from SingleCriticAgent.
    
    Provides overall score and per-dimension breakdown with justifications.
    Emphasizes coverage, detail, and style over factual correctness.
    """
    # Overall scoring
    overall_score: int = Field(default=0, ge=0, le=100, description="Overall weighted score from 0-100")
    overall_tier: str = Field(default="inadequate", description="Overall quality tier based on total score")
    
    # Per-dimension scores
    dimension_scores: Dict[str, DimensionScore] = Field(
        description="Detailed scores for each evaluation dimension"
    )
    
    # Summary and reasoning
    overall_justification: str = Field(
        description="High-level summary of the scoring rationale"
    )
    key_strengths: List[str] = Field(
        default_factory=list,
        description="Top 2-3 areas where the answer excelled"
    )
    key_weaknesses: List[str] = Field(
        default_factory=list, 
        description="Top 2-3 areas needing improvement"
    )
    
    # Step-by-step thinking (for transparency)
    thinking_process: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning that led to the scores"
    )
    
    # Metadata
    rubric_version: str = Field(default="1.0", description="Version of scoring rubric used")
    evaluation_focus: str = Field(
        default="coverage_detail_style",
        description="Confirms this evaluation focused on presentation quality not factual accuracy"
    )

    @model_validator(mode='before')
    @classmethod
    def calculate_scores(cls, values):
        """Calculate overall score and tier from dimension scores."""
        if isinstance(values, dict):
            dimension_scores = values.get('dimension_scores', {})
            
            if dimension_scores:
                # Calculate overall score from weighted dimensions
                # Handle case where dimension_scores contains dicts (from JSON/dict input)
                total_weighted_score = 0
                for dim in dimension_scores.values():
                    if hasattr(dim, 'weighted_score'):
                        total_weighted_score += dim.weighted_score
                    elif isinstance(dim, dict):
                        # Calculate weighted score if it's a dict
                        score = dim.get('score', 0)
                        weight = dim.get('weight', 0)
                        total_weighted_score += (score * weight / 100)
                
                calculated_score = round(total_weighted_score)
                
                # Update overall score if not explicitly set or if set to default
                overall_score = values.get('overall_score', 0)
                if overall_score == 0:
                    values['overall_score'] = calculated_score
            
            # Calculate tier based on score
            score = values.get('overall_score', 0)
            if score >= 90:
                tier = "excellent"
            elif score >= 75:
                tier = "good"
            elif score >= 60:
                tier = "adequate"
            elif score >= 40:
                tier = "poor"
            else:
                tier = "inadequate"
            
            # Update tier if not explicitly set or if set to default
            overall_tier = values.get('overall_tier', '')
            if overall_tier in ('', 'inadequate') and score > 39:
                values['overall_tier'] = tier
            elif overall_tier == '':
                values['overall_tier'] = tier
                
        return values

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "overall_score": 82,
                "overall_tier": "good",
                "dimension_scores": {
                    "coverage": {
                        "dimension_name": "Information Coverage",
                        "score": 85,
                        "tier": "good", 
                        "justification": "Addresses most key aspects with minor gaps",
                        "weight": 30.0,
                        "weighted_score": 25.5
                    }
                },
                "overall_justification": "Strong coverage and detail with clear structure",
                "key_strengths": ["Comprehensive coverage", "Clear organization"],
                "key_weaknesses": ["Could use more specific examples"],
                "thinking_process": [
                    "First evaluated coverage - found most topics addressed",
                    "Then checked detail level - good specificity overall"
                ],
                "rubric_version": "1.0",
                "evaluation_focus": "coverage_detail_style"
            }
        }


class CriticRequest(BaseModel):
    """Request format for SingleCriticAgent evaluation."""
    question: str = Field(description="The original question or prompt")
    answer: str = Field(description="The answer to be evaluated")
    context: Optional[str] = Field(default=None, description="Additional context if needed")
    evaluation_instructions: Optional[str] = Field(
        default=None,
        description="Any specific evaluation instructions"
    )

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "question": "What are the key features of Python?",
                "answer": "Python has several important features including...",
                "context": "This is for a programming course evaluation",
                "evaluation_instructions": "Focus on technical accuracy and completeness"
            }
        }