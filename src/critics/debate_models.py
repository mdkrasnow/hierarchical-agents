"""
Models for multi-agent debate and critic orchestration system.

Defines data structures for coordinating multiple specialized critics,
aggregating their scores, and managing debate rounds.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, model_validator
from critics.models import CriticScore, DimensionScore


class CriticRole(BaseModel):
    """Definition of a specialized critic role."""
    role_name: str = Field(description="Name of the critic role (e.g., 'coverage', 'depth')")
    display_name: str = Field(description="Human-readable name for the role")
    description: str = Field(description="Description of what this critic evaluates")
    focus_areas: List[str] = Field(description="Key areas this critic focuses on")
    template_name: str = Field(description="Template file name for this critic's prompt")
    weight: float = Field(ge=0, le=1, description="Weight in final score aggregation (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role_name": "coverage",
                "display_name": "Coverage Critic",
                "description": "Evaluates completeness and breadth of information",
                "focus_areas": ["completeness", "breadth", "scope"],
                "template_name": "coverage_critic",
                "weight": 0.3
            }
        }


class CriticResult(BaseModel):
    """Result from a single specialized critic."""
    critic_role: str = Field(description="The role/name of the critic")
    critic_score: CriticScore = Field(description="Detailed scoring from this critic")
    execution_time_ms: float = Field(description="Time taken for this critic's evaluation")
    confidence: float = Field(ge=0, le=1, description="Critic's confidence in this evaluation")
    
    # Additional metadata specific to this critic's evaluation
    focus_summary: str = Field(description="Summary of what this critic focused on")
    notable_observations: List[str] = Field(
        default_factory=list,
        description="Specific observations unique to this critic's perspective"
    )


class DebateRound(BaseModel):
    """Represents one round of the multi-critic debate process."""
    round_number: int = Field(description="Round number (1-based)")
    round_type: str = Field(description="Type of round: 'independent' or 'aggregation'")
    participants: List[str] = Field(description="List of critic roles participating in this round")
    results: List[CriticResult] = Field(
        default_factory=list,
        description="Results from critics in this round"
    )
    round_summary: str = Field(
        default="",
        description="Summary of what happened in this round"
    )
    execution_time_ms: float = Field(default=0.0, description="Total time for this round")


class ScoreAggregation(BaseModel):
    """Aggregated scoring results from multiple critics."""
    final_score: int = Field(ge=0, le=100, description="Final aggregated score 0-100")
    final_tier: str = Field(description="Final quality tier")
    
    # Aggregation methodology
    aggregation_method: str = Field(description="Method used for aggregation")
    individual_scores: Dict[str, int] = Field(description="Individual scores by critic role")
    score_variance: float = Field(description="Variance across individual scores")
    consensus_level: str = Field(description="Level of consensus: high/medium/low")
    
    # Reasoning about aggregation
    aggregation_reasoning: str = Field(description="Explanation of how final score was determined")
    disagreement_analysis: List[str] = Field(
        default_factory=list,
        description="Analysis of disagreements between critics"
    )
    consensus_points: List[str] = Field(
        default_factory=list,
        description="Areas where critics agreed"
    )
    
    # Final synthesis
    comprehensive_strengths: List[str] = Field(description="Strengths identified across all critics")
    comprehensive_weaknesses: List[str] = Field(description="Weaknesses identified across all critics")
    actionable_recommendations: List[str] = Field(description="Specific improvement recommendations")


class MultiCriticResult(BaseModel):
    """Complete result from multi-critic evaluation process."""
    
    # Basic information
    request_id: str = Field(description="Unique identifier for this evaluation request")
    question: str = Field(description="Original question that was evaluated")
    answer: str = Field(description="Answer that was evaluated")
    context: Optional[str] = Field(description="Evaluation context if provided")
    
    # Process information
    debate_rounds: List[DebateRound] = Field(description="All rounds of the debate process")
    total_execution_time_ms: float = Field(description="Total time for entire evaluation")
    critics_used: List[str] = Field(description="List of critic roles that participated")
    
    # Final results
    final_aggregation: ScoreAggregation = Field(description="Final aggregated score and reasoning")
    
    # Summary
    evaluation_summary: str = Field(description="High-level summary of the evaluation")
    confidence_level: float = Field(ge=0, le=1, description="Overall confidence in the evaluation")
    
    # Metadata
    timestamp: str = Field(description="ISO timestamp of evaluation")
    system_version: str = Field(default="1.0", description="Version of the evaluation system")
    
    @property
    def final_score(self) -> int:
        """Get the final aggregated score."""
        return self.final_aggregation.final_score
    
    @property
    def final_tier(self) -> str:
        """Get the final quality tier."""
        return self.final_aggregation.final_tier
    
    @property
    def individual_critic_scores(self) -> Dict[str, int]:
        """Get scores from individual critics."""
        return self.final_aggregation.individual_scores
    
    def get_critic_result(self, critic_role: str) -> Optional[CriticResult]:
        """Get result from a specific critic role."""
        for round_data in self.debate_rounds:
            for result in round_data.results:
                if result.critic_role == critic_role:
                    return result
        return None
    
    def get_round_results(self, round_number: int) -> Optional[DebateRound]:
        """Get results from a specific round."""
        for round_data in self.debate_rounds:
            if round_data.round_number == round_number:
                return round_data
        return None


class MultiCriticRequest(BaseModel):
    """Request format for multi-critic evaluation."""
    question: str = Field(description="The original question or prompt")
    answer: str = Field(description="The answer to be evaluated")
    context: Optional[str] = Field(default=None, description="Additional context if needed")
    evaluation_instructions: Optional[str] = Field(
        default=None,
        description="Any specific evaluation instructions"
    )
    
    # Configuration options
    critic_roles: Optional[List[str]] = Field(
        default=None,
        description="Specific critic roles to use (default: all available)"
    )
    enable_debate: bool = Field(
        default=True,
        description="Whether to enable debate/aggregation round"
    )
    show_individual_results: bool = Field(
        default=False,
        description="Whether to include detailed individual critic results"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the key features of Python?",
                "answer": "Python has several important features including...",
                "context": "Programming course evaluation",
                "evaluation_instructions": "Focus on technical depth and practical examples",
                "critic_roles": ["coverage", "depth", "style", "instruction_following"],
                "enable_debate": True,
                "show_individual_results": True
            }
        }


class AggregatorInput(BaseModel):
    """Input to the aggregator agent that reconciles critic disagreements."""
    question: str = Field(description="Original question")
    answer: str = Field(description="Answer being evaluated")
    context: Optional[str] = Field(description="Evaluation context")
    
    # Results from individual critics
    critic_results: List[CriticResult] = Field(description="Results from all critics")
    
    # Analysis of the scores
    score_statistics: Dict[str, Union[int, float]] = Field(description="Statistical analysis of scores")
    
    # Instructions for aggregation
    aggregation_instructions: str = Field(
        default="",
        description="Specific instructions for how to aggregate results"
    )


class CriticConfiguration(BaseModel):
    """Configuration for the multi-critic system."""
    
    # Available critic roles
    available_critics: List[CriticRole] = Field(description="All available critic roles")
    
    # Default settings
    default_critics: List[str] = Field(description="Default critic roles to use")
    aggregation_method: str = Field(default="reasoned", description="Default aggregation method")
    enable_parallel_execution: bool = Field(default=True, description="Whether to run critics in parallel")
    
    # Thresholds and constraints
    min_critics: int = Field(default=2, ge=1, description="Minimum number of critics required")
    max_critics: int = Field(default=6, ge=1, description="Maximum number of critics allowed")
    score_variance_threshold: float = Field(
        default=15.0, 
        description="Variance threshold above which scores are considered highly discrepant"
    )
    
    # Timeout and retry settings
    critic_timeout_ms: float = Field(default=30000, description="Timeout for individual critics")
    aggregator_timeout_ms: float = Field(default=45000, description="Timeout for aggregator")
    max_retries: int = Field(default=2, description="Maximum retries for failed critics")
    
    # Weight validation removed - weights will be normalized during aggregation as needed
    
    def get_critic_role(self, role_name: str) -> Optional[CriticRole]:
        """Get critic role configuration by name."""
        for critic in self.available_critics:
            if critic.role_name == role_name:
                return critic
        return None
    
    def get_default_critic_roles(self) -> List[CriticRole]:
        """Get default critic role configurations."""
        return [critic for critic in self.available_critics if critic.role_name in self.default_critics]