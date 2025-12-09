"""
Multi-Critic Orchestrator - Coordinates multiple specialized critics and aggregates results.

Implements the main orchestration logic for running multiple critics in parallel,
analyzing disagreements, and synthesizing final evaluations through debate rounds.
"""

import asyncio
import logging
import statistics
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set

from agents.base import BaseAgent, AgentResult
from agents.templates import TemplateManager, FileTemplateLoader
from critics.models import CriticRequest
from critics.debate_models import (
    MultiCriticRequest, MultiCriticResult, DebateRound, CriticResult, 
    ScoreAggregation, CriticConfiguration, AggregatorInput
)
from critics.multi_critic import CriticFactory, SpecializedCriticAgent
from utils.llm import LLMClient

logger = logging.getLogger(__name__)


class ScoreAggregatorAgent(BaseAgent):
    """
    Agent responsible for aggregating and reconciling results from multiple critics.
    
    Uses reasoning rather than simple averaging to resolve disagreements and
    synthesize final evaluations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the aggregator agent."""
        if 'template_manager' not in kwargs:
            template_manager = TemplateManager()
            file_loader = FileTemplateLoader("configs/prompts")
            template_manager.add_loader("file", file_loader)
            kwargs['template_manager'] = template_manager
        
        super().__init__(*args, **kwargs)
    
    @property
    def agent_type(self) -> str:
        return "ScoreAggregator"
    
    @property
    def role_description(self) -> str:
        return "Synthesizes and reconciles multiple critic evaluations into coherent final assessment"
    
    async def execute(self, aggregator_input: AggregatorInput, **kwargs) -> AgentResult:
        """
        Aggregate and synthesize results from multiple critics.
        
        Args:
            aggregator_input: Input containing critic results and metadata
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with ScoreAggregation data
        """
        try:
            self.logger.info("Starting score aggregation and synthesis")
            
            # Prepare data for aggregation
            critic_results_text = self._format_critic_results(aggregator_input.critic_results)
            score_stats_text = self._format_score_statistics(aggregator_input.score_statistics)
            
            # Set execution context
            self.set_context("aggregation_task", "multi_critic_synthesis")
            self.set_context("num_critics", len(aggregator_input.critic_results))
            self.set_context("score_variance", aggregator_input.score_statistics.get("variance", 0))
            
            # Prepare template variables
            template_vars = {
                "context": aggregator_input.context or "General evaluation",
                "question": aggregator_input.question,
                "answer": aggregator_input.answer,
                "critic_results": critic_results_text,
                "score_statistics": score_stats_text,
                "aggregation_instructions": aggregator_input.aggregation_instructions
            }
            
            # Call LLM for aggregation
            self.logger.info("Calling LLM for score aggregation")
            result = await self.llm_call_with_template(
                template_name="aggregator",
                template_variables=template_vars,
                response_format=ScoreAggregation,
                temperature=0.4,  # Slightly higher for reasoning flexibility
                loader_name="file"
            )
            
            # Process and validate result
            if isinstance(result, ScoreAggregation):
                aggregation = result
                self.logger.info(f"Aggregation completed with final score: {aggregation.final_score}")
            else:
                # Fallback aggregation if LLM returns unstructured response
                self.logger.warning("LLM returned unstructured response, using fallback aggregation")
                aggregation = self._fallback_aggregation(aggregator_input)
            
            # Additional validation
            self._validate_aggregation(aggregation, aggregator_input)
            
            return AgentResult(
                success=True,
                data={"score_aggregation": aggregation.dict()},
                agent_id=self.agent_id,
                execution_time_ms=0.0,
                metadata={
                    "num_critics": len(aggregator_input.critic_results),
                    "score_variance": aggregation.score_variance,
                    "consensus_level": aggregation.consensus_level,
                    "final_score": aggregation.final_score,
                    "aggregation_method": aggregation.aggregation_method
                }
            )
            
        except Exception as e:
            self.logger.error(f"Score aggregation failed: {e}")
            return AgentResult(
                success=False,
                error=f"Score aggregation failed: {str(e)}",
                agent_id=self.agent_id,
                execution_time_ms=0.0,
                metadata={
                    "num_critics": len(aggregator_input.critic_results) if aggregator_input.critic_results else 0
                }
            )
    
    def _format_critic_results(self, critic_results: List[CriticResult]) -> str:
        """Format critic results for template input."""
        formatted_lines = []
        
        for result in critic_results:
            score = result.critic_score
            lines = [
                f"\n--- {result.critic_role.upper()} CRITIC EVALUATION ---",
                f"Score: {score.overall_score}/100 ({score.overall_tier})",
                f"Focus: {result.focus_summary}",
                f"Confidence: {result.confidence:.2f}",
                f"\nJustification: {score.overall_justification}",
                f"\nKey Strengths: {'; '.join(score.key_strengths)}",
                f"Key Weaknesses: {'; '.join(score.key_weaknesses)}",
            ]
            
            if result.notable_observations:
                lines.append(f"Notable Observations: {'; '.join(result.notable_observations)}")
            
            # Include dimension breakdown
            lines.append("\nDimension Scores:")
            for dim_name, dim_score in score.dimension_scores.items():
                lines.append(f"  - {dim_score.dimension_name}: {dim_score.score}/100 ({dim_score.tier})")
            
            formatted_lines.extend(lines)
        
        return "\n".join(formatted_lines)
    
    def _format_score_statistics(self, stats: Dict) -> str:
        """Format score statistics for template input."""
        return f"""
Score Statistics:
- Mean Score: {stats.get('mean', 0):.1f}
- Median Score: {stats.get('median', 0):.1f}
- Standard Deviation: {stats.get('std_dev', 0):.1f}
- Variance: {stats.get('variance', 0):.1f}
- Score Range: {stats.get('min_score', 0)} - {stats.get('max_score', 0)}
- Number of Critics: {stats.get('count', 0)}
        """.strip()
    
    def _fallback_aggregation(self, aggregator_input: AggregatorInput) -> ScoreAggregation:
        """Create fallback aggregation when LLM fails."""
        self.logger.warning("Using fallback aggregation logic")
        
        scores = [result.critic_score.overall_score for result in aggregator_input.critic_results]
        individual_scores = {result.critic_role: result.critic_score.overall_score 
                           for result in aggregator_input.critic_results}
        
        # Simple weighted average as fallback
        final_score = int(round(statistics.mean(scores)))
        variance = statistics.variance(scores) if len(scores) > 1 else 0
        
        # Determine tier
        if final_score >= 90:
            tier = "excellent"
        elif final_score >= 75:
            tier = "good"
        elif final_score >= 60:
            tier = "adequate"
        elif final_score >= 40:
            tier = "poor"
        else:
            tier = "inadequate"
        
        # Determine consensus level
        if variance < 5:
            consensus = "high"
        elif variance < 15:
            consensus = "medium"
        else:
            consensus = "low"
        
        return ScoreAggregation(
            final_score=final_score,
            final_tier=tier,
            aggregation_method="fallback_averaging",
            individual_scores=individual_scores,
            score_variance=variance,
            consensus_level=consensus,
            aggregation_reasoning="Fallback aggregation used due to synthesis failure",
            disagreement_analysis=["Unable to analyze disagreements due to synthesis failure"],
            consensus_points=["Fallback averaging applied"],
            comprehensive_strengths=["Multiple critic perspectives considered"],
            comprehensive_weaknesses=["Detailed synthesis unavailable"],
            actionable_recommendations=["Review individual critic feedback for specific insights"]
        )
    
    def _validate_aggregation(self, aggregation: ScoreAggregation, input_data: AggregatorInput):
        """Validate the aggregation result."""
        # Check score range
        if not (0 <= aggregation.final_score <= 100):
            raise ValueError(f"Final score {aggregation.final_score} outside valid range 0-100")
        
        # Check that we have scores for all critics
        expected_critics = {result.critic_role for result in input_data.critic_results}
        actual_critics = set(aggregation.individual_scores.keys())
        
        if expected_critics != actual_critics:
            missing = expected_critics - actual_critics
            extra = actual_critics - expected_critics
            self.logger.warning(f"Critic score mismatch - Missing: {missing}, Extra: {extra}")


class MultiCriticOrchestrator:
    """
    Main orchestrator for multi-agent evaluation system.
    
    Coordinates specialized critics, manages debate rounds, and synthesizes final results.
    """
    
    def __init__(self, llm_client: LLMClient, configuration: Optional[CriticConfiguration] = None):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: LLM client for all agents
            configuration: System configuration (uses default if not provided)
        """
        self.llm_client = llm_client
        self.configuration = configuration or CriticFactory.get_default_configuration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize aggregator agent
        self.aggregator = ScoreAggregatorAgent(llm_client=llm_client)
    
    async def evaluate(self, request: MultiCriticRequest) -> MultiCriticResult:
        """
        Perform multi-critic evaluation.
        
        Args:
            request: Evaluation request with question, answer, and options
            
        Returns:
            MultiCriticResult with complete evaluation breakdown
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting multi-critic evaluation {request_id}")
        
        try:
            # Determine which critics to use
            critic_roles = self._determine_critic_roles(request)
            
            # Round 1: Independent parallel critic evaluation
            round1_result = await self._run_independent_critics(request, critic_roles, request_id)
            
            # Round 2: Aggregation and synthesis (if enabled)
            if request.enable_debate and len(round1_result.results) >= 2:
                round2_result, aggregation_result = await self._run_aggregation_round(request, round1_result, request_id)
                debate_rounds = [round1_result, round2_result]
                final_aggregation = aggregation_result
            else:
                # Skip aggregation, use simple combination
                debate_rounds = [round1_result]
                final_aggregation = await self._simple_aggregation(round1_result.results)
            
            # Create final result
            total_time = (time.time() - start_time) * 1000
            
            result = MultiCriticResult(
                request_id=request_id,
                question=request.question,
                answer=request.answer,
                context=request.context,
                debate_rounds=debate_rounds,
                total_execution_time_ms=total_time,
                critics_used=critic_roles,
                final_aggregation=final_aggregation,
                evaluation_summary=self._generate_summary(final_aggregation, debate_rounds),
                confidence_level=self._calculate_confidence(debate_rounds),
                timestamp=datetime.now().isoformat(),
                system_version="1.0"
            )
            
            self.logger.info(f"Multi-critic evaluation {request_id} completed: {final_aggregation.final_score}/100")
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-critic evaluation {request_id} failed: {e}")
            raise RuntimeError(f"Multi-critic evaluation failed: {str(e)}")
    
    def _determine_critic_roles(self, request: MultiCriticRequest) -> List[str]:
        """Determine which critic roles to use based on request."""
        if request.critic_roles:
            # Validate requested roles
            available_roles = {critic.role_name for critic in self.configuration.available_critics}
            invalid_roles = set(request.critic_roles) - available_roles
            if invalid_roles:
                raise ValueError(f"Invalid critic roles requested: {invalid_roles}")
            return request.critic_roles
        else:
            # Use default critics
            return self.configuration.default_critics
    
    async def _run_independent_critics(self, request: MultiCriticRequest, 
                                     critic_roles: List[str], request_id: str) -> DebateRound:
        """Run independent critic evaluations in parallel."""
        self.logger.info(f"Round 1: Running {len(critic_roles)} critics in parallel")
        
        round_start = time.time()
        
        # Create critic instances
        critics = {}
        for role_name in critic_roles:
            critics[role_name] = CriticFactory.create_critic(role_name, llm_client=self.llm_client)
        
        # Create critic request
        critic_request = CriticRequest(
            question=request.question,
            answer=request.answer,
            context=request.context,
            evaluation_instructions=request.evaluation_instructions
        )
        
        # Run critics in parallel
        tasks = []
        for role_name, critic in critics.items():
            task = asyncio.create_task(
                self._run_single_critic(critic, critic_request, role_name),
                name=f"critic_{role_name}"
            )
            tasks.append((role_name, task))
        
        # Wait for all critics to complete
        results = []
        for role_name, task in tasks:
            try:
                critic_result = await task
                results.append(critic_result)
                self.logger.info(f"Critic {role_name} completed: {critic_result.critic_score.overall_score}/100")
            except Exception as e:
                self.logger.error(f"Critic {role_name} failed: {e}")
                # Continue with other critics
        
        round_time = (time.time() - round_start) * 1000
        
        return DebateRound(
            round_number=1,
            round_type="independent",
            participants=list(critics.keys()),
            results=results,
            round_summary=f"Independent evaluation by {len(results)} specialized critics",
            execution_time_ms=round_time
        )
    
    async def _run_single_critic(self, critic: SpecializedCriticAgent, 
                                request: CriticRequest, role_name: str) -> CriticResult:
        """Run a single critic evaluation and package the result."""
        start_time = time.time()
        
        # Execute critic
        agent_result = await critic.execute(request)
        
        execution_time = (time.time() - start_time) * 1000
        
        if not agent_result.success:
            raise RuntimeError(f"Critic execution failed: {agent_result.error}")
        
        # Package result
        critic_score = agent_result.data["critic_score"]
        
        return CriticResult(
            critic_role=role_name,
            critic_score=critic_score,
            execution_time_ms=execution_time,
            confidence=0.8,  # Default confidence
            focus_summary=f"Focused on {', '.join(critic.critic_role.focus_areas)}",
            notable_observations=[]  # Could be enhanced to extract key observations
        )
    
    async def _run_aggregation_round(self, request: MultiCriticRequest,
                                   round1: DebateRound, request_id: str) -> DebateRound:
        """Run aggregation round to synthesize critic results."""
        self.logger.info("Round 2: Running aggregation and synthesis")
        
        round_start = time.time()
        
        # Calculate score statistics
        scores = [result.critic_score.overall_score for result in round1.results]
        score_stats = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "variance": statistics.variance(scores) if len(scores) > 1 else 0,
            "min_score": min(scores),
            "max_score": max(scores),
            "count": len(scores)
        }
        
        # Create aggregator input
        aggregator_input = AggregatorInput(
            question=request.question,
            answer=request.answer,
            context=request.context,
            critic_results=round1.results,
            score_statistics=score_stats,
            aggregation_instructions="Synthesize critic perspectives into balanced final evaluation"
        )
        
        # Run aggregation
        aggregator_result = await self.aggregator.execute(aggregator_input)
        
        round_time = (time.time() - round_start) * 1000
        
        if not aggregator_result.success:
            raise RuntimeError(f"Aggregation failed: {aggregator_result.error}")
        
        # Create synthetic CriticResult for aggregator output
        aggregation = aggregator_result.data["score_aggregation"]
        
        # Convert ScoreAggregation to CriticScore format for consistency
        synthetic_critic_score = {
            "overall_score": aggregation["final_score"],
            "overall_tier": aggregation["final_tier"],
            "dimension_scores": {},  # Empty since aggregator works at higher level
            "overall_justification": aggregation["aggregation_reasoning"],
            "key_strengths": aggregation["comprehensive_strengths"],
            "key_weaknesses": aggregation["comprehensive_weaknesses"],
            "thinking_process": [aggregation["aggregation_reasoning"]],
            "rubric_version": "1.0",
            "evaluation_focus": "multi_critic_synthesis"
        }
        
        aggregator_critic_result = CriticResult(
            critic_role="aggregator",
            critic_score=synthetic_critic_score,
            execution_time_ms=round_time,
            confidence=0.9,  # High confidence in synthesis
            focus_summary="Synthesized multiple critic perspectives",
            notable_observations=aggregation["disagreement_analysis"]
        )
        
        debate_round = DebateRound(
            round_number=2,
            round_type="aggregation",
            participants=["aggregator"],
            results=[aggregator_critic_result],
            round_summary=f"Synthesis and aggregation of {len(round1.results)} critic evaluations",
            execution_time_ms=round_time
        )
        
        # Return both the debate round and the actual ScoreAggregation
        return debate_round, ScoreAggregation(**aggregation)
    
    async def _simple_aggregation(self, critic_results: List[CriticResult]) -> ScoreAggregation:
        """Simple aggregation when debate is disabled."""
        scores = [result.critic_score.overall_score for result in critic_results]
        individual_scores = {result.critic_role: result.critic_score.overall_score 
                           for result in critic_results}
        
        final_score = int(round(statistics.mean(scores)))
        variance = statistics.variance(scores) if len(scores) > 1 else 0
        
        # Determine consensus and tier
        consensus = "high" if variance < 5 else ("medium" if variance < 15 else "low")
        
        if final_score >= 90:
            tier = "excellent"
        elif final_score >= 75:
            tier = "good"
        elif final_score >= 60:
            tier = "adequate"
        elif final_score >= 40:
            tier = "poor"
        else:
            tier = "inadequate"
        
        # Collect comprehensive insights
        all_strengths = []
        all_weaknesses = []
        for result in critic_results:
            all_strengths.extend(result.critic_score.key_strengths)
            all_weaknesses.extend(result.critic_score.key_weaknesses)
        
        return ScoreAggregation(
            final_score=final_score,
            final_tier=tier,
            aggregation_method="simple_averaging",
            individual_scores=individual_scores,
            score_variance=variance,
            consensus_level=consensus,
            aggregation_reasoning=f"Simple average of {len(scores)} critic scores",
            disagreement_analysis=[],
            consensus_points=[],
            comprehensive_strengths=list(set(all_strengths)),
            comprehensive_weaknesses=list(set(all_weaknesses)),
            actionable_recommendations=["Review individual critic feedback for specific insights"]
        )
    
    def _generate_summary(self, aggregation: ScoreAggregation, rounds: List[DebateRound]) -> str:
        """Generate high-level summary of the evaluation."""
        num_critics = len(rounds[0].results) if rounds else 0
        variance_text = "low" if aggregation.score_variance < 5 else ("moderate" if aggregation.score_variance < 15 else "high")
        
        return (f"Multi-critic evaluation with {num_critics} specialized critics resulted in "
                f"{aggregation.final_score}/100 ({aggregation.final_tier}). "
                f"Score variance was {variance_text} ({aggregation.score_variance:.1f}), "
                f"indicating {aggregation.consensus_level} consensus among critics.")
    
    def _calculate_confidence(self, rounds: List[DebateRound]) -> float:
        """Calculate overall confidence in the evaluation."""
        if not rounds or not rounds[0].results:
            return 0.0
        
        # Base confidence from number of critics
        num_critics = len(rounds[0].results)
        base_confidence = min(0.5 + (num_critics - 1) * 0.1, 0.9)
        
        # Adjust based on score variance (lower variance = higher confidence)
        scores = [result.critic_score.overall_score for result in rounds[0].results]
        if len(scores) > 1:
            variance = statistics.variance(scores)
            variance_penalty = min(variance / 100, 0.3)  # Max penalty of 0.3
            base_confidence -= variance_penalty
        
        return max(base_confidence, 0.1)  # Minimum confidence of 0.1