"""
Multi-Critic Agent System - Specialized critics for different evaluation dimensions.

Implements specialized critic agents (Coverage, Depth, Style, Instruction-Following)
that focus on specific aspects of answer evaluation.
"""

import logging
import asyncio
from typing import Dict, Optional, List
from pathlib import Path

from agents.base import BaseAgent, AgentResult
from agents.templates import TemplateManager, FileTemplateLoader
from critics.models import CriticScore, CriticRequest
from critics.debate_models import CriticRole, CriticResult, CriticConfiguration
from critics.single_critic import SingleCriticAgent

logger = logging.getLogger(__name__)


class SpecializedCriticAgent(SingleCriticAgent):
    """
    Base class for specialized critic agents that focus on specific evaluation dimensions.
    
    Extends SingleCriticAgent with specialized prompts and evaluation focus.
    """
    
    def __init__(self, critic_role: CriticRole, *args, **kwargs):
        """Initialize with a specific critic role configuration."""
        self.critic_role = critic_role
        
        # Ensure template manager is configured
        if 'template_manager' not in kwargs:
            template_manager = TemplateManager()
            file_loader = FileTemplateLoader("configs/prompts")
            template_manager.add_loader("file", file_loader)
            kwargs['template_manager'] = template_manager
            
        super().__init__(*args, **kwargs)
    
    @property
    def agent_type(self) -> str:
        """Return the specialized agent type."""
        return f"SpecializedCritic_{self.critic_role.role_name}"
    
    @property
    def role_description(self) -> str:
        """Return description of this critic's specialized role."""
        return f"{self.critic_role.display_name}: {self.critic_role.description}"
    
    async def execute(self, request: CriticRequest, **kwargs) -> AgentResult:
        """
        Execute specialized criticism using the role's specific template and focus.
        
        Args:
            request: CriticRequest with question, answer, and context
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with CriticScore data focused on this critic's specialty
        """
        try:
            self.logger.info(f"Starting {self.critic_role.display_name} evaluation")
            
            # Set execution context specific to this critic
            self.set_context("critic_role", self.critic_role.role_name)
            self.set_context("question", request.question)
            self.set_context("answer", request.answer)
            self.set_context("evaluation_context", request.context or "General evaluation")
            
            # Prepare template variables for specialized prompt
            template_vars = {
                "question": request.question,
                "answer": request.answer,
                "context": request.context or "General evaluation",
                "evaluation_instructions": request.evaluation_instructions or ""
            }
            
            # Use the specialized template for this critic role
            self.logger.info(f"Calling LLM with {self.critic_role.template_name} template")
            result = await self.llm_call_with_template(
                template_name=self.critic_role.template_name,
                template_variables=template_vars,
                response_format=CriticScore,
                temperature=0.3,
                loader_name="file"
            )
            
            # Process and validate the result
            if isinstance(result, CriticScore):
                critic_score = result
                self.logger.info(f"{self.critic_role.display_name} evaluation completed: {critic_score.overall_score}")
            else:
                # Handle unstructured response
                self.logger.warning(f"{self.critic_role.display_name} returned unstructured response")
                critic_score = await self._parse_unstructured_response(result, template_vars)
            
            # Validate the score
            self._validate_critic_score(critic_score)
            
            return AgentResult(
                success=True,
                data={"critic_score": critic_score.dict()},
                agent_id=self.agent_id,
                execution_time_ms=0.0,  # Set by base class
                metadata={
                    "critic_role": self.critic_role.role_name,
                    "critic_display_name": self.critic_role.display_name,
                    "focus_areas": self.critic_role.focus_areas,
                    "question_length": len(request.question),
                    "answer_length": len(request.answer),
                    "overall_score": critic_score.overall_score,
                    "overall_tier": critic_score.overall_tier,
                    "evaluation_focus": critic_score.evaluation_focus
                }
            )
            
        except Exception as e:
            self.logger.error(f"{self.critic_role.display_name} evaluation failed: {e}")
            return AgentResult(
                success=False,
                error=f"{self.critic_role.display_name} evaluation failed: {str(e)}",
                agent_id=self.agent_id,
                execution_time_ms=0.0,
                metadata={
                    "critic_role": self.critic_role.role_name,
                    "critic_display_name": self.critic_role.display_name,
                    "question_length": len(request.question) if request.question else 0,
                    "answer_length": len(request.answer) if request.answer else 0
                }
            )


class CoverageCritic(SpecializedCriticAgent):
    """Specialized critic focusing on information coverage and completeness."""
    
    def __init__(self, *args, **kwargs):
        role = CriticRole(
            role_name="coverage",
            display_name="Coverage Critic",
            description="Evaluates completeness and breadth of information addressing the question",
            focus_areas=["completeness", "breadth", "scope", "gap_identification", "question_alignment"],
            template_name="coverage_critic",
            weight=0.3
        )
        super().__init__(critic_role=role, *args, **kwargs)


class DepthCritic(SpecializedCriticAgent):
    """Specialized critic focusing on detail level, specificity, and technical depth."""
    
    def __init__(self, *args, **kwargs):
        role = CriticRole(
            role_name="depth",
            display_name="Depth Critic", 
            description="Evaluates detail level, specificity, and technical depth of information",
            focus_areas=["technical_depth", "specific_details", "evidence_quality", "granularity", "substantiveness"],
            template_name="depth_critic",
            weight=0.25
        )
        super().__init__(critic_role=role, *args, **kwargs)


class StyleCritic(SpecializedCriticAgent):
    """Specialized critic focusing on writing style, tone, and presentation."""
    
    def __init__(self, *args, **kwargs):
        role = CriticRole(
            role_name="style",
            display_name="Style Critic",
            description="Evaluates writing style, tone, and presentation quality",
            focus_areas=["writing_clarity", "tone_appropriateness", "flow_coherence", "language_quality", "engagement"],
            template_name="style_critic", 
            weight=0.20
        )
        super().__init__(critic_role=role, *args, **kwargs)


class InstructionCritic(SpecializedCriticAgent):
    """Specialized critic focusing on adherence to instructions and requirements."""
    
    def __init__(self, *args, **kwargs):
        role = CriticRole(
            role_name="instruction_following",
            display_name="Instruction-Following Critic",
            description="Evaluates adherence to specific instructions, format requirements, and constraints",
            focus_areas=["requirement_compliance", "format_adherence", "constraint_respect", "instruction_precision", "task_alignment"],
            template_name="instruction_critic",
            weight=0.15
        )
        super().__init__(critic_role=role, *args, **kwargs)


class CriticFactory:
    """Factory for creating specialized critic instances."""
    
    _critic_classes = {
        "coverage": CoverageCritic,
        "depth": DepthCritic,
        "style": StyleCritic,
        "instruction_following": InstructionCritic
    }
    
    @classmethod
    def create_critic(cls, role_name: str, llm_client=None, **kwargs) -> SpecializedCriticAgent:
        """
        Create a specialized critic instance.
        
        Args:
            role_name: Name of the critic role (coverage, depth, style, instruction_following)
            llm_client: LLM client to use
            **kwargs: Additional arguments for critic initialization
            
        Returns:
            SpecializedCriticAgent instance
            
        Raises:
            ValueError: If role_name is not supported
        """
        if role_name not in cls._critic_classes:
            available = list(cls._critic_classes.keys())
            raise ValueError(f"Unknown critic role '{role_name}'. Available: {available}")
        
        critic_class = cls._critic_classes[role_name]
        return critic_class(llm_client=llm_client, **kwargs)
    
    @classmethod
    def create_all_critics(cls, llm_client=None, **kwargs) -> Dict[str, SpecializedCriticAgent]:
        """
        Create all available specialized critics.
        
        Args:
            llm_client: LLM client to use for all critics
            **kwargs: Additional arguments for critic initialization
            
        Returns:
            Dict mapping role names to critic instances
        """
        critics = {}
        for role_name in cls._critic_classes:
            critics[role_name] = cls.create_critic(role_name, llm_client=llm_client, **kwargs)
        return critics
    
    @classmethod
    def get_available_roles(cls) -> List[str]:
        """Get list of available critic role names."""
        return list(cls._critic_classes.keys())
    
    @classmethod
    def get_default_configuration(cls) -> CriticConfiguration:
        """Get default configuration for the multi-critic system."""
        # Create role definitions
        roles = []
        for role_name, critic_class in cls._critic_classes.items():
            # Create temporary instance to get role info
            temp_critic = critic_class()
            roles.append(temp_critic.critic_role)
        
        return CriticConfiguration(
            available_critics=roles,
            default_critics=list(cls._critic_classes.keys()),
            aggregation_method="reasoned_synthesis",
            enable_parallel_execution=True,
            min_critics=2,
            max_critics=6,
            score_variance_threshold=15.0,
            critic_timeout_ms=30000,
            aggregator_timeout_ms=45000,
            max_retries=2
        )


# Convenience functions for direct usage

async def evaluate_with_coverage_critic(question: str, answer: str, context: str = None, 
                                       llm_client=None) -> CriticScore:
    """Evaluate answer using Coverage Critic."""
    critic = CoverageCritic(llm_client=llm_client)
    request = CriticRequest(question=question, answer=answer, context=context)
    result = await critic.execute(request)
    if result.success:
        return CriticScore(**result.data["critic_score"])
    else:
        raise RuntimeError(f"Coverage evaluation failed: {result.error}")


async def evaluate_with_depth_critic(question: str, answer: str, context: str = None,
                                    llm_client=None) -> CriticScore:
    """Evaluate answer using Depth Critic."""
    critic = DepthCritic(llm_client=llm_client)
    request = CriticRequest(question=question, answer=answer, context=context)
    result = await critic.execute(request)
    if result.success:
        return CriticScore(**result.data["critic_score"])
    else:
        raise RuntimeError(f"Depth evaluation failed: {result.error}")


async def evaluate_with_style_critic(question: str, answer: str, context: str = None,
                                    llm_client=None) -> CriticScore:
    """Evaluate answer using Style Critic.""" 
    critic = StyleCritic(llm_client=llm_client)
    request = CriticRequest(question=question, answer=answer, context=context)
    result = await critic.execute(request)
    if result.success:
        return CriticScore(**result.data["critic_score"])
    else:
        raise RuntimeError(f"Style evaluation failed: {result.error}")


async def evaluate_with_instruction_critic(question: str, answer: str, context: str = None,
                                          llm_client=None) -> CriticScore:
    """Evaluate answer using Instruction-Following Critic."""
    critic = InstructionCritic(llm_client=llm_client)
    request = CriticRequest(question=question, answer=answer, context=context)
    result = await critic.execute(request)
    if result.success:
        return CriticScore(**result.data["critic_score"])
    else:
        raise RuntimeError(f"Instruction-following evaluation failed: {result.error}")


async def evaluate_with_all_critics(question: str, answer: str, context: str = None,
                                   llm_client=None) -> Dict[str, CriticScore]:
    """
    Evaluate answer using all specialized critics in parallel.
    
    Args:
        question: Original question
        answer: Answer to evaluate
        context: Optional context
        llm_client: LLM client to use
        
    Returns:
        Dict mapping critic role names to their CriticScore results
    """
    critics = CriticFactory.create_all_critics(llm_client=llm_client)
    request = CriticRequest(question=question, answer=answer, context=context)
    
    # Run all critics in parallel
    tasks = [critic.execute(request) for critic in critics.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    critic_scores = {}
    for i, (role_name, result) in enumerate(zip(critics.keys(), results)):
        if isinstance(result, Exception):
            logger.error(f"Critic {role_name} failed with exception: {result}")
            continue
            
        if result.success:
            critic_scores[role_name] = CriticScore(**result.data["critic_score"])
        else:
            logger.error(f"Critic {role_name} failed: {result.error}")
    
    return critic_scores