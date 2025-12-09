"""
SingleCriticAgent implementation for evaluating answers based on coverage/detail/style.

Emphasizes information coverage, detail, and writing style over factual correctness.
Uses the scoring rubric system to provide structured, justified evaluations.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

from agents.base import BaseAgent, AgentResult
from agents.templates import TemplateManager, FileTemplateLoader
from critics.models import CriticScore, CriticRequest, DimensionScore


logger = logging.getLogger(__name__)


class SingleCriticAgent(BaseAgent):
    """
    Single critic agent that evaluates answers using coverage/detail/style rubric.
    
    Key features:
    - Emphasizes presentation quality over factual correctness
    - Uses weighted scoring across multiple dimensions
    - Provides detailed justifications and step-by-step reasoning
    - Returns structured CriticScore with 0-100 scores
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the SingleCriticAgent."""
        # Configure template manager with file loader if needed
        if 'template_manager' not in kwargs:
            template_manager = TemplateManager()
            file_loader = FileTemplateLoader("configs/prompts")
            template_manager.add_loader("file", file_loader)
            kwargs['template_manager'] = template_manager
        
        super().__init__(*args, **kwargs)
        self._rubric_config = None
        self._load_rubric_config()
    
    @property
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        return "SingleCriticAgent"
    
    @property  
    def role_description(self) -> str:
        """Return description of this agent's role."""
        return "Evaluates answers based on information coverage, detail, and writing style quality (not factual correctness)"
    
    def _load_rubric_config(self):
        """Load the scoring rubric configuration."""
        try:
            rubric_path = Path("configs/scoring_rubric.yaml")
            if rubric_path.exists():
                with open(rubric_path, 'r') as f:
                    self._rubric_config = yaml.safe_load(f)
                    self.logger.info("Loaded scoring rubric configuration")
            else:
                self.logger.warning(f"Rubric config not found at {rubric_path}, will use defaults")
                self._rubric_config = self._get_default_rubric_config()
        except Exception as e:
            self.logger.error(f"Failed to load rubric config: {e}")
            self._rubric_config = self._get_default_rubric_config()
    
    def _get_default_rubric_config(self) -> Dict:
        """Get default rubric configuration if file loading fails."""
        return {
            "dimensions": {
                "coverage": {"name": "Information Coverage", "weight": 30},
                "detail_specificity": {"name": "Detail & Specificity", "weight": 25},
                "structure_coherence": {"name": "Structure & Coherence", "weight": 20},
                "style_tone": {"name": "Style & Tone", "weight": 15},
                "instruction_following": {"name": "Instruction Following", "weight": 10}
            }
        }
    
    def _get_tier_from_score(self, score: int) -> str:
        """Convert numeric score to quality tier."""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "adequate"
        elif score >= 40:
            return "poor"
        else:
            return "inadequate"
    
    async def execute(self, request: CriticRequest, **kwargs) -> AgentResult:
        """
        Evaluate an answer using the scoring rubric.
        
        Args:
            request: CriticRequest with question, answer, and context
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with CriticScore data
        """
        try:
            self.logger.info(f"Starting evaluation for question: {request.question[:100]}...")
            
            # Set execution context
            self.set_context("question", request.question)
            self.set_context("answer", request.answer)
            self.set_context("evaluation_context", request.context or "General evaluation")
            
            # Prepare template variables
            template_vars = {
                "question": request.question,
                "answer": request.answer,
                "context": request.context or "General evaluation",
                "evaluation_instructions": request.evaluation_instructions or ""
            }
            
            # Make LLM call with template
            self.logger.info("Calling LLM for answer evaluation")
            result = await self.llm_call_with_template(
                template_name="critic_agent",
                template_variables=template_vars,
                response_format=CriticScore,
                temperature=0.3,  # Lower temperature for more consistent scoring
                loader_name="file"  # Use the file loader for templates
            )
            
            # Validate and process the result
            if isinstance(result, CriticScore):
                critic_score = result
                self.logger.info(f"Evaluation completed with overall score: {critic_score.overall_score}")
            else:
                # Handle case where LLM returns raw text instead of structured data
                self.logger.warning("LLM returned unstructured response, attempting to parse")
                critic_score = await self._parse_unstructured_response(result, template_vars)
            
            # Additional validation
            self._validate_critic_score(critic_score)
            
            return AgentResult(
                success=True,
                data={"critic_score": critic_score.dict()},
                agent_id=self.agent_id,
                execution_time_ms=0.0,  # Will be set by base class
                metadata={
                    "question_length": len(request.question),
                    "answer_length": len(request.answer),
                    "overall_score": critic_score.overall_score,
                    "overall_tier": critic_score.overall_tier
                }
            )
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return AgentResult(
                success=False,
                error=f"Evaluation failed: {str(e)}",
                agent_id=self.agent_id,
                execution_time_ms=0.0,
                metadata={
                    "question_length": len(request.question) if request.question else 0,
                    "answer_length": len(request.answer) if request.answer else 0
                }
            )
    
    async def _parse_unstructured_response(self, raw_response: str, template_vars: Dict) -> CriticScore:
        """
        Fallback parser for when LLM returns unstructured text.
        
        Creates a basic CriticScore with default scoring if structured parsing fails.
        """
        self.logger.warning("Using fallback parsing for unstructured LLM response")
        
        # Create basic dimension scores using rubric config
        dimension_scores = {}
        rubric_dimensions = self._rubric_config.get("dimensions", {})
        
        for dim_key, dim_config in rubric_dimensions.items():
            dimension_scores[dim_key] = DimensionScore(
                dimension_name=dim_config["name"],
                score=70,  # Default adequate score
                tier="adequate",
                justification=f"Unable to parse detailed evaluation for {dim_config['name']}",
                weight=dim_config["weight"],
                weighted_score=(70 * dim_config["weight"] / 100)
            )
        
        return CriticScore(
            overall_score=70,
            overall_tier="adequate",
            dimension_scores=dimension_scores,
            overall_justification="Evaluation parsing failed, using default adequate scores",
            key_strengths=["Response provided but evaluation details unclear"],
            key_weaknesses=["Unable to parse detailed evaluation"],
            thinking_process=["LLM response was unstructured", "Applied default scoring"],
            rubric_version="1.0",
            evaluation_focus="coverage_detail_style"
        )
    
    def _validate_critic_score(self, critic_score: CriticScore):
        """Validate the CriticScore for correctness and consistency."""
        
        # Check score ranges
        if not (0 <= critic_score.overall_score <= 100):
            raise ValueError(f"Overall score {critic_score.overall_score} outside valid range 0-100")
        
        # Check dimension scores
        for dim_name, dim_score in critic_score.dimension_scores.items():
            if not (0 <= dim_score.score <= 100):
                raise ValueError(f"Dimension score {dim_score.score} for {dim_name} outside valid range 0-100")
            
            # Verify tier matches score
            expected_tier = self._get_tier_from_score(dim_score.score)
            if dim_score.tier != expected_tier:
                self.logger.warning(f"Tier mismatch for {dim_name}: score {dim_score.score} vs tier {dim_score.tier}")
        
        # Check that we have expected dimensions
        expected_dimensions = set(self._rubric_config.get("dimensions", {}).keys())
        actual_dimensions = set(critic_score.dimension_scores.keys())
        
        if expected_dimensions != actual_dimensions:
            missing = expected_dimensions - actual_dimensions
            extra = actual_dimensions - expected_dimensions
            self.logger.warning(f"Dimension mismatch - Missing: {missing}, Extra: {extra}")
    
    async def evaluate_batch(self, requests: list[CriticRequest]) -> list[AgentResult]:
        """
        Evaluate multiple answers in batch.
        
        Args:
            requests: List of CriticRequest objects
            
        Returns:
            List of AgentResult objects with evaluations
        """
        self.logger.info(f"Starting batch evaluation of {len(requests)} items")
        
        results = []
        for i, request in enumerate(requests):
            self.logger.info(f"Processing batch item {i+1}/{len(requests)}")
            result = await self.execute(request)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Batch evaluation completed: {successful}/{len(requests)} successful")
        
        return results


# Convenience functions for direct usage

async def score_answer(question: str, answer: str, context: str = None, 
                      llm_client=None) -> CriticScore:
    """
    Convenience function to score a single answer.
    
    Args:
        question: The original question
        answer: The answer to evaluate
        context: Optional context for evaluation
        llm_client: Optional LLM client (will create default if not provided)
        
    Returns:
        CriticScore with detailed evaluation
    """
    from utils.llm import create_llm_client, LLMClient
    
    if llm_client is None:
        llm_client = create_llm_client("mock")  # Use mock by default
    
    agent = SingleCriticAgent(llm_client=llm_client)
    
    request = CriticRequest(
        question=question,
        answer=answer,
        context=context
    )
    
    result = await agent.execute(request)
    
    if result.success:
        return CriticScore(**result.data["critic_score"])
    else:
        raise RuntimeError(f"Evaluation failed: {result.error}")


async def score_answers_batch(questions: list[str], answers: list[str], 
                             contexts: list[str] = None, llm_client=None) -> list[CriticScore]:
    """
    Convenience function to score multiple answers in batch.
    
    Args:
        questions: List of original questions
        answers: List of answers to evaluate
        contexts: Optional list of contexts for evaluation
        llm_client: Optional LLM client (will create default if not provided)
        
    Returns:
        List of CriticScore objects with detailed evaluations
    """
    from utils.llm import create_llm_client
    
    if len(questions) != len(answers):
        raise ValueError("Questions and answers lists must have same length")
    
    if contexts and len(contexts) != len(questions):
        raise ValueError("Contexts list must match questions length if provided")
    
    contexts = contexts or [None] * len(questions)
    
    if llm_client is None:
        llm_client = create_llm_client("mock")  # Use mock by default
    
    agent = SingleCriticAgent(llm_client=llm_client)
    
    requests = [
        CriticRequest(question=q, answer=a, context=c)
        for q, a, c in zip(questions, answers, contexts)
    ]
    
    results = await agent.evaluate_batch(requests)
    
    scores = []
    for result in results:
        if result.success:
            scores.append(CriticScore(**result.data["critic_score"]))
        else:
            # Create an error score
            scores.append(CriticScore(
                overall_score=0,
                overall_tier="inadequate",
                dimension_scores={},
                overall_justification=f"Evaluation failed: {result.error}",
                key_strengths=[],
                key_weaknesses=["Evaluation system error"],
                thinking_process=[f"Error occurred: {result.error}"],
                rubric_version="1.0",
                evaluation_focus="coverage_detail_style"
            ))
    
    return scores