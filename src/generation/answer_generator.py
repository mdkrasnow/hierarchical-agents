"""
Abstract base class for answer generation strategies.

Defines the interface and core data structures for generating candidate answers
to evaluation questions using various approaches including LLMs and agents.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GenerationStrategy(Enum):
    """Types of answer generation strategies."""
    LLM_GENERIC = "llm_generic"  # Direct LLM prompt
    LLM_CHAIN_OF_THOUGHT = "llm_cot"  # LLM with chain-of-thought
    HIERARCHICAL_AGENT = "hierarchical_agent"  # Use hierarchical agents
    MULTI_STAGE = "multi_stage"  # Multiple generation stages
    ENSEMBLE = "ensemble"  # Multiple strategies combined


@dataclass 
class GenerationConfig:
    """Configuration for answer generation."""
    strategy: GenerationStrategy = GenerationStrategy.LLM_GENERIC
    
    # LLM settings
    temperature: float = 0.7
    max_tokens: int = 2048
    model_name: Optional[str] = None
    
    # Generation strategy settings
    use_chain_of_thought: bool = False
    include_reasoning: bool = False
    target_length: Optional[int] = None  # Target answer length in words
    
    # Agent-specific settings
    agent_types: List[str] = field(default_factory=list)
    coordination_strategy: Optional[str] = None
    
    # Quality control
    enable_validation: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3
    
    # Metadata
    custom_instructions: Optional[str] = None
    generation_context: Dict[str, Any] = field(default_factory=dict)


class GenerationResult(BaseModel):
    """Result of answer generation process."""
    
    # Core results
    answer: str = Field(..., description="Generated answer text")
    success: bool = Field(..., description="Whether generation succeeded")
    
    # Generation metadata
    generation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_used: str = Field(..., description="Strategy that produced this answer")
    generation_time_ms: float = Field(..., description="Time taken to generate answer")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, description="Generator's confidence in answer quality")
    estimated_quality: Optional[str] = Field(None, description="Estimated quality tier (poor/fair/good/excellent)")
    
    # Process information
    reasoning_trace: List[str] = Field(default_factory=list, description="Step-by-step reasoning process")
    intermediate_outputs: Dict[str, Any] = Field(default_factory=dict, description="Intermediate generation outputs")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
    warnings: List[str] = Field(default_factory=list, description="Warnings during generation")
    
    # Resource usage
    llm_calls_made: int = Field(default=0, description="Number of LLM calls made")
    total_tokens_used: int = Field(default=0, description="Total tokens consumed")
    
    # Agent-specific data (if applicable)
    agent_outputs: Dict[str, Any] = Field(default_factory=dict, description="Individual agent outputs")
    coordination_details: Optional[Dict[str, Any]] = Field(None, description="Agent coordination information")
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def add_reasoning_step(self, step: str):
        """Add a step to the reasoning trace."""
        self.reasoning_trace.append(step)
    
    def set_error(self, error_message: str):
        """Mark generation as failed with error message."""
        self.success = False
        self.error_message = error_message


class AnswerGenerator(ABC):
    """
    Abstract base class for answer generation strategies.
    
    Provides a standardized interface for generating candidate answers
    to evaluation questions using various approaches.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize generator with configuration."""
        self.config = config or GenerationConfig()
        self.generator_id = f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this generation strategy."""
        pass
    
    @property
    @abstractmethod
    def supported_question_types(self) -> List[str]:
        """Return list of question types this generator supports well."""
        pass
    
    @abstractmethod
    async def generate_answer(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        config_override: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate an answer to the given question.
        
        Args:
            question: The question/prompt to answer
            context: Additional context for answer generation
            config_override: Override default configuration for this generation
            
        Returns:
            GenerationResult containing the answer and metadata
        """
        pass
    
    async def generate_multiple_answers(
        self,
        questions: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
        config_override: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """
        Generate answers for multiple questions.
        
        Args:
            questions: List of questions/prompts to answer
            contexts: Optional list of contexts (one per question)
            config_override: Override default configuration
            
        Returns:
            List of GenerationResult objects
        """
        if contexts and len(contexts) != len(questions):
            raise ValueError("Number of contexts must match number of questions")
        
        contexts = contexts or [None] * len(questions)
        
        # Generate answers concurrently
        tasks = []
        for question, context in zip(questions, contexts):
            task = self.generate_answer(question, context, config_override)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed GenerationResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_result = GenerationResult(
                    answer="",
                    success=False,
                    strategy_used=self.strategy_name,
                    generation_time_ms=0.0,
                    error_message=f"Generation failed: {str(result)}"
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def validate_answer(
        self,
        question: str,
        answer: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate a generated answer for basic quality checks.
        
        Args:
            question: Original question
            answer: Generated answer to validate
            context: Additional context used in generation
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "quality_score": 0.0,
            "recommendations": []
        }
        
        # Basic checks
        if not answer or not answer.strip():
            validation["is_valid"] = False
            validation["issues"].append("Answer is empty or whitespace only")
            return validation
        
        answer_length = len(answer.split())
        
        # Length checks
        if answer_length < 10:
            validation["issues"].append("Answer is very short (< 10 words)")
            validation["quality_score"] -= 0.2
        elif answer_length > 1000:
            validation["issues"].append("Answer is very long (> 1000 words)")
            validation["quality_score"] -= 0.1
        
        # Content checks
        if answer.lower().strip() == question.lower().strip():
            validation["is_valid"] = False
            validation["issues"].append("Answer is identical to question")
        
        if "I don't know" in answer or "I cannot answer" in answer:
            validation["issues"].append("Answer indicates inability to respond")
            validation["quality_score"] -= 0.3
        
        # Set base quality score
        if len(validation["issues"]) == 0:
            validation["quality_score"] = 0.8
        elif validation["is_valid"]:
            validation["quality_score"] = max(0.3, 0.8 - len(validation["issues"]) * 0.1)
        
        return validation
    
    def _create_base_result(
        self,
        answer: str = "",
        success: bool = True,
        start_time: Optional[float] = None
    ) -> GenerationResult:
        """Create a base GenerationResult with common fields filled."""
        generation_time = (time.time() - start_time) * 1000 if start_time else 0.0
        
        return GenerationResult(
            answer=answer,
            success=success,
            strategy_used=self.strategy_name,
            generation_time_ms=generation_time
        )
    
    def _merge_configs(
        self,
        override: Optional[GenerationConfig]
    ) -> GenerationConfig:
        """Merge override config with default config."""
        if not override:
            return self.config
        
        # Create a copy of the base config and update with overrides
        merged_context = {**self.config.generation_context}
        if override.generation_context:
            merged_context.update(override.generation_context)
        
        merged = GenerationConfig(
            strategy=override.strategy if override.strategy != GenerationStrategy.LLM_GENERIC else self.config.strategy,
            temperature=override.temperature,  # Always use override if provided
            max_tokens=override.max_tokens,    # Always use override if provided
            model_name=override.model_name or self.config.model_name,
            use_chain_of_thought=override.use_chain_of_thought,  # Boolean override
            include_reasoning=override.include_reasoning,        # Boolean override
            target_length=override.target_length or self.config.target_length,
            agent_types=override.agent_types if override.agent_types else self.config.agent_types,
            coordination_strategy=override.coordination_strategy or self.config.coordination_strategy,
            enable_validation=override.enable_validation,       # Boolean override
            retry_on_failure=override.retry_on_failure,         # Boolean override  
            max_retries=override.max_retries,                   # Always use override if provided
            custom_instructions=override.custom_instructions or self.config.custom_instructions,
            generation_context=merged_context
        )
        
        return merged
    
    def __repr__(self) -> str:
        """String representation of generator."""
        return f"{self.__class__.__name__}(id={self.generator_id}, strategy={self.strategy_name})"


# Utility functions for creating generators
def create_generator(
    strategy: Union[str, GenerationStrategy],
    config: Optional[GenerationConfig] = None,
    **kwargs
) -> AnswerGenerator:
    """
    Create an AnswerGenerator instance based on strategy.
    
    Args:
        strategy: Generation strategy to use
        config: Generator configuration
        **kwargs: Additional arguments passed to generator constructor
        
    Returns:
        Initialized AnswerGenerator instance
    """
    if isinstance(strategy, str):
        strategy = GenerationStrategy(strategy)
    
    if strategy == GenerationStrategy.LLM_GENERIC:
        from .llm_generator import LLMGenerator
        return LLMGenerator(config=config, **kwargs)
    elif strategy == GenerationStrategy.HIERARCHICAL_AGENT:
        from .hierarchical_generator import HierarchicalGenerator
        return HierarchicalGenerator(config=config, **kwargs)
    else:
        raise ValueError(f"Unsupported generation strategy: {strategy}")


async def generate_answer_simple(
    question: str,
    strategy: Union[str, GenerationStrategy] = GenerationStrategy.LLM_GENERIC,
    **kwargs
) -> str:
    """
    Simple convenience function to generate a single answer.
    
    Args:
        question: Question to answer
        strategy: Generation strategy to use
        **kwargs: Additional configuration options
        
    Returns:
        Generated answer string
    """
    config = GenerationConfig(**kwargs)
    generator = create_generator(strategy, config)
    result = await generator.generate_answer(question)
    
    if result.success:
        return result.answer
    else:
        raise RuntimeError(f"Answer generation failed: {result.error_message}")