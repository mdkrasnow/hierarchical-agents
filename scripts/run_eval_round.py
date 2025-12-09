#!/usr/bin/env python3
"""
Full evaluation round script for meta-evaluation system.

This script runs a complete evaluation round: question loading → answer generation 
→ multi-critic scoring → result logging. Supports multiple models, batch processing,
and different answer generation strategies.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generation import (
    AnswerGenerator, GenerationConfig, GenerationStrategy, 
    LLMGenerator, HierarchicalGenerator, create_generator
)
from critics.single_critic import score_answer
from critics.models import CriticRequest
from scoring.loaders import QuestionLoader, get_default_loader
from scoring.dataset import Question, QuestionDataset
from utils.llm import create_llm_client, LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvalRoundConfig:
    """Configuration for evaluation round."""
    
    def __init__(
        self,
        questions_source: str = "default",
        model: str = "mock",
        run_id: str = None,
        n_samples: int = 5,
        generation_strategy: str = "llm_generic",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_chain_of_thought: bool = False,
        enable_scoring: bool = True,
        output_dir: str = "eval_results",
        save_individual_results: bool = True,
        questions_filter: Dict[str, Any] = None,
        custom_instructions: str = None
    ):
        self.questions_source = questions_source
        self.model = model
        self.run_id = run_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.n_samples = n_samples
        self.generation_strategy = generation_strategy
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_chain_of_thought = use_chain_of_thought
        self.enable_scoring = enable_scoring
        self.output_dir = Path(output_dir)
        self.save_individual_results = save_individual_results
        self.questions_filter = questions_filter or {}
        self.custom_instructions = custom_instructions


class EvalRoundResult:
    """Results from a complete evaluation round."""
    
    def __init__(self, config: EvalRoundConfig):
        self.config = config
        self.start_time = time.time()
        self.end_time = None
        self.questions_processed = []
        self.generation_results = []
        self.scoring_results = []
        self.errors = []
        self.summary_stats = {}
    
    def add_question_result(self, question: Question, generation_result, scoring_result=None, error=None):
        """Add results for a single question."""
        self.questions_processed.append(question.id)
        self.generation_results.append(generation_result)
        
        if scoring_result:
            self.scoring_results.append(scoring_result)
        
        if error:
            self.errors.append({
                "question_id": question.id,
                "error": str(error),
                "timestamp": time.time()
            })
    
    def finalize(self):
        """Finalize the evaluation round and compute statistics."""
        self.end_time = time.time()
        
        # Compute summary statistics
        total_questions = len(self.questions_processed)
        successful_generations = sum(1 for r in self.generation_results if r and r.success)
        successful_scorings = len(self.scoring_results)
        
        generation_times = [r.generation_time_ms for r in self.generation_results if r and r.success]
        avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
        
        if self.scoring_results:
            scores = [r.overall_score for r in self.scoring_results if hasattr(r, 'overall_score')]
            avg_score = sum(scores) / len(scores) if scores else 0
        else:
            avg_score = 0
        
        self.summary_stats = {
            "run_id": self.config.run_id,
            "total_runtime_seconds": self.end_time - self.start_time,
            "total_questions": total_questions,
            "successful_generations": successful_generations,
            "successful_scorings": successful_scorings,
            "generation_success_rate": successful_generations / total_questions if total_questions > 0 else 0,
            "scoring_success_rate": successful_scorings / successful_generations if successful_generations > 0 else 0,
            "avg_generation_time_ms": avg_generation_time,
            "avg_score": avg_score,
            "total_errors": len(self.errors),
            "config": {
                "model": self.config.model,
                "strategy": self.config.generation_strategy,
                "n_samples": self.config.n_samples,
                "temperature": self.config.temperature
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "summary": self.summary_stats,
            "questions_processed": self.questions_processed,
            "generation_results": [
                r.dict() if r and hasattr(r, 'dict') else str(r) 
                for r in self.generation_results
            ],
            "scoring_results": [
                r.dict() if hasattr(r, 'dict') else str(r)
                for r in self.scoring_results
            ],
            "errors": self.errors,
            "timestamp": datetime.now().isoformat()
        }


async def load_questions(config: EvalRoundConfig) -> QuestionDataset:
    """Load questions based on configuration."""
    logger.info(f"Loading questions from: {config.questions_source}")
    
    if config.questions_source == "default":
        loader = get_default_loader()
        questions = loader.create_test_set(n_questions=config.n_samples, **config.questions_filter)
    else:
        # Custom question source
        try:
            loader = QuestionLoader(questions_path=config.questions_source)
            all_questions = loader.get_all_questions()
            if config.questions_filter:
                questions = all_questions.filter_questions(**config.questions_filter)
            else:
                questions = all_questions
            
            # Sample if more questions than requested
            if len(questions.questions) > config.n_samples:
                questions = questions.sample_questions(config.n_samples)
                
        except Exception as e:
            logger.error(f"Failed to load custom questions: {e}")
            logger.info("Falling back to default question loader")
            loader = get_default_loader()
            questions = loader.create_test_set(n_questions=config.n_samples, **config.questions_filter)
    
    logger.info(f"Loaded {len(questions.questions)} questions for evaluation")
    return questions


async def create_answer_generator(config: EvalRoundConfig, llm_client: LLMClient) -> AnswerGenerator:
    """Create answer generator based on configuration."""
    generation_config = GenerationConfig(
        strategy=GenerationStrategy(config.generation_strategy),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        use_chain_of_thought=config.use_chain_of_thought,
        custom_instructions=config.custom_instructions
    )
    
    if config.generation_strategy == "hierarchical_agent":
        generator = HierarchicalGenerator(llm_client=llm_client, config=generation_config)
    else:
        generator = LLMGenerator(llm_client=llm_client, config=generation_config)
    
    logger.info(f"Created {generator.strategy_name} generator")
    return generator


async def generate_answer_for_question(
    question: Question,
    generator: AnswerGenerator,
    config: EvalRoundConfig
) -> Any:
    """Generate answer for a single question."""
    logger.debug(f"Generating answer for question: {question.id}")
    
    # Prepare context
    context = {
        "question_id": question.id,
        "category": question.category,
        "difficulty": question.difficulty,
        "tags": question.tags
    }
    
    try:
        result = await generator.generate_answer(
            question=question.prompt,
            context=context
        )
        
        logger.debug(f"Generated answer for {question.id}: {len(result.answer.split())} words, success={result.success}")
        return result
        
    except Exception as e:
        logger.error(f"Answer generation failed for question {question.id}: {e}")
        # Return a failed result
        from generation.answer_generator import GenerationResult
        return GenerationResult(
            answer="",
            success=False,
            strategy_used=generator.strategy_name,
            generation_time_ms=0.0,
            error_message=str(e)
        )


async def score_generated_answer(
    question: Question,
    answer: str,
    llm_client: LLMClient
) -> Optional[Any]:
    """Score a generated answer using the critic system."""
    if not answer or not answer.strip():
        logger.warning(f"Skipping scoring for question {question.id} - no answer generated")
        return None
    
    try:
        logger.debug(f"Scoring answer for question: {question.id}")
        
        score = await score_answer(
            question=question.prompt,
            answer=answer,
            context=f"Category: {question.category}, Difficulty: {question.difficulty}",
            llm_client=llm_client
        )
        
        logger.debug(f"Scored answer for {question.id}: {score.overall_score}/100 ({score.overall_tier})")
        return score
        
    except Exception as e:
        logger.error(f"Answer scoring failed for question {question.id}: {e}")
        return None


async def process_single_question(
    question: Question,
    generator: AnswerGenerator,
    llm_client: LLMClient,
    config: EvalRoundConfig,
    result_tracker: EvalRoundResult
) -> Dict[str, Any]:
    """Process a single question through the full pipeline."""
    question_start_time = time.time()
    
    try:
        # Generate answer
        generation_result = await generate_answer_for_question(question, generator, config)
        
        # Score answer if generation succeeded and scoring is enabled
        scoring_result = None
        if generation_result.success and config.enable_scoring:
            scoring_result = await score_generated_answer(
                question, generation_result.answer, llm_client
            )
        
        # Track results
        result_tracker.add_question_result(
            question=question,
            generation_result=generation_result,
            scoring_result=scoring_result
        )
        
        processing_time = (time.time() - question_start_time) * 1000
        
        return {
            "question_id": question.id,
            "question": question.prompt,
            "category": question.category,
            "difficulty": question.difficulty,
            "answer": generation_result.answer if generation_result.success else None,
            "generation_success": generation_result.success,
            "generation_time_ms": generation_result.generation_time_ms,
            "generation_error": generation_result.error_message,
            "score": scoring_result.overall_score if scoring_result else None,
            "score_tier": scoring_result.overall_tier if scoring_result else None,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Failed to process question {question.id}: {e}")
        result_tracker.add_question_result(question=question, generation_result=None, error=e)
        return {
            "question_id": question.id,
            "error": str(e)
        }


async def run_evaluation_round(config: EvalRoundConfig) -> EvalRoundResult:
    """Run a complete evaluation round."""
    logger.info(f"Starting evaluation round: {config.run_id}")
    logger.info(f"Configuration: {config.generation_strategy} with {config.model}, {config.n_samples} questions")
    
    result_tracker = EvalRoundResult(config)
    
    try:
        # Setup LLM client
        llm_client = create_llm_client(
            provider_type=config.model
        )
        
        # Load questions
        questions = await load_questions(config)
        
        # Create answer generator
        generator = await create_answer_generator(config, llm_client)
        
        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(questions.questions)} questions...")
        
        # Process questions - can be done in parallel
        if config.n_samples <= 5:
            # Process in parallel for small batches
            tasks = [
                process_single_question(question, generator, llm_client, config, result_tracker)
                for question in questions.questions
            ]
            question_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially for larger batches to avoid overwhelming APIs
            question_results = []
            for i, question in enumerate(questions.questions, 1):
                logger.info(f"Processing question {i}/{len(questions.questions)}: {question.id}")
                result = await process_single_question(question, generator, llm_client, config, result_tracker)
                question_results.append(result)
        
        # Save individual results if requested
        if config.save_individual_results:
            individual_results_file = config.output_dir / f"{config.run_id}_individual.json"
            with open(individual_results_file, 'w') as f:
                json.dump(question_results, f, indent=2, default=str)
            logger.info(f"Saved individual results to: {individual_results_file}")
        
        # Finalize results
        result_tracker.finalize()
        
        # Save summary results
        summary_file = config.output_dir / f"{config.run_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(result_tracker.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Evaluation round completed: {config.run_id}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return result_tracker
        
    except Exception as e:
        logger.error(f"Evaluation round failed: {e}")
        result_tracker.errors.append({
            "error": str(e),
            "timestamp": time.time(),
            "phase": "overall"
        })
        result_tracker.finalize()
        return result_tracker


def print_results_summary(result: EvalRoundResult):
    """Print a human-readable summary of results."""
    stats = result.summary_stats
    
    print("\n" + "=" * 60)
    print(f"🎯 EVALUATION ROUND SUMMARY: {stats['run_id']}")
    print("=" * 60)
    
    print(f"📊 Overall Statistics:")
    print(f"  Total Questions: {stats['total_questions']}")
    print(f"  Runtime: {stats['total_runtime_seconds']:.1f}s")
    print(f"  Generation Success: {stats['successful_generations']}/{stats['total_questions']} ({stats['generation_success_rate']:.1%})")
    print(f"  Scoring Success: {stats['successful_scorings']}/{stats['successful_generations']} ({stats['scoring_success_rate']:.1%})")
    print(f"  Avg Generation Time: {stats['avg_generation_time_ms']:.0f}ms")
    print(f"  Errors: {stats['total_errors']}")
    
    if stats['avg_score'] > 0:
        print(f"\n📈 Scoring Results:")
        print(f"  Average Score: {stats['avg_score']:.1f}/100")
    
    print(f"\n⚙️  Configuration:")
    print(f"  Model: {stats['config']['model']}")
    print(f"  Strategy: {stats['config']['strategy']}")
    print(f"  Temperature: {stats['config']['temperature']}")
    
    if result.errors:
        print(f"\n⚠️  Errors Encountered:")
        for error in result.errors[-5:]:  # Show last 5 errors
            print(f"  - {error.get('question_id', 'Unknown')}: {error['error']}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")
    
    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete evaluation rounds: question loading → answer generation → scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with 5 questions
  python scripts/run_eval_round.py --n-samples 5
  
  # Use hierarchical agents for answer generation
  python scripts/run_eval_round.py --strategy hierarchical_agent --n-samples 3
  
  # Use Claude model with chain-of-thought
  python scripts/run_eval_round.py --model claude --cot --n-samples 10
  
  # Filter to specific question categories
  python scripts/run_eval_round.py --filter-category "danielson-evaluation" --n-samples 5
  
  # Custom run with specific parameters
  python scripts/run_eval_round.py --run-id "test_run_1" --temperature 0.2 --max-tokens 1000
        """
    )
    
    # Core options
    parser.add_argument('--questions', '-q', default='default',
                       help='Questions source file or "default" for built-in questions')
    parser.add_argument('--model', '-m', default='mock',
                       choices=['mock', 'claude'],
                       help='LLM model to use (default: mock)')
    parser.add_argument('--run-id', help='Unique identifier for this evaluation run')
    parser.add_argument('--n-samples', '-n', type=int, default=5,
                       help='Number of questions to evaluate (default: 5)')
    
    # Generation strategy options
    parser.add_argument('--strategy', default='llm_generic',
                       choices=['llm_generic', 'hierarchical_agent'],
                       help='Answer generation strategy (default: llm_generic)')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                       help='LLM temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=2048,
                       help='Maximum tokens for generation (default: 2048)')
    parser.add_argument('--cot', '--chain-of-thought', action='store_true',
                       help='Enable chain-of-thought reasoning')
    parser.add_argument('--custom-instructions',
                       help='Custom instructions for answer generation')
    
    # Question filtering
    parser.add_argument('--filter-category', help='Filter questions by category')
    parser.add_argument('--filter-difficulty', help='Filter questions by difficulty')
    parser.add_argument('--filter-tag', help='Filter questions by tag')
    
    # Output options
    parser.add_argument('--output-dir', '-o', default='eval_results',
                       help='Output directory for results (default: eval_results)')
    parser.add_argument('--no-scoring', action='store_true',
                       help='Skip answer scoring step')
    parser.add_argument('--no-individual-results', action='store_true',
                       help='Skip saving individual question results')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Build questions filter
    questions_filter = {}
    if args.filter_category:
        questions_filter['category'] = args.filter_category
    if args.filter_difficulty:
        questions_filter['difficulty'] = args.filter_difficulty
    if args.filter_tag:
        questions_filter['has_tag'] = args.filter_tag
    
    # Create configuration
    config = EvalRoundConfig(
        questions_source=args.questions,
        model=args.model,
        run_id=args.run_id,
        n_samples=args.n_samples,
        generation_strategy=args.strategy,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_chain_of_thought=args.cot,
        enable_scoring=not args.no_scoring,
        output_dir=args.output_dir,
        save_individual_results=not args.no_individual_results,
        questions_filter=questions_filter,
        custom_instructions=args.custom_instructions
    )
    
    # Run evaluation
    try:
        result = asyncio.run(run_evaluation_round(config))
        
        if not args.quiet:
            print_results_summary(result)
        
        # Exit with error code if there were significant failures
        if result.summary_stats['generation_success_rate'] < 0.5:
            logger.error("Less than 50% of generations succeeded")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()