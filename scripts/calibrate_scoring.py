#!/usr/bin/env python3
"""
Calibration script for scoring system quality assurance.

This script helps calibrate and validate the automated scoring system by:
1. Selecting representative questions for human labeling
2. Facilitating human evaluation collection
3. Computing alignment metrics between human and automated scores
4. Providing recommendations for system improvement

Usage:
    # Select calibration questions
    python scripts/calibrate_scoring.py select --output configs/calibration_questions.json

    # Collect human evaluations (interactive)
    python scripts/calibrate_scoring.py collect --questions configs/calibration_questions.json

    # Analyze alignment and calibration
    python scripts/calibrate_scoring.py analyze --human data/human_labels.jsonl --automated data/automated_scores.jsonl
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.calibration import (
    CalibrationDataSelector, CalibrationAnalyzer, HumanScore, AutomatedScore,
    save_human_scores, load_human_scores, save_calibration_result
)
from critics.single_critic import SingleCriticAgent
from critics.orchestrator import MultiCriticOrchestrator
from critics.debate_models import MultiCriticRequest
from utils.llm import create_llm_client


def print_banner():
    """Print CLI banner."""
    print("=" * 80)
    print("🎯 Scoring System Calibration & Quality Assurance")
    print("=" * 80)
    print("Purpose: Validate automated scoring against human expert evaluation")
    print("Process: Question Selection → Human Evaluation → Alignment Analysis")
    print("Output: Calibration metrics, recommendations, and improvement areas")
    print("=" * 80)
    print()


def select_calibration_questions(output_file: str, target_size: int, strategy: str):
    """Select representative questions for human evaluation."""
    print("🔍 SELECTING CALIBRATION QUESTIONS")
    print("=" * 50)
    
    # Initialize selector
    selector = CalibrationDataSelector()
    
    print(f"📊 Available questions: {len(selector.questions)}")
    print(f"🎯 Target selection size: {target_size}")
    print(f"📋 Selection strategy: {strategy}")
    print()
    
    # Select questions
    print("🔄 Selecting questions...")
    selected_questions = selector.select_calibration_subset(target_size, strategy)
    
    # Add metadata
    calibration_data = {
        'metadata': {
            'selection_date': datetime.now().isoformat(),
            'total_available': len(selector.questions),
            'selected_count': len(selected_questions),
            'selection_strategy': strategy,
            'purpose': 'human_evaluation_calibration'
        },
        'questions': selected_questions
    }
    
    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"✅ Selected {len(selected_questions)} questions")
    print(f"💾 Saved to: {output_file}")
    
    # Print selection summary
    print("\n📈 SELECTION SUMMARY:")
    categories = {}
    difficulties = {}
    for q in selected_questions:
        cat = q.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        
        diff = q.get('difficulty', 'unknown')
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"Categories: {dict(categories)}")
    print(f"Difficulties: {dict(difficulties)}")
    print()
    
    print("📝 NEXT STEPS:")
    print("1. Review selected questions in the output file")
    print("2. Distribute questions to human evaluators")
    print("3. Use 'collect' command to gather human evaluations")
    print("4. Run 'analyze' command to compute alignment metrics")
    

def get_multiline_input(prompt: str) -> str:
    """Get multiline input from user."""
    print(f"{prompt}")
    print("(Enter your text below. Type 'END' on a new line when finished)")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled.")
            sys.exit(0)
    
    return '\n'.join(lines)


def get_single_input(prompt: str, required: bool = True) -> str:
    """Get single line input from user."""
    while True:
        try:
            value = input(f"{prompt}: ").strip()
            if value or not required:
                return value
            print("This field is required. Please enter a value.")
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled.")
            sys.exit(0)


def get_numeric_input(prompt: str, min_val: float = None, max_val: float = None, 
                     default: float = None) -> float:
    """Get numeric input from user."""
    while True:
        try:
            prompt_text = prompt
            if default is not None:
                prompt_text += f" (default: {default})"
            prompt_text += ": "
            
            value_str = input(prompt_text).strip()
            
            if not value_str and default is not None:
                return default
            
            value = float(value_str)
            
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
                
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
                
            return value
            
        except ValueError:
            print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled.")
            sys.exit(0)


def collect_human_evaluations(questions_file: str, output_file: str, evaluator_id: str):
    """Interactive collection of human evaluations."""
    print("👥 COLLECTING HUMAN EVALUATIONS")
    print("=" * 50)
    
    # Load calibration questions
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    print(f"📋 Loaded {len(questions)} questions for evaluation")
    print(f"👤 Evaluator ID: {evaluator_id}")
    print()
    
    # Load existing evaluations if file exists
    existing_scores = []
    if Path(output_file).exists():
        try:
            existing_scores = load_human_scores(output_file)
            evaluated_ids = {score.question_id for score in existing_scores}
            print(f"📁 Found {len(existing_scores)} existing evaluations")
            
            # Filter out already evaluated questions
            questions = [q for q in questions if q['id'] not in evaluated_ids]
            print(f"📝 {len(questions)} questions remaining to evaluate")
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing scores: {e}")
    
    if not questions:
        print("✅ All questions have been evaluated!")
        return
    
    print("\n" + "=" * 80)
    print("EVALUATION INSTRUCTIONS")
    print("=" * 80)
    print("You will evaluate answers using the same 0-100 rubric as the automated system.")
    print("Focus on: Coverage • Detail • Structure • Style • Instruction Following")
    print("NOTE: Do NOT prioritize factual correctness - focus on presentation quality!")
    print("=" * 80)
    
    # Collect evaluations
    new_scores = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"QUESTION {i}/{len(questions)} - ID: {question['id']}")
        print(f"Category: {question.get('category', 'N/A')} | Difficulty: {question.get('difficulty', 'N/A')}")
        print("="*60)
        
        print("\n📝 QUESTION/PROMPT:")
        print("-" * 40)
        print(question['prompt'])
        print("-" * 40)
        
        # Get sample answer
        print("\n📄 SAMPLE ANSWER TO EVALUATE:")
        answer = get_multiline_input("Paste or type the answer to evaluate")
        
        if not answer.strip():
            print("⚠️  Empty answer. Skipping question.")
            continue
        
        print(f"\n✅ Answer captured ({len(answer)} characters)")
        
        # Evaluation timing start
        eval_start_time = time.time()
        
        # Get overall score
        print("\n📊 SCORING (Use 0-100 scale):")
        overall_score = int(get_numeric_input("Overall score", 0, 100))
        
        # Get dimension scores
        print("\n📋 Dimension scores (0-100 each):")
        dimension_scores = {}
        
        # Standard dimensions from rubric
        dimensions = [
            ("coverage", "Information Coverage"),
            ("detail_specificity", "Detail & Specificity"), 
            ("structure_coherence", "Structure & Coherence"),
            ("style_tone", "Style & Tone"),
            ("instruction_following", "Instruction Following")
        ]
        
        for dim_key, dim_name in dimensions:
            score = int(get_numeric_input(f"{dim_name}", 0, 100))
            dimension_scores[dim_key] = score
        
        # Get justification
        print("\n💭 JUSTIFICATION:")
        justification = get_multiline_input("Explain your scoring reasoning")
        
        # Get confidence and notes
        confidence = get_numeric_input("\n🎯 Your confidence in this evaluation (0.0-1.0)", 0.0, 1.0)
        
        notes = get_single_input("\n📝 Additional notes (optional)", required=False)
        
        eval_time_minutes = (time.time() - eval_start_time) / 60
        
        # Create human score
        human_score = HumanScore(
            question_id=question['id'],
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            justification=justification,
            evaluator_id=evaluator_id,
            evaluation_date=datetime.now().isoformat(),
            confidence=confidence,
            time_spent_minutes=eval_time_minutes,
            notes=notes if notes else None
        )
        
        new_scores.append(human_score)
        
        # Save incrementally
        all_scores = existing_scores + new_scores
        save_human_scores(all_scores, output_file)
        
        print(f"\n✅ Evaluation {i}/{len(questions)} saved")
        
        # Ask if user wants to continue
        if i < len(questions):
            continue_eval = get_single_input("\nContinue to next question? (Y/n)", required=False)
            if continue_eval.lower().startswith('n'):
                print("\n⏸️  Evaluation paused. Progress saved.")
                break
    
    print(f"\n🎉 EVALUATION SESSION COMPLETE!")
    print(f"📊 Total evaluations: {len(existing_scores) + len(new_scores)}")
    print(f"💾 Saved to: {output_file}")
    
    if len(existing_scores) + len(new_scores) >= len(data['questions']):
        print("\n📈 NEXT STEPS:")
        print("1. Collect automated scores for the same questions")
        print("2. Run 'analyze' command to compute alignment metrics")


async def generate_automated_scores(questions_file: str, output_file: str, 
                                   scoring_method: str, llm_provider: str):
    """Generate automated scores for calibration questions."""
    print(f"🤖 GENERATING AUTOMATED SCORES ({scoring_method})")
    print("=" * 60)
    
    # Load questions
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    print(f"📋 Loaded {len(questions)} questions")
    print(f"🎯 Scoring method: {scoring_method}")
    print(f"🔗 LLM provider: {llm_provider}")
    print()
    
    # Initialize LLM client and scoring agents
    llm_client = create_llm_client(llm_provider)
    
    if scoring_method == "single_critic":
        agent = SingleCriticAgent(llm_client=llm_client)
    elif scoring_method == "multi_critic":
        orchestrator = MultiCriticOrchestrator(llm_client)
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")
    
    # Generate sample answers and scores
    automated_scores = []
    
    for i, question in enumerate(questions, 1):
        print(f"🔄 Processing question {i}/{len(questions)} - {question['id']}")
        
        # For calibration, we need actual answers to score
        # This is a limitation - in real usage, you'd have actual answers from your system
        sample_answer = f"[This would be a sample answer for question: {question['prompt'][:100]}...]"
        
        try:
            start_time = time.time()
            
            if scoring_method == "single_critic":
                from critics.models import CriticRequest
                request = CriticRequest(
                    question=question['prompt'],
                    answer=sample_answer,
                    context="Calibration evaluation"
                )
                
                result = await agent.execute(request)
                
                if result.success:
                    critic_score = result.data["critic_score"]
                    
                    # Convert to AutomatedScore format
                    auto_score = AutomatedScore(
                        question_id=question['id'],
                        overall_score=critic_score['overall_score'],
                        dimension_scores={k: v['score'] for k, v in critic_score['dimension_scores'].items()},
                        scoring_method=scoring_method,
                        evaluation_date=datetime.now().isoformat(),
                        execution_time_ms=(time.time() - start_time) * 1000,
                        metadata={'question_category': question.get('category')}
                    )
                    
                    automated_scores.append(auto_score)
                else:
                    print(f"⚠️  Failed to score question {question['id']}: {result.error}")
                    
            elif scoring_method == "multi_critic":
                request = MultiCriticRequest(
                    question=question['prompt'],
                    answer=sample_answer,
                    context="Calibration evaluation",
                    enable_debate=True
                )
                
                result = await orchestrator.evaluate(request)
                
                # Convert to AutomatedScore format
                auto_score = AutomatedScore(
                    question_id=question['id'],
                    overall_score=result.final_aggregation.final_score,
                    dimension_scores={},  # Multi-critic doesn't break down by traditional dimensions
                    scoring_method=scoring_method,
                    evaluation_date=datetime.now().isoformat(),
                    execution_time_ms=result.total_execution_time_ms,
                    confidence=result.confidence_level,
                    metadata={
                        'question_category': question.get('category'),
                        'consensus_level': result.final_aggregation.consensus_level,
                        'critics_used': result.critics_used
                    }
                )
                
                automated_scores.append(auto_score)
                
        except Exception as e:
            print(f"❌ Error scoring question {question['id']}: {e}")
            continue
        
        print(f"   Score: {auto_score.overall_score}/100")
    
    # Save automated scores
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for score in automated_scores:
            score_dict = {
                'question_id': score.question_id,
                'overall_score': score.overall_score,
                'dimension_scores': score.dimension_scores,
                'scoring_method': score.scoring_method,
                'evaluation_date': score.evaluation_date,
                'execution_time_ms': score.execution_time_ms,
                'confidence': score.confidence,
                'metadata': score.metadata
            }
            f.write(json.dumps(score_dict) + '\n')
    
    print(f"\n✅ Generated {len(automated_scores)} automated scores")
    print(f"💾 Saved to: {output_file}")


def analyze_calibration(human_file: str, automated_file: str, output_file: str):
    """Analyze alignment between human and automated scores."""
    print("📊 ANALYZING SCORING ALIGNMENT")
    print("=" * 50)
    
    # Load scores
    print("📁 Loading score files...")
    human_scores = load_human_scores(human_file)
    
    automated_scores = []
    with open(automated_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                score = AutomatedScore(
                    question_id=data['question_id'],
                    overall_score=data['overall_score'],
                    dimension_scores=data['dimension_scores'],
                    scoring_method=data['scoring_method'],
                    evaluation_date=data['evaluation_date'],
                    execution_time_ms=data['execution_time_ms'],
                    confidence=data.get('confidence'),
                    metadata=data.get('metadata')
                )
                automated_scores.append(score)
    
    print(f"👥 Human scores: {len(human_scores)}")
    print(f"🤖 Automated scores: {len(automated_scores)}")
    
    # Perform alignment analysis
    print("\n🔄 Computing alignment metrics...")
    analyzer = CalibrationAnalyzer()
    
    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'human_scores_file': human_file,
        'automated_scores_file': automated_file
    }
    
    result = analyzer.analyze_calibration(human_scores, automated_scores, metadata)
    
    # Display results
    print("\n" + "=" * 70)
    print("📈 CALIBRATION ANALYSIS RESULTS")
    print("=" * 70)
    
    metrics = result.alignment_metrics
    
    print(f"🎯 Overall Quality: {result.calibration_quality.upper()}")
    print(f"🔒 Confidence Level: {result.confidence_level:.2f}")
    print(f"📊 Sample Size: {metrics.n_samples}")
    print()
    
    print("📈 CORRELATION METRICS:")
    print(f"  Pearson Correlation: {metrics.overall_pearson:.3f} (p={metrics.overall_pearson_p_value:.3f})")
    print(f"  Spearman Correlation: {metrics.overall_spearman:.3f} (p={metrics.overall_spearman_p_value:.3f})")
    print()
    
    print("🎯 AGREEMENT METRICS:")
    print(f"  Within 5 points: {metrics.within_5_points_count}/{metrics.n_samples} ({metrics.agreement_percentage_5:.1f}%)")
    print(f"  Within 10 points: {metrics.within_10_points_count}/{metrics.n_samples} ({metrics.agreement_percentage_10:.1f}%)")
    print(f"  Exact agreement: {metrics.exact_agreement_count}/{metrics.n_samples}")
    print()
    
    print("📏 ERROR METRICS:")
    print(f"  Mean Absolute Error: {metrics.mean_absolute_error:.2f}")
    print(f"  Root Mean Square Error: {metrics.root_mean_square_error:.2f}")
    print(f"  Mean Bias Error: {metrics.mean_bias_error:+.2f}")
    if metrics.systematic_bias_detected:
        bias_direction = "higher" if metrics.mean_bias_error > 0 else "lower"
        print(f"  ⚠️  Systematic bias detected: Automated scores are {bias_direction}")
    print()
    
    print("📊 SCORE DISTRIBUTIONS:")
    print(f"  Human:     μ={metrics.human_score_stats['mean']:.1f}, σ={metrics.human_score_stats['std_dev']:.1f}")
    print(f"  Automated: μ={metrics.automated_score_stats['mean']:.1f}, σ={metrics.automated_score_stats['std_dev']:.1f}")
    print()
    
    # Dimension analysis
    if metrics.dimension_correlations:
        print("📋 DIMENSION CORRELATIONS:")
        for dimension, corr_data in metrics.dimension_correlations.items():
            print(f"  {dimension:20s}: r={corr_data['pearson']:5.3f} (n={corr_data['n_samples']})")
        print()
    
    # Recommendations and concerns
    if result.recommendations:
        print("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        print()
    
    if result.areas_of_concern:
        print("⚠️  AREAS OF CONCERN:")
        for i, concern in enumerate(result.areas_of_concern, 1):
            print(f"  {i}. {concern}")
        print()
    
    # Save detailed results
    save_calibration_result(result, output_file)
    print(f"💾 Detailed analysis saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scoring system calibration and quality assurance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select 15 questions for calibration
  python scripts/calibrate_scoring.py select --output configs/calibration_questions.json --size 15

  # Collect human evaluations interactively  
  python scripts/calibrate_scoring.py collect --questions configs/calibration_questions.json --evaluator alice

  # Generate automated scores for comparison
  python scripts/calibrate_scoring.py generate --questions configs/calibration_questions.json --method single_critic

  # Analyze alignment between human and automated scores
  python scripts/calibrate_scoring.py analyze --human data/human_labels.jsonl --automated data/automated_scores.jsonl

  # Run full calibration workflow
  python scripts/calibrate_scoring.py select --size 12
  python scripts/calibrate_scoring.py collect --questions configs/calibration_questions.json --evaluator expert1
  python scripts/calibrate_scoring.py generate --questions configs/calibration_questions.json --method multi_critic
  python scripts/calibrate_scoring.py analyze --human data/human_labels.jsonl --automated data/automated_scores.jsonl
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Select command
    select_parser = subparsers.add_parser('select', help='Select calibration questions')
    select_parser.add_argument('--output', default='configs/calibration_questions.json',
                              help='Output file for selected questions')
    select_parser.add_argument('--size', type=int, default=15,
                              help='Number of questions to select (10-20 recommended)')
    select_parser.add_argument('--strategy', default='balanced_sampling',
                              choices=['balanced_sampling', 'random', 'stratified'],
                              help='Selection strategy')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect human evaluations')
    collect_parser.add_argument('--questions', required=True,
                               help='File with calibration questions')
    collect_parser.add_argument('--output', default='data/human_labels.jsonl',
                               help='Output file for human scores')
    collect_parser.add_argument('--evaluator', required=True,
                               help='Evaluator ID (for tracking)')
    
    # Generate command  
    generate_parser = subparsers.add_parser('generate', help='Generate automated scores')
    generate_parser.add_argument('--questions', required=True,
                                help='File with calibration questions')
    generate_parser.add_argument('--output', default='data/automated_scores.jsonl',
                                help='Output file for automated scores')
    generate_parser.add_argument('--method', default='single_critic',
                                choices=['single_critic', 'multi_critic'],
                                help='Automated scoring method')
    generate_parser.add_argument('--provider', default=None,
                                choices=['gemini', 'claude'],
                                help='LLM provider for scoring (auto-selects based on available API keys)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze score alignment')
    analyze_parser.add_argument('--human', required=True,
                               help='File with human scores (JSONL)')
    analyze_parser.add_argument('--automated', required=True,
                               help='File with automated scores (JSONL)')
    analyze_parser.add_argument('--output', default='data/calibration_analysis.json',
                               help='Output file for analysis results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    print_banner()
    
    if args.command == 'select':
        select_calibration_questions(args.output, args.size, args.strategy)
    
    elif args.command == 'collect':
        collect_human_evaluations(args.questions, args.output, args.evaluator)
    
    elif args.command == 'generate':
        asyncio.run(generate_automated_scores(args.questions, args.output, args.method, args.provider))
    
    elif args.command == 'analyze':
        analyze_calibration(args.human, args.automated, args.output)
    
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()