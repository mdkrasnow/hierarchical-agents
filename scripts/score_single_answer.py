#!/usr/bin/env python3
"""
CLI script for terminal-based single answer scoring using SingleCriticAgent.

Provides an interactive terminal interface for evaluating answers based on
coverage, detail, and style (not factual correctness).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from critics.single_critic import SingleCriticAgent, score_answer
from critics.models import CriticRequest
from utils.llm import create_llm_client


def print_banner():
    """Print the CLI banner."""
    print("=" * 70)
    print("🎯 Single Answer Critic - Coverage/Detail/Style Evaluator")
    print("=" * 70)
    print("This tool evaluates answers based on PRESENTATION QUALITY")
    print("Focuses on: Coverage • Detail • Structure • Style • Instructions")
    print("Does NOT evaluate: Factual correctness or accuracy")
    print("=" * 70)
    print()


def print_score_summary(score, detailed: bool = False):
    """Print a formatted score summary."""
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Score: {score.overall_score}/100 ({score.overall_tier.upper()})")
    print()
    
    # Dimension breakdown
    print("📋 Dimension Scores:")
    for dim_name, dim_score in score.dimension_scores.items():
        print(f"  {dim_score.dimension_name:25} {dim_score.score:3d}/100 ({dim_score.tier})")
        if detailed:
            print(f"    → {dim_score.justification}")
            print(f"    → Weight: {dim_score.weight}% | Weighted: {dim_score.weighted_score:.1f}")
        print()
    
    # Strengths and weaknesses
    if score.key_strengths:
        print("✅ Key Strengths:")
        for strength in score.key_strengths:
            print(f"  • {strength}")
        print()
    
    if score.key_weaknesses:
        print("⚠️  Areas for Improvement:")
        for weakness in score.key_weaknesses:
            print(f"  • {weakness}")
        print()
    
    if detailed and score.thinking_process:
        print("🤔 Evaluator Reasoning:")
        for i, step in enumerate(score.thinking_process, 1):
            print(f"  {i}. {step}")
        print()
    
    print(f"Overall Assessment: {score.overall_justification}")
    print("=" * 50)


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


async def interactive_mode(llm_provider: str):
    """Run interactive scoring mode."""
    print_banner()
    
    print("🔧 Interactive Scoring Mode")
    print("Enter the question and answer to evaluate.\n")
    
    # Get inputs
    question = get_multiline_input("📝 Enter the original question/prompt")
    print(f"\n✅ Question captured ({len(question)} characters)\n")
    
    answer = get_multiline_input("📄 Enter the answer to evaluate")
    print(f"\n✅ Answer captured ({len(answer)} characters)\n")
    
    context = get_single_input("🏷️  Enter evaluation context (optional)", required=False)
    if not context:
        context = "General evaluation"
    
    instructions = get_single_input("📋 Enter special evaluation instructions (optional)", required=False)
    
    # Perform evaluation
    print(f"\n🔄 Evaluating answer using {llm_provider} provider...")
    print("   Focus: Coverage, Detail, Structure, Style, Instructions")
    print("   Note: Factual correctness is deliberately de-emphasized")
    
    try:
        llm_client = create_llm_client(llm_provider)
        score = await score_answer(
            question=question,
            answer=answer,
            context=context,
            llm_client=llm_client
        )
        
        print_score_summary(score, detailed=True)
        
        # Offer to save results
        save = get_single_input("\n💾 Save results to file? (y/N)", required=False)
        if save.lower().startswith('y'):
            filename = get_single_input("📁 Enter filename (default: scoring_results.json)", required=False)
            if not filename:
                filename = "scoring_results.json"
            
            results = {
                "request": {
                    "question": question,
                    "answer": answer, 
                    "context": context,
                    "evaluation_instructions": instructions
                },
                "score": score.dict()
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Results saved to {filename}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return False
    
    return True


async def file_mode(question_file: str, answer_file: str, output_file: Optional[str], 
                   context: str, llm_provider: str):
    """Run file-based scoring mode."""
    print_banner()
    
    print("📁 File-based Scoring Mode")
    print(f"Question file: {question_file}")
    print(f"Answer file: {answer_file}")
    print(f"Output file: {output_file or 'stdout'}")
    print(f"Context: {context}")
    print(f"Provider: {llm_provider}\n")
    
    # Read input files
    try:
        question = Path(question_file).read_text().strip()
        answer = Path(answer_file).read_text().strip()
    except Exception as e:
        print(f"❌ Failed to read input files: {e}")
        return False
    
    print(f"📝 Question: {len(question)} characters")
    print(f"📄 Answer: {len(answer)} characters\n")
    
    # Perform evaluation
    print("🔄 Evaluating answer...")
    
    try:
        llm_client = create_llm_client(llm_provider)
        score = await score_answer(
            question=question,
            answer=answer,
            context=context,
            llm_client=llm_client
        )
        
        # Output results
        if output_file:
            results = {
                "request": {
                    "question": question,
                    "answer": answer,
                    "context": context
                },
                "score": score.dict()
            }
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Results saved to {output_file}")
            print_score_summary(score)
        else:
            # Print to stdout
            print_score_summary(score, detailed=True)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate answers based on coverage, detail, and style (not factual correctness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/score_single_answer.py

  # File mode  
  python scripts/score_single_answer.py -q question.txt -a answer.txt -o results.json

  # With context
  python scripts/score_single_answer.py -q question.txt -a answer.txt --context "Educational assessment"

  # Different LLM provider
  python scripts/score_single_answer.py --provider claude
        """
    )
    
    # File mode arguments
    parser.add_argument('-q', '--question', help='File containing the question/prompt')
    parser.add_argument('-a', '--answer', help='File containing the answer to evaluate')
    parser.add_argument('-o', '--output', help='Output JSON file (default: stdout)')
    parser.add_argument('--context', default='General evaluation', 
                       help='Evaluation context (default: "General evaluation")')
    
    # General options
    parser.add_argument('--provider', default='mock', 
                       choices=['mock', 'claude', 'openai'],
                       help='LLM provider to use (default: mock)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed evaluation reasoning')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.question and args.answer:
        # File mode
        result = asyncio.run(file_mode(
            args.question, args.answer, args.output, args.context, args.provider
        ))
    else:
        if args.question or args.answer:
            print("❌ Error: Both --question and --answer are required for file mode")
            print("   Use interactive mode by running without arguments")
            sys.exit(1)
        
        # Interactive mode
        result = asyncio.run(interactive_mode(args.provider))
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()