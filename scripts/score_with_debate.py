#!/usr/bin/env python3
"""
CLI script for multi-critic debate evaluation system.

Provides comprehensive evaluation using multiple specialized critics with
debate rounds and intelligent aggregation.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from critics.orchestrator import MultiCriticOrchestrator
from critics.multi_critic import CriticFactory
from critics.debate_models import MultiCriticRequest, MultiCriticResult
from utils.llm import create_llm_client


def print_banner():
    """Print the CLI banner."""
    print("=" * 80)
    print("🎪 Multi-Critic Debate System - Comprehensive Answer Evaluation")
    print("=" * 80)
    print("Specialized Critics: Coverage • Depth • Style • Instruction-Following")
    print("Process: Independent Evaluation → Debate & Aggregation → Final Synthesis")
    print("Focus: Presentation Quality (NOT factual correctness)")
    print("=" * 80)
    print()


def print_critic_breakdown(result: MultiCriticResult, show_individual: bool = False):
    """Print detailed breakdown of critic evaluations."""
    print("\n" + "=" * 60)
    print("📊 MULTI-CRITIC EVALUATION RESULTS")
    print("=" * 60)
    
    # Final score
    print(f"🎯 FINAL SCORE: {result.final_score}/100 ({result.final_tier.upper()})")
    print(f"📈 Confidence Level: {result.confidence_level:.2f}")
    print(f"⏱️  Total Evaluation Time: {result.total_execution_time_ms:.1f}ms")
    print()
    
    # Individual critic scores
    print("🏛️  CRITIC PANEL SCORES:")
    for critic_role, score in result.individual_critic_scores.items():
        # Get the critic result for additional details
        critic_result = result.get_critic_result(critic_role)
        if critic_result:
            print(f"  {critic_result.critic_role.title():20} {score:3d}/100 "
                  f"({critic_result.critic_score.overall_tier})")
        else:
            print(f"  {critic_role.title():20} {score:3d}/100")
    print()
    
    # Consensus analysis
    aggregation = result.final_aggregation
    print(f"🤝 CONSENSUS ANALYSIS:")
    print(f"  Score Variance: {aggregation.score_variance:.1f}")
    print(f"  Consensus Level: {aggregation.consensus_level.upper()}")
    print(f"  Aggregation Method: {aggregation.aggregation_method}")
    print()
    
    # Key insights
    if aggregation.comprehensive_strengths:
        print("✅ COMPREHENSIVE STRENGTHS:")
        for strength in aggregation.comprehensive_strengths[:5]:  # Limit to top 5
            print(f"  • {strength}")
        print()
    
    if aggregation.comprehensive_weaknesses:
        print("⚠️  AREAS FOR IMPROVEMENT:")
        for weakness in aggregation.comprehensive_weaknesses[:5]:  # Limit to top 5
            print(f"  • {weakness}")
        print()
    
    # Recommendations
    if aggregation.actionable_recommendations:
        print("💡 ACTIONABLE RECOMMENDATIONS:")
        for rec in aggregation.actionable_recommendations[:5]:  # Limit to top 5
            print(f"  • {rec}")
        print()
    
    # Aggregation reasoning
    print("🧠 SYNTHESIS REASONING:")
    print(f"  {aggregation.aggregation_reasoning}")
    print()
    
    # Individual critic details (if requested)
    if show_individual:
        print("\n" + "=" * 60)
        print("👥 INDIVIDUAL CRITIC DETAILS")
        print("=" * 60)
        
        for round_data in result.debate_rounds:
            if round_data.round_type == "independent":
                for critic_result in round_data.results:
                    print(f"\n--- {critic_result.critic_role.upper()} CRITIC ---")
                    score = critic_result.critic_score
                    print(f"Score: {score.overall_score}/100 ({score.overall_tier})")
                    print(f"Focus: {critic_result.focus_summary}")
                    print(f"Confidence: {critic_result.confidence:.2f}")
                    print(f"Time: {critic_result.execution_time_ms:.1f}ms")
                    print(f"\nJustification: {score.overall_justification}")
                    
                    if score.key_strengths:
                        print(f"Strengths: {'; '.join(score.key_strengths)}")
                    if score.key_weaknesses:
                        print(f"Weaknesses: {'; '.join(score.key_weaknesses)}")
                    
                    if critic_result.notable_observations:
                        print(f"Notable: {'; '.join(critic_result.notable_observations)}")
                break
    
    print("=" * 60)


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


def select_critics() -> Optional[List[str]]:
    """Interactive critic selection."""
    available_critics = CriticFactory.get_available_roles()
    
    print("\n🎭 Available Critics:")
    print("  1. All Critics (recommended)")
    for i, critic in enumerate(available_critics, 2):
        print(f"  {i}. {critic.title()} Critic only")
    print(f"  {len(available_critics) + 2}. Custom selection")
    
    while True:
        try:
            choice = input("\nSelect critics (1): ").strip()
            if not choice:
                choice = "1"
            
            choice_num = int(choice)
            
            if choice_num == 1:
                return None  # Use all critics
            elif 2 <= choice_num <= len(available_critics) + 1:
                # Single critic
                critic_index = choice_num - 2
                return [available_critics[critic_index]]
            elif choice_num == len(available_critics) + 2:
                # Custom selection
                print("\nSelect critics (enter numbers separated by spaces):")
                for i, critic in enumerate(available_critics, 1):
                    print(f"  {i}. {critic.title()} Critic")
                
                selection = input("Critics: ").strip()
                if not selection:
                    continue
                
                try:
                    indices = [int(x) - 1 for x in selection.split()]
                    if all(0 <= i < len(available_critics) for i in indices):
                        return [available_critics[i] for i in indices]
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter numbers separated by spaces.")
            else:
                print("Invalid choice. Please try again.")
                
        except ValueError:
            print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled.")
            sys.exit(0)


async def interactive_mode(llm_provider: str):
    """Run interactive multi-critic evaluation."""
    print_banner()
    
    print("🔧 Interactive Multi-Critic Evaluation")
    print("Enter the question and answer for comprehensive evaluation.\n")
    
    # Get inputs
    question = get_multiline_input("📝 Enter the original question/prompt")
    print(f"\n✅ Question captured ({len(question)} characters)\n")
    
    answer = get_multiline_input("📄 Enter the answer to evaluate")
    print(f"\n✅ Answer captured ({len(answer)} characters)\n")
    
    context = get_single_input("🏷️  Enter evaluation context (optional)", required=False)
    if not context:
        context = "General multi-critic evaluation"
    
    instructions = get_single_input("📋 Enter special evaluation instructions (optional)", required=False)
    
    # Critic selection
    critic_roles = select_critics()
    if critic_roles:
        print(f"\n🎭 Selected Critics: {', '.join(critic_roles)}")
    else:
        print("\n🎭 Using All Available Critics")
    
    # Options
    show_individual = get_single_input("\n🔍 Show individual critic details? (y/N)", required=False)
    show_individual = show_individual.lower().startswith('y')
    
    enable_debate = get_single_input("💬 Enable debate/aggregation round? (Y/n)", required=False)
    enable_debate = not enable_debate.lower().startswith('n')
    
    # Perform evaluation
    print(f"\n🔄 Running multi-critic evaluation with {llm_provider} provider...")
    print("   Phase 1: Independent critic evaluations (parallel)")
    if enable_debate:
        print("   Phase 2: Debate and aggregation synthesis")
    print("   Note: Factual correctness is deliberately de-emphasized")
    print()
    
    try:
        llm_client = create_llm_client(llm_provider)
        orchestrator = MultiCriticOrchestrator(llm_client)
        
        request = MultiCriticRequest(
            question=question,
            answer=answer,
            context=context,
            evaluation_instructions=instructions,
            critic_roles=critic_roles,
            enable_debate=enable_debate,
            show_individual_results=show_individual
        )
        
        result = await orchestrator.evaluate(request)
        
        # Display results
        print_critic_breakdown(result, show_individual=show_individual)
        
        # Offer to save results
        save = get_single_input("\n💾 Save results to file? (y/N)", required=False)
        if save.lower().startswith('y'):
            filename = get_single_input("📁 Enter filename (default: multi_critic_results.json)", required=False)
            if not filename:
                filename = "multi_critic_results.json"
            
            # Save full results
            with open(filename, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
            
            print(f"✅ Complete results saved to {filename}")
        
    except Exception as e:
        print(f"\n❌ Multi-critic evaluation failed: {e}")
        return False
    
    return True


async def file_mode(question_file: str, answer_file: str, output_file: Optional[str],
                   context: str, critics: Optional[List[str]], enable_debate: bool,
                   show_individual: bool, llm_provider: str):
    """Run file-based multi-critic evaluation."""
    print_banner()
    
    print("📁 File-based Multi-Critic Evaluation")
    print(f"Question file: {question_file}")
    print(f"Answer file: {answer_file}")
    print(f"Output file: {output_file or 'stdout'}")
    print(f"Context: {context}")
    print(f"Critics: {', '.join(critics) if critics else 'All'}")
    print(f"Debate enabled: {enable_debate}")
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
    print("🔄 Running multi-critic evaluation...")
    
    try:
        llm_client = create_llm_client(llm_provider)
        orchestrator = MultiCriticOrchestrator(llm_client)
        
        request = MultiCriticRequest(
            question=question,
            answer=answer,
            context=context,
            critic_roles=critics,
            enable_debate=enable_debate,
            show_individual_results=show_individual
        )
        
        result = await orchestrator.evaluate(request)
        
        # Output results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
            
            print(f"✅ Complete results saved to {output_file}")
            print_critic_breakdown(result, show_individual=False)  # Summary to console
        else:
            # Print to stdout
            print_critic_breakdown(result, show_individual=show_individual)
        
    except Exception as e:
        print(f"❌ Multi-critic evaluation failed: {e}")
        return False
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-critic debate evaluation system for comprehensive answer assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with all critics
  python scripts/score_with_debate.py

  # File mode with all critics
  python scripts/score_with_debate.py -q question.txt -a answer.txt -o results.json

  # Specific critics only
  python scripts/score_with_debate.py -q question.txt -a answer.txt --critics coverage depth

  # Disable debate round (independent critics only)
  python scripts/score_with_debate.py -q question.txt -a answer.txt --no-debate

  # Show individual critic details
  python scripts/score_with_debate.py -q question.txt -a answer.txt --show-critics

  # Different LLM provider
  python scripts/score_with_debate.py --provider claude
        """
    )
    
    # File mode arguments
    parser.add_argument('-q', '--question', help='File containing the question/prompt')
    parser.add_argument('-a', '--answer', help='File containing the answer to evaluate')
    parser.add_argument('-o', '--output', help='Output JSON file (default: stdout)')
    parser.add_argument('--context', default='Multi-critic evaluation', 
                       help='Evaluation context (default: "Multi-critic evaluation")')
    
    # Critic selection
    available_critics = CriticFactory.get_available_roles()
    parser.add_argument('--critics', nargs='+', choices=available_critics,
                       help=f'Specific critics to use (default: all). Options: {", ".join(available_critics)}')
    
    # Debate options
    parser.add_argument('--no-debate', action='store_true',
                       help='Disable debate/aggregation round (independent critics only)')
    parser.add_argument('--show-critics', action='store_true',
                       help='Show detailed individual critic results')
    
    # General options
    parser.add_argument('--provider', default=None, 
                       choices=['gemini', 'claude'],
                       help='LLM provider to use (auto-selects based on available API keys)')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.question and args.answer:
        # File mode
        result = asyncio.run(file_mode(
            args.question, args.answer, args.output, args.context,
            args.critics, not args.no_debate, args.show_critics, args.provider
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