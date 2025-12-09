#!/usr/bin/env python3
"""
Analyze Runs Script - Command-line interface for experiment analysis.

Provides comprehensive analysis of multi-critic evaluation results including
run summaries, performance trends, and challenging question identification.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.tracking import ExperimentTracker, create_tracker
from eval.analysis import PerformanceAnalyzer


def print_json_pretty(data: Dict[str, Any], indent: int = 2):
    """Print JSON data with pretty formatting."""
    print(json.dumps(data, indent=indent, default=str))


def print_summary_table(summary: Dict[str, Any]):
    """Print a formatted summary table."""
    if "error" in summary:
        print(f"Error: {summary['error']}")
        return
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    exp_summary = summary.get("experiment_summary", {})
    print(f"Total Evaluations: {exp_summary.get('total_evaluations', 0)}")
    print(f"Total Runs: {exp_summary.get('total_runs', 0)}")
    print(f"Unique Questions: {exp_summary.get('unique_questions', 0)}")
    print(f"Unique Models: {exp_summary.get('unique_models', 0)}")
    
    date_range = exp_summary.get("date_range", {})
    if date_range.get("earliest") and date_range.get("latest"):
        print(f"Date Range: {date_range['earliest']} to {date_range['latest']}")
    
    print("\n" + "-"*40)
    print("SCORE STATISTICS")
    print("-"*40)
    
    score_stats = summary.get("score_statistics", {})
    print(f"Mean Score: {score_stats.get('mean_score', 0):.1f}")
    print(f"Median Score: {score_stats.get('median_score', 0):.1f}")
    print(f"Standard Deviation: {score_stats.get('std_dev', 0):.1f}")
    print(f"Score Range: {score_stats.get('min_score', 0)} - {score_stats.get('max_score', 0)}")
    
    print("\nScore Distribution:")
    distribution = score_stats.get("score_distribution", {})
    for tier, count in distribution.items():
        percentage = (count / exp_summary.get('total_evaluations', 1)) * 100
        print(f"  {tier}: {count} ({percentage:.1f}%)")
    
    print("\n" + "-"*40)
    print("MODEL PERFORMANCE")
    print("-"*40)
    
    model_performance = summary.get("model_performance", {})
    for model, stats in list(model_performance.items())[:10]:  # Top 10 models
        print(f"{model}: {stats.get('mean_score', 0):.1f} avg ({stats.get('evaluations', 0)} evals)")
    
    print("\n" + "-"*40)
    print("PERFORMANCE INSIGHTS")
    print("-"*40)
    
    insights = summary.get("performance_insights", [])
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print("\n" + "="*60)


def print_run_analysis(analysis: Dict[str, Any]):
    """Print formatted run analysis."""
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    metadata = analysis.get("run_metadata", {})
    summary = analysis.get("summary", {})
    
    print("\n" + "="*60)
    print(f"RUN ANALYSIS: {analysis.get('run_id', 'Unknown')}")
    print("="*60)
    
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"Description: {metadata.get('description', 'No description')}")
    if metadata.get('tags'):
        print(f"Tags: {', '.join(metadata['tags'])}")
    
    print("\n" + "-"*40)
    print("PERFORMANCE SUMMARY")
    print("-"*40)
    
    print(f"Total Questions: {summary.get('total_questions', 0)}")
    print(f"Mean Score: {summary.get('mean_score', 0):.1f}")
    print(f"Median Score: {summary.get('median_score', 0):.1f}")
    print(f"Standard Deviation: {summary.get('std_dev_score', 0):.1f}")
    print(f"Score Range: {summary.get('min_score', 0)} - {summary.get('max_score', 0)}")
    print(f"Mean Confidence: {summary.get('mean_confidence', 0):.2f}")
    print(f"Mean Execution Time: {summary.get('mean_execution_time_ms', 0):.1f} ms")
    
    # Score distribution
    distribution = analysis.get("score_distribution", {}).get("counts", {})
    if distribution:
        print("\nScore Distribution:")
        for tier, count in distribution.items():
            print(f"  {tier}: {count}")
    
    # Challenging questions
    challenging = analysis.get("challenging_questions", [])
    if challenging:
        print(f"\n{'-'*40}")
        print(f"CHALLENGING QUESTIONS ({len(challenging)})")
        print("-"*40)
        for q in challenging[:5]:  # Show top 5
            print(f"  {q['question_id']}: {q['score']}/100 ({q['tier']})")
            print(f"    {q['question_preview']}")
            print()
    
    # High-performing questions
    high_performing = analysis.get("high_performing_questions", [])
    if high_performing:
        print(f"{'-'*40}")
        print(f"HIGH-PERFORMING QUESTIONS ({len(high_performing)})")
        print("-"*40)
        for q in high_performing[:3]:  # Show top 3
            print(f"  {q['question_id']}: {q['score']}/100 ({q['tier']})")
            print(f"    {q['question_preview']}")
            print()


def print_comparison(comparison: Dict[str, Any]):
    """Print formatted run comparison."""
    if "error" in comparison:
        print(f"Error: {comparison['error']}")
        return
    
    run1 = comparison.get("run_1", {})
    run2 = comparison.get("run_2", {})
    score_comp = comparison.get("score_comparison", {})
    perf_comp = comparison.get("performance_comparison", {})
    
    print("\n" + "="*60)
    print("RUN COMPARISON")
    print("="*60)
    
    print(f"Run 1: {run1.get('run_id', 'Unknown')} ({run1.get('model', 'Unknown')})")
    print(f"Run 2: {run2.get('run_id', 'Unknown')} ({run2.get('model', 'Unknown')})")
    
    print("\n" + "-"*40)
    print("SCORE COMPARISON")
    print("-"*40)
    
    mean_diff = score_comp.get("mean_score_diff", 0)
    print(f"Mean Score Difference: {mean_diff:+.1f} (Run 2 vs Run 1)")
    
    median_diff = score_comp.get("median_score_diff", 0)
    print(f"Median Score Difference: {median_diff:+.1f} (Run 2 vs Run 1)")
    
    consistency_diff = score_comp.get("consistency_diff", 0)
    print(f"Consistency Change: {consistency_diff:+.1f} (lower = more consistent)")
    
    print(f"\nQuestion Outcomes:")
    print(f"  Run 2 Better: {score_comp.get('run_2_better_questions', 0)}")
    print(f"  Run 1 Better: {score_comp.get('run_1_better_questions', 0)}")
    print(f"  Tied: {score_comp.get('tied_questions', 0)}")
    
    print("\n" + "-"*40)
    print("PERFORMANCE COMPARISON")
    print("-"*40)
    
    exec_diff = perf_comp.get("execution_time_diff_ms", 0)
    print(f"Execution Time Difference: {exec_diff:+.1f} ms")
    
    conf_diff = perf_comp.get("confidence_diff", 0)
    print(f"Confidence Difference: {conf_diff:+.2f}")
    
    # Significant changes
    improvements = comparison.get("significant_changes", {}).get("improvements", [])
    regressions = comparison.get("significant_changes", {}).get("regressions", [])
    
    if improvements:
        print(f"\n{'-'*40}")
        print(f"SIGNIFICANT IMPROVEMENTS ({len(improvements)})")
        print("-"*40)
        for imp in improvements[:5]:
            print(f"  {imp['question_id']}: {imp['run_1_score']} → {imp['run_2_score']} (+{imp['score_diff']})")
    
    if regressions:
        print(f"\n{'-'*40}")
        print(f"SIGNIFICANT REGRESSIONS ({len(regressions)})")
        print("-"*40)
        for reg in regressions[:5]:
            print(f"  {reg['question_id']}: {reg['run_1_score']} → {reg['run_2_score']} ({reg['score_diff']})")


def print_challenging_questions(questions: List[Dict[str, Any]]):
    """Print challenging questions analysis."""
    if not questions:
        print("No challenging questions found.")
        return
    
    print(f"\n{'='*60}")
    print(f"CHALLENGING QUESTIONS ({len(questions)})")
    print("="*60)
    
    for i, q in enumerate(questions[:10], 1):  # Show top 10
        print(f"{i}. {q['question_id']}")
        print(f"   Mean Score: {q['mean_score']:.1f} (across {q['num_evaluations']} evaluations)")
        print(f"   Models: {', '.join(q['models_evaluated'])}")
        print(f"   Consistent Difficulty: {'Yes' if q['consistent_difficulty'] else 'No'}")
        print(f"   Question: {q['question_text']}")
        print()


def print_trends(trends: Dict[str, Any]):
    """Print model trends analysis."""
    if "error" in trends:
        print(f"Error: {trends['error']}")
        return
    
    model_name = trends.get("model_name", "Unknown")
    period = trends.get("analysis_period", {})
    trend_analysis = trends.get("trend_analysis", {})
    
    print(f"\n{'='*60}")
    print(f"MODEL TRENDS: {model_name}")
    print("="*60)
    
    print(f"Analysis Period: {period.get('start_date', 'Unknown')} to {period.get('end_date', 'Unknown')}")
    print(f"Number of Runs: {period.get('num_runs', 0)}")
    
    print(f"\nScore Trend: {trend_analysis.get('score_trend', 'unknown').title()}")
    print(f"Consistency Trend: {trend_analysis.get('consistency_trend', 'unknown').title()}")
    print(f"Performance Range: {trend_analysis.get('performance_range', 0):.1f} points")
    
    best_run = trend_analysis.get("best_run", {})
    worst_run = trend_analysis.get("worst_run", {})
    
    if best_run:
        print(f"\nBest Run: {best_run.get('run_id', 'Unknown')} ({best_run.get('mean_score', 0):.1f})")
    if worst_run:
        print(f"Worst Run: {worst_run.get('run_id', 'Unknown')} ({worst_run.get('mean_score', 0):.1f})")
    
    # Show recent runs
    recent_runs = trends.get("recent_runs", [])
    if recent_runs:
        print(f"\n{'-'*40}")
        print("RECENT RUNS")
        print("-"*40)
        for run in recent_runs:
            print(f"  {run.get('run_id', 'Unknown')}: {run.get('mean_score', 0):.1f} "
                  f"({run.get('num_questions', 0)} questions, std: {run.get('std_dev', 0):.1f})")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-critic evaluation runs")
    parser.add_argument("--storage-type", choices=["jsonl", "sqlite"], default="jsonl",
                      help="Storage backend type")
    parser.add_argument("--storage-path", default="runs",
                      help="Path to storage directory or database file")
    
    # Analysis commands
    subparsers = parser.add_subparsers(dest="command", help="Analysis commands")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show overall experiment summary")
    summary_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Run analysis command
    run_parser = subparsers.add_parser("run", help="Analyze specific run")
    run_parser.add_argument("run_id", help="Run ID to analyze")
    run_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Compare runs command
    compare_parser = subparsers.add_parser("compare", help="Compare two runs")
    compare_parser.add_argument("run_id_1", help="First run ID")
    compare_parser.add_argument("run_id_2", help="Second run ID")
    compare_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # List runs command
    list_parser = subparsers.add_parser("list", help="List all runs")
    list_parser.add_argument("--model", help="Filter by model name")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of runs to show")
    
    # Challenging questions command
    challenging_parser = subparsers.add_parser("challenging", help="Identify challenging questions")
    challenging_parser.add_argument("--min-results", type=int, default=3,
                                   help="Minimum results required for a question")
    challenging_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Model trends command
    trends_parser = subparsers.add_parser("trends", help="Analyze model performance trends")
    trends_parser.add_argument("model_name", help="Model name to analyze")
    trends_parser.add_argument("--limit", type=int, default=10, help="Number of recent runs to analyze")
    trends_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize tracker
        tracker = create_tracker(storage_type=args.storage_type, storage_path=args.storage_path)
        analyzer = PerformanceAnalyzer(tracker)
        
        if args.command == "summary":
            summary = analyzer.get_summary_statistics()
            if args.json:
                print_json_pretty(summary)
            else:
                print_summary_table(summary)
        
        elif args.command == "run":
            analysis = analyzer.analyze_run_performance(args.run_id)
            if args.json:
                print_json_pretty(analysis)
            else:
                print_run_analysis(analysis)
        
        elif args.command == "compare":
            comparison = analyzer.compare_runs(args.run_id_1, args.run_id_2)
            if args.json:
                print_json_pretty(comparison)
            else:
                print_comparison(comparison)
        
        elif args.command == "list":
            all_runs = tracker.get_all_runs()
            
            if args.model:
                all_runs = [run for run in all_runs if run.model_name == args.model]
            
            all_runs = sorted(all_runs, key=lambda x: x.timestamp, reverse=True)[:args.limit]
            
            print(f"\n{'='*80}")
            print(f"EVALUATION RUNS ({len(all_runs)})")
            print("="*80)
            print(f"{'Run ID':<40} {'Model':<15} {'Timestamp':<20} {'Description'}")
            print("-"*80)
            
            for run in all_runs:
                description = (run.description[:20] + "...") if len(run.description) > 20 else run.description
                print(f"{run.run_id:<40} {run.model_name:<15} {run.timestamp:<20} {description}")
        
        elif args.command == "challenging":
            questions = analyzer.identify_challenging_questions(args.min_results)
            if args.json:
                print_json_pretty(questions)
            else:
                print_challenging_questions(questions)
        
        elif args.command == "trends":
            trends = analyzer.analyze_model_trends(args.model_name, args.limit)
            if args.json:
                print_json_pretty(trends)
            else:
                print_trends(trends)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())