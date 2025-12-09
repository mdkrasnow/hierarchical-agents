#!/usr/bin/env python3
"""
Compare Runs Script - Detailed comparison of multi-critic evaluation runs.

Provides comprehensive comparison capabilities for analyzing differences between
evaluation runs, including statistical comparisons, regression analysis, and
detailed question-by-question breakdowns.
"""

import argparse
import json
import sys
import statistics
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.tracking import ExperimentTracker, create_tracker, ExperimentRecord
from eval.analysis import PerformanceAnalyzer


def generate_comparison_plots(run_1_results: List[ExperimentRecord], 
                            run_2_results: List[ExperimentRecord], 
                            output_dir: str = "comparison_plots"):
    """Generate visual comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping plots.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Align results by question ID
    common_questions = set(r.question_id for r in run_1_results) & set(r.question_id for r in run_2_results)
    
    run_1_scores = {}
    run_2_scores = {}
    
    for result in run_1_results:
        if result.question_id in common_questions:
            run_1_scores[result.question_id] = result.final_score
    
    for result in run_2_results:
        if result.question_id in common_questions:
            run_2_scores[result.question_id] = result.final_score
    
    if not common_questions:
        print("No common questions found for visualization.")
        return
    
    # Score distribution comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    scores_1 = [r.final_score for r in run_1_results]
    scores_2 = [r.final_score for r in run_2_results]
    
    plt.hist(scores_1, alpha=0.7, label='Run 1', bins=20)
    plt.hist(scores_2, alpha=0.7, label='Run 2', bins=20)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution Comparison')
    plt.legend()
    
    # Scatter plot of paired comparisons
    plt.subplot(1, 2, 2)
    aligned_1 = [run_1_scores[qid] for qid in sorted(common_questions)]
    aligned_2 = [run_2_scores[qid] for qid in sorted(common_questions)]
    
    plt.scatter(aligned_1, aligned_2, alpha=0.6)
    plt.plot([0, 100], [0, 100], 'r--', alpha=0.8, label='Equal performance')
    plt.xlabel('Run 1 Score')
    plt.ylabel('Run 2 Score')
    plt.title('Question-by-Question Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "score_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Box plot comparison
    plt.figure(figsize=(8, 6))
    
    data_for_boxplot = [scores_1, scores_2]
    labels = ['Run 1', 'Run 2']
    
    plt.boxplot(data_for_boxplot, labels=labels)
    plt.ylabel('Score')
    plt.title('Score Distribution Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path / "boxplot_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_path}/")


def perform_statistical_tests(run_1_results: List[ExperimentRecord], 
                             run_2_results: List[ExperimentRecord]) -> Dict[str, Any]:
    """Perform statistical significance tests on the comparison."""
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not available. Skipping statistical tests.")
        return {"error": "scipy not available"}
    
    # Align results by question ID
    common_questions = set(r.question_id for r in run_1_results) & set(r.question_id for r in run_2_results)
    
    if len(common_questions) < 3:
        return {"error": "Insufficient common questions for statistical testing"}
    
    run_1_scores = {}
    run_2_scores = {}
    
    for result in run_1_results:
        if result.question_id in common_questions:
            run_1_scores[result.question_id] = result.final_score
    
    for result in run_2_results:
        if result.question_id in common_questions:
            run_2_scores[result.question_id] = result.final_score
    
    # Aligned paired data
    aligned_1 = [run_1_scores[qid] for qid in sorted(common_questions)]
    aligned_2 = [run_2_scores[qid] for qid in sorted(common_questions)]
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(aligned_2, aligned_1)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, w_pvalue = stats.wilcoxon(aligned_2, aligned_1, alternative='two-sided')
    except ValueError:
        w_stat, w_pvalue = None, None
    
    # Effect size (Cohen's d)
    differences = [s2 - s1 for s1, s2 in zip(aligned_1, aligned_2)]
    mean_diff = statistics.mean(differences)
    std_diff = statistics.stdev(differences) if len(differences) > 1 else 0
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    return {
        "sample_size": len(common_questions),
        "paired_t_test": {
            "t_statistic": t_stat,
            "p_value": t_pvalue,
            "significant_at_0.05": t_pvalue < 0.05 if t_pvalue is not None else False
        },
        "wilcoxon_test": {
            "statistic": w_stat,
            "p_value": w_pvalue,
            "significant_at_0.05": w_pvalue < 0.05 if w_pvalue is not None else False
        } if w_stat is not None else {"error": "Unable to perform Wilcoxon test"},
        "effect_size": {
            "cohens_d": cohens_d,
            "interpretation": get_effect_size_interpretation(abs(cohens_d))
        },
        "mean_difference": mean_diff,
        "std_difference": std_diff
    }


def get_effect_size_interpretation(cohens_d: float) -> str:
    """Get interpretation of Cohen's d effect size."""
    if cohens_d < 0.2:
        return "negligible"
    elif cohens_d < 0.5:
        return "small"
    elif cohens_d < 0.8:
        return "medium"
    else:
        return "large"


def analyze_performance_regressions(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance regressions in detail."""
    question_comparisons = comparison.get("question_comparisons", [])
    
    if not question_comparisons:
        return {"error": "No question comparisons available"}
    
    # Categorize changes
    large_improvements = [q for q in question_comparisons if q["score_diff"] >= 15]
    moderate_improvements = [q for q in question_comparisons if 5 <= q["score_diff"] < 15]
    large_regressions = [q for q in question_comparisons if q["score_diff"] <= -15]
    moderate_regressions = [q for q in question_comparisons if -15 < q["score_diff"] <= -5]
    
    # Tier changes
    tier_improvements = [q for q in question_comparisons 
                        if q["run_2_tier"] != q["run_1_tier"] and q["score_diff"] > 0]
    tier_regressions = [q for q in question_comparisons 
                       if q["run_2_tier"] != q["run_1_tier"] and q["score_diff"] < 0]
    
    return {
        "change_categories": {
            "large_improvements": len(large_improvements),
            "moderate_improvements": len(moderate_improvements),
            "large_regressions": len(large_regressions),
            "moderate_regressions": len(moderate_regressions),
            "no_change": len([q for q in question_comparisons if q["score_diff"] == 0])
        },
        "tier_changes": {
            "tier_improvements": len(tier_improvements),
            "tier_regressions": len(tier_regressions)
        },
        "detailed_regressions": {
            "large_regressions": sorted(large_regressions, key=lambda x: x["score_diff"])[:10],
            "large_improvements": sorted(large_improvements, key=lambda x: x["score_diff"], reverse=True)[:10]
        },
        "performance_consistency": {
            "questions_with_same_score": len([q for q in question_comparisons if q["score_diff"] == 0]),
            "questions_within_5_points": len([q for q in question_comparisons if abs(q["score_diff"]) <= 5]),
            "questions_within_10_points": len([q for q in question_comparisons if abs(q["score_diff"]) <= 10])
        }
    }


def generate_detailed_report(tracker: ExperimentTracker, run_id_1: str, run_id_2: str, 
                           output_file: str = None) -> Dict[str, Any]:
    """Generate comprehensive comparison report."""
    analyzer = PerformanceAnalyzer(tracker)
    
    # Get basic comparison
    comparison = analyzer.compare_runs(run_id_1, run_id_2)
    
    if "error" in comparison:
        return comparison
    
    # Get detailed results
    run_1_results = tracker.get_run_results(run_id_1)
    run_2_results = tracker.get_run_results(run_id_2)
    
    # Additional analyses
    regression_analysis = analyze_performance_regressions(comparison)
    statistical_tests = perform_statistical_tests(run_1_results, run_2_results)
    
    # Combine all analyses
    detailed_report = {
        "basic_comparison": comparison,
        "regression_analysis": regression_analysis,
        "statistical_analysis": statistical_tests,
        "raw_data": {
            "run_1_results_count": len(run_1_results),
            "run_2_results_count": len(run_2_results),
            "common_questions": len(set(r.question_id for r in run_1_results) & 
                                 set(r.question_id for r in run_2_results))
        }
    }
    
    # Save report if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        print(f"Detailed report saved to {output_path}")
    
    return detailed_report


def print_statistical_analysis(stats: Dict[str, Any]):
    """Print statistical analysis results."""
    if "error" in stats:
        print(f"Statistical Analysis Error: {stats['error']}")
        return
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    print(f"Sample Size (Common Questions): {stats.get('sample_size', 0)}")
    print(f"Mean Difference: {stats.get('mean_difference', 0):.2f} points")
    
    # Effect size
    effect = stats.get("effect_size", {})
    cohens_d = effect.get("cohens_d", 0)
    interpretation = effect.get("interpretation", "unknown")
    print(f"Effect Size (Cohen's d): {cohens_d:.3f} ({interpretation})")
    
    # T-test results
    t_test = stats.get("paired_t_test", {})
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_test.get('t_statistic', 0):.3f}")
    print(f"  p-value: {t_test.get('p_value', 1):.6f}")
    print(f"  Significant at α=0.05: {t_test.get('significant_at_0.05', False)}")
    
    # Wilcoxon test results
    wilcoxon = stats.get("wilcoxon_test", {})
    if "error" not in wilcoxon:
        print(f"\nWilcoxon signed-rank test:")
        print(f"  Statistic: {wilcoxon.get('statistic', 0)}")
        print(f"  p-value: {wilcoxon.get('p_value', 1):.6f}")
        print(f"  Significant at α=0.05: {wilcoxon.get('significant_at_0.05', False)}")


def print_comparison_basic(comparison: Dict[str, Any]):
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


def print_regression_analysis(regression: Dict[str, Any]):
    """Print detailed regression analysis."""
    if "error" in regression:
        print(f"Regression Analysis Error: {regression['error']}")
        return
    
    print("\n" + "="*60)
    print("PERFORMANCE CHANGE ANALYSIS")
    print("="*60)
    
    categories = regression.get("change_categories", {})
    print(f"Large Improvements (≥15 pts): {categories.get('large_improvements', 0)}")
    print(f"Moderate Improvements (5-14 pts): {categories.get('moderate_improvements', 0)}")
    print(f"Large Regressions (≤-15 pts): {categories.get('large_regressions', 0)}")
    print(f"Moderate Regressions (-14 to -5 pts): {categories.get('moderate_regressions', 0)}")
    print(f"No Change: {categories.get('no_change', 0)}")
    
    tier_changes = regression.get("tier_changes", {})
    print(f"\nTier Changes:")
    print(f"  Tier Improvements: {tier_changes.get('tier_improvements', 0)}")
    print(f"  Tier Regressions: {tier_changes.get('tier_regressions', 0)}")
    
    # Show worst regressions
    detailed = regression.get("detailed_regressions", {})
    large_regressions = detailed.get("large_regressions", [])
    large_improvements = detailed.get("large_improvements", [])
    
    if large_regressions:
        print(f"\n{'-'*40}")
        print(f"WORST REGRESSIONS ({len(large_regressions)})")
        print("-"*40)
        for reg in large_regressions[:5]:
            print(f"  {reg['question_id']}: {reg['run_1_score']} → {reg['run_2_score']} ({reg['score_diff']:+d})")
    
    if large_improvements:
        print(f"\n{'-'*40}")
        print(f"BEST IMPROVEMENTS ({len(large_improvements)})")
        print("-"*40)
        for imp in large_improvements[:5]:
            print(f"  {imp['question_id']}: {imp['run_1_score']} → {imp['run_2_score']} ({imp['score_diff']:+d})")


def main():
    parser = argparse.ArgumentParser(description="Compare multi-critic evaluation runs")
    parser.add_argument("run_id_1", help="First run ID to compare")
    parser.add_argument("run_id_2", help="Second run ID to compare")
    parser.add_argument("--storage-type", choices=["jsonl", "sqlite"], default="jsonl",
                      help="Storage backend type")
    parser.add_argument("--storage-path", default="runs",
                      help="Path to storage directory or database file")
    
    # Output options
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--detailed", action="store_true", help="Include detailed statistical analysis")
    parser.add_argument("--plots", action="store_true", help="Generate comparison plots")
    parser.add_argument("--output-dir", default="comparison_output",
                      help="Directory for output files")
    parser.add_argument("--report-file", help="Save detailed report to JSON file")
    
    # Analysis options
    parser.add_argument("--stats", action="store_true", help="Include statistical significance tests")
    parser.add_argument("--regressions", action="store_true", help="Detailed regression analysis")
    
    args = parser.parse_args()
    
    try:
        # Initialize tracker
        tracker = create_tracker(storage_type=args.storage_type, storage_path=args.storage_path)
        
        if args.detailed or args.report_file:
            # Generate comprehensive report
            report = generate_detailed_report(tracker, args.run_id_1, args.run_id_2, args.report_file)
            
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                # Print basic comparison
                basic_comparison = report.get("basic_comparison", {})
                if "error" not in basic_comparison:
                    print_comparison_basic(basic_comparison)
                
                # Print additional analyses
                if args.stats or args.detailed:
                    print_statistical_analysis(report.get("statistical_analysis", {}))
                
                if args.regressions or args.detailed:
                    print_regression_analysis(report.get("regression_analysis", {}))
        
        else:
            # Basic comparison
            analyzer = PerformanceAnalyzer(tracker)
            comparison = analyzer.compare_runs(args.run_id_1, args.run_id_2)
            
            if args.json:
                print(json.dumps(comparison, indent=2, default=str))
            else:
                print_comparison_basic(comparison)
                
                if args.stats:
                    run_1_results = tracker.get_run_results(args.run_id_1)
                    run_2_results = tracker.get_run_results(args.run_id_2)
                    stats = perform_statistical_tests(run_1_results, run_2_results)
                    print_statistical_analysis(stats)
                
                if args.regressions:
                    regression_analysis = analyze_performance_regressions(comparison)
                    print_regression_analysis(regression_analysis)
        
        # Generate plots if requested
        if args.plots:
            run_1_results = tracker.get_run_results(args.run_id_1)
            run_2_results = tracker.get_run_results(args.run_id_2)
            generate_comparison_plots(run_1_results, run_2_results, args.output_dir)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())