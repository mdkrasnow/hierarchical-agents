#!/usr/bin/env python3
"""
Dedicated script for computing alignment metrics between scoring systems.

This script provides focused functionality for analyzing agreement between:
- Human evaluators vs automated scoring
- Single-critic vs multi-critic systems  
- Different scoring configurations or models

Outputs detailed statistical analysis, visualizations, and recommendations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.calibration import (
    HumanScore, AutomatedScore, ScoreAlignmentAnalyzer, 
    load_human_scores, AlignmentMetrics
)


def print_banner():
    """Print CLI banner."""
    print("=" * 80)
    print("📊 Score Alignment Analysis & Visualization")
    print("=" * 80)
    print("Purpose: Deep statistical analysis of scoring system alignment")
    print("Features: Correlations • Agreement • Bias Detection • Visualizations")
    print("Output: Statistical reports, charts, and actionable insights")
    print("=" * 80)
    print()


def load_automated_scores(filepath: str) -> List[AutomatedScore]:
    """Load automated scores from JSONL file."""
    scores = []
    
    with open(filepath, 'r') as f:
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
                scores.append(score)
    
    return scores


def create_alignment_visualizations(human_scores: List[HumanScore], 
                                   automated_scores: List[AutomatedScore],
                                   output_dir: str):
    """Create comprehensive visualizations of score alignment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Align scores for plotting
    analyzer = ScoreAlignmentAnalyzer()
    aligned_pairs = analyzer._align_scores(human_scores, automated_scores)
    
    if not aligned_pairs:
        print("⚠️  No aligned scores found for visualization")
        return
    
    # Extract data for plotting
    human_overall = [pair[0].overall_score for pair in aligned_pairs]
    auto_overall = [pair[1].overall_score for pair in aligned_pairs]
    question_ids = [pair[0].question_id for pair in aligned_pairs]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'human_score': human_overall,
        'automated_score': auto_overall,
        'question_id': question_ids,
        'difference': np.array(auto_overall) - np.array(human_overall)
    })
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Scatter plot with correlation
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot
    axes[0, 0].scatter(human_overall, auto_overall, alpha=0.7, s=50)
    axes[0, 0].plot([0, 100], [0, 100], 'r--', alpha=0.8, label='Perfect Agreement')
    
    # Add correlation info
    r_pearson, p_pearson = pearsonr(human_overall, auto_overall)
    r_spearman, p_spearman = spearmanr(human_overall, auto_overall)
    
    axes[0, 0].set_xlabel('Human Scores')
    axes[0, 0].set_ylabel('Automated Scores')
    axes[0, 0].set_title(f'Score Alignment\nPearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Difference distribution
    axes[0, 1].hist(df['difference'], bins=15, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(df['difference']), color='red', linestyle='--', 
                       label=f'Mean Bias: {np.mean(df["difference"]):.2f}')
    axes[0, 1].axvline(0, color='green', linestyle='-', alpha=0.5, label='No Bias')
    axes[0, 1].set_xlabel('Score Difference (Automated - Human)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Bias Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bland-Altman plot
    mean_scores = (np.array(human_overall) + np.array(auto_overall)) / 2
    diff_scores = np.array(auto_overall) - np.array(human_overall)
    
    axes[1, 0].scatter(mean_scores, diff_scores, alpha=0.7, s=50)
    axes[1, 0].axhline(np.mean(diff_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(diff_scores):.2f}')
    axes[1, 0].axhline(np.mean(diff_scores) + 1.96*np.std(diff_scores), 
                       color='red', linestyle=':', alpha=0.7, label='95% Limits')
    axes[1, 0].axhline(np.mean(diff_scores) - 1.96*np.std(diff_scores), 
                       color='red', linestyle=':', alpha=0.7)
    axes[1, 0].axhline(0, color='green', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('Mean Score (Human + Automated)/2')
    axes[1, 0].set_ylabel('Score Difference (Automated - Human)')
    axes[1, 0].set_title('Bland-Altman Plot')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Agreement categories
    agreement_5 = np.abs(df['difference']) <= 5
    agreement_10 = np.abs(df['difference']) <= 10
    agreement_15 = np.abs(df['difference']) <= 15
    
    categories = ['±5 pts', '±10 pts', '±15 pts', '>15 pts']
    counts = [
        np.sum(agreement_5),
        np.sum(agreement_10) - np.sum(agreement_5),
        np.sum(agreement_15) - np.sum(agreement_10),
        len(df) - np.sum(agreement_15)
    ]
    
    colors = ['green', 'yellowgreen', 'orange', 'red']
    axes[1, 1].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
    axes[1, 1].set_title('Agreement Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path / 'score_alignment_overview.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved alignment overview: {output_path / 'score_alignment_overview.png'}")
    plt.close()
    
    # 5. Dimension-wise analysis (if available)
    dimension_correlations = create_dimension_analysis_plot(aligned_pairs, output_path)
    
    # 6. Individual question analysis
    create_question_level_analysis(df, output_path)
    
    print(f"📈 All visualizations saved to: {output_path}")


def create_dimension_analysis_plot(aligned_pairs: List[Tuple], output_path: Path):
    """Create dimension-wise correlation analysis."""
    # Collect dimension data
    all_dimensions = set()
    for human_score, auto_score in aligned_pairs:
        all_dimensions.update(human_score.dimension_scores.keys())
        all_dimensions.update(auto_score.dimension_scores.keys())
    
    if not all_dimensions:
        print("⚠️  No dimension data available for visualization")
        return
    
    dimension_data = {}
    for dimension in all_dimensions:
        human_dim = []
        auto_dim = []
        
        for human_score, auto_score in aligned_pairs:
            if (dimension in human_score.dimension_scores and 
                dimension in auto_score.dimension_scores):
                human_dim.append(human_score.dimension_scores[dimension])
                auto_dim.append(auto_score.dimension_scores[dimension])
        
        if len(human_dim) >= 2:
            r, p = pearsonr(human_dim, auto_dim)
            dimension_data[dimension] = {
                'correlation': r,
                'p_value': p,
                'n_samples': len(human_dim),
                'human_scores': human_dim,
                'auto_scores': auto_dim
            }
    
    if dimension_data:
        # Create dimension correlation plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of correlations
        dims = list(dimension_data.keys())
        corrs = [dimension_data[d]['correlation'] for d in dims]
        colors = ['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' for c in corrs]
        
        bars = axes[0].bar(range(len(dims)), corrs, color=colors, alpha=0.7)
        axes[0].set_xticks(range(len(dims)))
        axes[0].set_xticklabels(dims, rotation=45, ha='right')
        axes[0].set_ylabel('Pearson Correlation')
        axes[0].set_title('Dimension-wise Correlations')
        axes[0].axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Good (≥0.7)')
        axes[0].axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (≥0.5)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Add correlation values on bars
        for bar, corr in zip(bars, corrs):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Heatmap of dimension scores
        if len(dims) > 1:
            human_matrix = []
            auto_matrix = []
            
            for human_score, auto_score in aligned_pairs:
                human_row = []
                auto_row = []
                for dim in dims:
                    if (dim in human_score.dimension_scores and 
                        dim in auto_score.dimension_scores):
                        human_row.append(human_score.dimension_scores[dim])
                        auto_row.append(auto_score.dimension_scores[dim])
                    else:
                        human_row.append(np.nan)
                        auto_row.append(np.nan)
                human_matrix.append(human_row)
                auto_matrix.append(auto_row)
            
            # Calculate correlation matrix between dimensions
            df_dims = pd.DataFrame(human_matrix, columns=dims)
            corr_matrix = df_dims.corr()
            
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, ax=axes[1], square=True)
            axes[1].set_title('Inter-dimension Correlations (Human)')
        else:
            axes[1].text(0.5, 0.5, 'Need >1 dimension\nfor correlation matrix', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Inter-dimension Analysis')
        
        plt.tight_layout()
        plt.savefig(output_path / 'dimension_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 Saved dimension analysis: {output_path / 'dimension_analysis.png'}")
        plt.close()


def create_question_level_analysis(df: pd.DataFrame, output_path: Path):
    """Create question-level analysis plots."""
    # Sort by absolute difference for easier visualization
    df_sorted = df.reindex(df['difference'].abs().sort_values(ascending=False).index)
    
    # Create question-level plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top disagreements
    n_top = min(10, len(df_sorted))
    top_disagreements = df_sorted.head(n_top)
    
    x_pos = range(n_top)
    axes[0].bar(x_pos, top_disagreements['human_score'], alpha=0.7, label='Human', width=0.4)
    axes[0].bar([x + 0.4 for x in x_pos], top_disagreements['automated_score'], 
               alpha=0.7, label='Automated', width=0.4)
    
    axes[0].set_xticks([x + 0.2 for x in x_pos])
    axes[0].set_xticklabels(top_disagreements['question_id'], rotation=45, ha='right')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Top {n_top} Score Disagreements')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Score differences by question
    axes[1].bar(range(len(df_sorted)), df_sorted['difference'], 
               color=['red' if x < 0 else 'blue' for x in df_sorted['difference']], alpha=0.7)
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Questions (sorted by disagreement)')
    axes[1].set_ylabel('Score Difference (Automated - Human)')
    axes[1].set_title('Score Differences Across All Questions')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'question_level_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved question analysis: {output_path / 'question_level_analysis.png'}")
    plt.close()


def generate_detailed_report(metrics: AlignmentMetrics, human_scores: List[HumanScore], 
                            automated_scores: List[AutomatedScore], output_file: str):
    """Generate a detailed statistical report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("DETAILED SCORE ALIGNMENT ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    
    # Determine overall assessment
    if metrics.overall_pearson > 0.8 and metrics.agreement_percentage_10 > 80:
        assessment = "EXCELLENT alignment"
    elif metrics.overall_pearson > 0.6 and metrics.agreement_percentage_10 > 70:
        assessment = "GOOD alignment"
    elif metrics.overall_pearson > 0.4 and metrics.agreement_percentage_10 > 60:
        assessment = "MODERATE alignment"
    else:
        assessment = "POOR alignment"
    
    report_lines.append(f"Overall Assessment: {assessment}")
    report_lines.append(f"Sample Size: {metrics.n_samples} paired evaluations")
    report_lines.append(f"Primary Correlation: r = {metrics.overall_pearson:.3f}")
    report_lines.append(f"Agreement within 10 points: {metrics.agreement_percentage_10:.1f}%")
    
    if metrics.systematic_bias_detected:
        bias_dir = "higher" if metrics.mean_bias_error > 0 else "lower"
        report_lines.append(f"⚠️  Systematic bias detected: Automated scores {bias_dir}")
    else:
        report_lines.append("✓ No significant systematic bias")
    
    report_lines.append("")
    
    # Statistical Details
    report_lines.append("DETAILED STATISTICAL ANALYSIS")
    report_lines.append("-" * 40)
    
    report_lines.append("Correlation Analysis:")
    report_lines.append(f"  Pearson correlation: r = {metrics.overall_pearson:.4f}")
    report_lines.append(f"    Significance: p = {metrics.overall_pearson_p_value:.4f}")
    report_lines.append(f"    Interpretation: {'Significant' if metrics.overall_pearson_p_value < 0.05 else 'Not significant'}")
    report_lines.append(f"  Spearman correlation: ρ = {metrics.overall_spearman:.4f}")
    report_lines.append(f"    Significance: p = {metrics.overall_spearman_p_value:.4f}")
    report_lines.append("")
    
    report_lines.append("Error Analysis:")
    report_lines.append(f"  Mean Absolute Error: {metrics.mean_absolute_error:.2f} points")
    report_lines.append(f"  Root Mean Square Error: {metrics.root_mean_square_error:.2f} points")
    report_lines.append(f"  Mean Bias Error: {metrics.mean_bias_error:+.2f} points")
    report_lines.append(f"  Outliers detected: {metrics.outlier_count} ({metrics.outlier_count/metrics.n_samples*100:.1f}%)")
    report_lines.append("")
    
    report_lines.append("Agreement Analysis:")
    report_lines.append(f"  Exact agreement (same score): {metrics.exact_agreement_count}")
    report_lines.append(f"  Within ±5 points: {metrics.within_5_points_count} ({metrics.agreement_percentage_5:.1f}%)")
    report_lines.append(f"  Within ±10 points: {metrics.within_10_points_count} ({metrics.agreement_percentage_10:.1f}%)")
    report_lines.append("")
    
    # Distribution comparison
    report_lines.append("Score Distribution Comparison:")
    h_stats = metrics.human_score_stats
    a_stats = metrics.automated_score_stats
    
    report_lines.append(f"  Human scores:     μ={h_stats['mean']:.1f}, σ={h_stats['std_dev']:.1f}, range=[{h_stats['min']}-{h_stats['max']}]")
    report_lines.append(f"  Automated scores: μ={a_stats['mean']:.1f}, σ={a_stats['std_dev']:.1f}, range=[{a_stats['min']}-{a_stats['max']}]")
    report_lines.append(f"  Mean difference: {a_stats['mean'] - h_stats['mean']:+.1f} points")
    report_lines.append("")
    
    # Dimension analysis
    if metrics.dimension_correlations:
        report_lines.append("Dimension-wise Analysis:")
        for dimension, corr_data in sorted(metrics.dimension_correlations.items()):
            quality = ("Excellent" if corr_data['pearson'] > 0.8 else
                      "Good" if corr_data['pearson'] > 0.6 else
                      "Fair" if corr_data['pearson'] > 0.4 else "Poor")
            
            report_lines.append(f"  {dimension:25s}: r={corr_data['pearson']:6.3f} ({quality}, n={corr_data['n_samples']})")
        report_lines.append("")
    
    # Human evaluator analysis
    evaluator_stats = {}
    for score in human_scores:
        eid = score.evaluator_id
        if eid not in evaluator_stats:
            evaluator_stats[eid] = {
                'scores': [],
                'confidences': [],
                'eval_times': []
            }
        evaluator_stats[eid]['scores'].append(score.overall_score)
        evaluator_stats[eid]['confidences'].append(score.confidence)
        if score.time_spent_minutes:
            evaluator_stats[eid]['eval_times'].append(score.time_spent_minutes)
    
    if len(evaluator_stats) > 1:
        report_lines.append("Human Evaluator Analysis:")
        for evaluator, stats in evaluator_stats.items():
            mean_score = np.mean(stats['scores'])
            mean_conf = np.mean(stats['confidences'])
            mean_time = np.mean(stats['eval_times']) if stats['eval_times'] else None
            
            report_lines.append(f"  {evaluator}: μ_score={mean_score:.1f}, μ_confidence={mean_conf:.2f}" +
                              (f", μ_time={mean_time:.1f}min" if mean_time else ""))
        report_lines.append("")
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Detailed report saved: {output_file}")
    
    # Also print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"🎯 Assessment: {assessment}")
    print(f"📊 Correlation: r = {metrics.overall_pearson:.3f}")
    print(f"🎯 Agreement: {metrics.agreement_percentage_10:.1f}% within ±10 points")
    if metrics.systematic_bias_detected:
        bias_dir = "higher" if metrics.mean_bias_error > 0 else "lower"
        print(f"⚠️  Bias: Automated scores {metrics.mean_bias_error:+.1f} points {bias_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute detailed alignment metrics between scoring systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic alignment analysis
  python scripts/compute_alignment.py --human data/human_labels.jsonl --automated data/automated_scores.jsonl

  # With full visualizations and detailed report
  python scripts/compute_alignment.py --human data/human_labels.jsonl --automated data/automated_scores.jsonl --visualize --report

  # Compare two automated methods
  python scripts/compute_alignment.py --automated1 data/single_critic_scores.jsonl --automated2 data/multi_critic_scores.jsonl

  # Custom output location
  python scripts/compute_alignment.py --human data/human_labels.jsonl --automated data/automated_scores.jsonl --output alignment_analysis
        """
    )
    
    # Input files
    parser.add_argument('--human', help='Human scores file (JSONL format)')
    parser.add_argument('--automated', help='Automated scores file (JSONL format)')
    parser.add_argument('--automated1', help='First automated scores file (for method comparison)')
    parser.add_argument('--automated2', help='Second automated scores file (for method comparison)')
    
    # Output options
    parser.add_argument('--output', default='data/alignment_analysis',
                       help='Output directory/prefix for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed statistical report')
    
    # Analysis options
    parser.add_argument('--json-output', action='store_true',
                       help='Save metrics as JSON file')
    parser.add_argument('--include-dimensions', action='store_true', default=True,
                       help='Include dimension-wise analysis')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.human and args.automated:
        mode = "human_vs_automated"
    elif args.automated1 and args.automated2:
        mode = "automated_comparison"
    else:
        parser.print_help()
        print("\nError: Must provide either (--human AND --automated) OR (--automated1 AND --automated2)")
        sys.exit(1)
    
    print_banner()
    
    if mode == "human_vs_automated":
        # Load scores
        print("📁 Loading score files...")
        human_scores = load_human_scores(args.human)
        automated_scores = load_automated_scores(args.automated)
        
        print(f"👥 Human scores: {len(human_scores)}")
        print(f"🤖 Automated scores: {len(automated_scores)}")
        
        # Compute alignment
        print("\n🔄 Computing alignment metrics...")
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, automated_scores)
        
        # Display results
        print(f"\n📊 ALIGNMENT RESULTS")
        print(f"Correlation (Pearson): {metrics.overall_pearson:.3f}")
        print(f"Agreement (±10 points): {metrics.agreement_percentage_10:.1f}%")
        print(f"Mean Absolute Error: {metrics.mean_absolute_error:.2f}")
        print(f"Systematic Bias: {metrics.systematic_bias_detected}")
        
        # Generate outputs
        if args.json_output:
            json_file = f"{args.output}_metrics.json"
            metrics_dict = {
                'overall_pearson': metrics.overall_pearson,
                'overall_spearman': metrics.overall_spearman,
                'agreement_percentage_10': metrics.agreement_percentage_10,
                'mean_absolute_error': metrics.mean_absolute_error,
                'mean_bias_error': metrics.mean_bias_error,
                'systematic_bias_detected': metrics.systematic_bias_detected,
                'n_samples': metrics.n_samples
            }
            with open(json_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"💾 Metrics saved: {json_file}")
        
        if args.visualize:
            print("\n🎨 Generating visualizations...")
            create_alignment_visualizations(human_scores, automated_scores, f"{args.output}_plots")
        
        if args.report:
            print("\n📄 Generating detailed report...")
            generate_detailed_report(metrics, human_scores, automated_scores, f"{args.output}_report.txt")
    
    elif mode == "automated_comparison":
        # Compare two automated methods
        print("📁 Loading automated score files...")
        auto_scores_1 = load_automated_scores(args.automated1)
        auto_scores_2 = load_automated_scores(args.automated2)
        
        print(f"🤖 Method 1 scores: {len(auto_scores_1)}")
        print(f"🤖 Method 2 scores: {len(auto_scores_2)}")
        
        # Convert second set to "human" format for comparison
        pseudo_human_scores = []
        for score in auto_scores_1:
            pseudo_human = HumanScore(
                question_id=score.question_id,
                overall_score=score.overall_score,
                dimension_scores=score.dimension_scores,
                justification=f"Automated by {score.scoring_method}",
                evaluator_id=score.scoring_method,
                evaluation_date=score.evaluation_date,
                confidence=score.confidence or 0.8
            )
            pseudo_human_scores.append(pseudo_human)
        
        # Analyze alignment
        print("\n🔄 Computing method comparison...")
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(pseudo_human_scores, auto_scores_2)
        
        print(f"\n📊 METHOD COMPARISON RESULTS")
        print(f"Correlation: {metrics.overall_pearson:.3f}")
        print(f"Agreement: {metrics.agreement_percentage_10:.1f}%")
        print(f"Mean difference: {metrics.mean_bias_error:+.2f} points")
        
        method1_name = auto_scores_1[0].scoring_method if auto_scores_1 else "method1"
        method2_name = auto_scores_2[0].scoring_method if auto_scores_2 else "method2"
        print(f"({method2_name} - {method1_name})")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: pip install matplotlib seaborn pandas")
        sys.exit(1)
    
    main()