"""
Experiment Analysis System for Multi-Agent Scoring Results.

Provides comprehensive analysis capabilities for experiment tracking data,
including performance comparisons, trend analysis, and challenging question identification.
"""

import statistics
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import logging

from .tracking import ExperimentRecord, RunMetadata, ExperimentTracker

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzes performance across evaluation runs."""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
    
    def analyze_run_performance(self, run_id: str) -> Dict[str, Any]:
        """
        Analyze performance for a specific run.
        
        Args:
            run_id: ID of the run to analyze
            
        Returns:
            Dictionary with performance metrics and analysis
        """
        results = self.tracker.get_run_results(run_id)
        metadata = self.tracker.get_run_metadata(run_id)
        
        if not results:
            return {"error": f"No results found for run {run_id}"}
        
        # Basic statistics
        scores = [r.final_score for r in results]
        execution_times = [r.execution_time_ms for r in results]
        confidence_levels = [r.confidence_level for r in results]
        
        analysis = {
            "run_id": run_id,
            "run_metadata": {
                "model_name": metadata.model_name if metadata else "unknown",
                "timestamp": metadata.timestamp if metadata else "unknown",
                "description": metadata.description if metadata else "",
                "tags": metadata.tags if metadata else []
            },
            "summary": {
                "total_questions": len(results),
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "std_dev_score": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_confidence": statistics.mean(confidence_levels),
                "mean_execution_time_ms": statistics.mean(execution_times),
                "total_execution_time_ms": sum(execution_times)
            },
            "score_distribution": self._analyze_score_distribution(results),
            "critic_performance": self._analyze_critic_performance(results),
            "consensus_analysis": self._analyze_consensus_patterns(results),
            "challenging_questions": self._identify_challenging_questions(results),
            "high_performing_questions": self._identify_high_performing_questions(results)
        }
        
        return analysis
    
    def compare_runs(self, run_id_1: str, run_id_2: str) -> Dict[str, Any]:
        """
        Compare performance between two runs.
        
        Args:
            run_id_1: First run to compare
            run_id_2: Second run to compare
            
        Returns:
            Comparison analysis
        """
        analysis_1 = self.analyze_run_performance(run_id_1)
        analysis_2 = self.analyze_run_performance(run_id_2)
        
        if "error" in analysis_1 or "error" in analysis_2:
            return {"error": "One or both runs not found"}
        
        comparison = {
            "run_1": {
                "run_id": run_id_1,
                "model": analysis_1["run_metadata"]["model_name"],
                "timestamp": analysis_1["run_metadata"]["timestamp"]
            },
            "run_2": {
                "run_id": run_id_2,
                "model": analysis_2["run_metadata"]["model_name"],
                "timestamp": analysis_2["run_metadata"]["timestamp"]
            },
            "score_comparison": {
                "mean_score_diff": analysis_2["summary"]["mean_score"] - analysis_1["summary"]["mean_score"],
                "median_score_diff": analysis_2["summary"]["median_score"] - analysis_1["summary"]["median_score"],
                "consistency_diff": analysis_1["summary"]["std_dev_score"] - analysis_2["summary"]["std_dev_score"],
                "run_1_better_questions": 0,
                "run_2_better_questions": 0,
                "tied_questions": 0
            },
            "performance_comparison": {
                "execution_time_diff_ms": analysis_2["summary"]["mean_execution_time_ms"] - analysis_1["summary"]["mean_execution_time_ms"],
                "confidence_diff": analysis_2["summary"]["mean_confidence"] - analysis_1["summary"]["mean_confidence"]
            }
        }
        
        # Question-by-question comparison
        results_1 = {r.question_id: r for r in self.tracker.get_run_results(run_id_1)}
        results_2 = {r.question_id: r for r in self.tracker.get_run_results(run_id_2)}
        
        common_questions = set(results_1.keys()) & set(results_2.keys())
        question_comparisons = []
        
        for question_id in common_questions:
            r1, r2 = results_1[question_id], results_2[question_id]
            score_diff = r2.final_score - r1.final_score
            
            if score_diff > 0:
                comparison["score_comparison"]["run_2_better_questions"] += 1
            elif score_diff < 0:
                comparison["score_comparison"]["run_1_better_questions"] += 1
            else:
                comparison["score_comparison"]["tied_questions"] += 1
            
            question_comparisons.append({
                "question_id": question_id,
                "score_diff": score_diff,
                "run_1_score": r1.final_score,
                "run_2_score": r2.final_score,
                "run_1_tier": r1.final_tier,
                "run_2_tier": r2.final_tier,
                "confidence_diff": r2.confidence_level - r1.confidence_level,
                "execution_time_diff": r2.execution_time_ms - r1.execution_time_ms
            })
        
        comparison["question_comparisons"] = sorted(
            question_comparisons, 
            key=lambda x: abs(x["score_diff"]), 
            reverse=True
        )
        
        # Significant improvements and regressions
        comparison["significant_changes"] = {
            "improvements": [q for q in question_comparisons if q["score_diff"] >= 10],
            "regressions": [q for q in question_comparisons if q["score_diff"] <= -10]
        }
        
        return comparison
    
    def analyze_model_trends(self, model_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Analyze performance trends for a specific model over time.
        
        Args:
            model_name: Name of model to analyze
            limit: Maximum number of recent runs to include
            
        Returns:
            Trend analysis
        """
        all_results = self.tracker.get_results_by_model(model_name)
        
        if not all_results:
            return {"error": f"No results found for model {model_name}"}
        
        # Group by run and sort by timestamp
        run_groups = defaultdict(list)
        for result in all_results:
            run_groups[result.run_id].append(result)
        
        # Get run metadata and sort by timestamp
        run_summaries = []
        for run_id, results in run_groups.items():
            metadata = self.tracker.get_run_metadata(run_id)
            if metadata:
                scores = [r.final_score for r in results]
                run_summaries.append({
                    "run_id": run_id,
                    "timestamp": metadata.timestamp,
                    "mean_score": statistics.mean(scores),
                    "num_questions": len(results),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "config_hash": metadata.config_hash
                })
        
        # Sort by timestamp and limit
        run_summaries.sort(key=lambda x: x["timestamp"])
        recent_runs = run_summaries[-limit:] if limit else run_summaries
        
        if len(recent_runs) < 2:
            return {
                "model_name": model_name,
                "recent_runs": recent_runs,
                "trend_analysis": "Insufficient data for trend analysis (need at least 2 runs)"
            }
        
        # Calculate trends
        scores = [run["mean_score"] for run in recent_runs]
        consistency = [run["std_dev"] for run in recent_runs]
        
        # Simple linear trend
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            x = list(range(len(values)))
            n = len(values)
            xy_sum = sum(i * v for i, v in zip(x, values))
            x_sum = sum(x)
            y_sum = sum(values)
            x_sq_sum = sum(i * i for i in x)
            
            slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum) if n * x_sq_sum - x_sum * x_sum != 0 else 0
            return slope
        
        score_trend = calculate_trend(scores)
        consistency_trend = calculate_trend(consistency)
        
        return {
            "model_name": model_name,
            "analysis_period": {
                "start_date": recent_runs[0]["timestamp"],
                "end_date": recent_runs[-1]["timestamp"],
                "num_runs": len(recent_runs)
            },
            "recent_runs": recent_runs,
            "trend_analysis": {
                "score_trend": "improving" if score_trend > 1 else "declining" if score_trend < -1 else "stable",
                "score_trend_value": score_trend,
                "consistency_trend": "more_consistent" if consistency_trend < -0.5 else "less_consistent" if consistency_trend > 0.5 else "stable",
                "consistency_trend_value": consistency_trend,
                "best_run": max(recent_runs, key=lambda x: x["mean_score"]),
                "worst_run": min(recent_runs, key=lambda x: x["mean_score"]),
                "performance_range": max(scores) - min(scores)
            }
        }
    
    def identify_challenging_questions(self, min_results: int = 3) -> List[Dict[str, Any]]:
        """
        Identify questions that consistently receive low scores across runs.
        
        Args:
            min_results: Minimum number of results required for a question to be considered
            
        Returns:
            List of challenging questions with statistics
        """
        all_results = self.tracker.storage.get_results()
        
        # Group by question
        question_groups = defaultdict(list)
        for result in all_results:
            question_groups[result.question_id].append(result)
        
        challenging = []
        for question_id, results in question_groups.items():
            if len(results) >= min_results:
                scores = [r.final_score for r in results]
                mean_score = statistics.mean(scores)
                
                if mean_score < 60:  # Threshold for "challenging"
                    challenging.append({
                        "question_id": question_id,
                        "question_text": results[0].question_text[:200] + "..." if len(results[0].question_text) > 200 else results[0].question_text,
                        "num_evaluations": len(results),
                        "mean_score": mean_score,
                        "median_score": statistics.median(scores),
                        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                        "min_score": min(scores),
                        "max_score": max(scores),
                        "models_evaluated": list(set(r.model_name for r in results)),
                        "consistent_difficulty": statistics.stdev(scores) < 10 if len(scores) > 1 else True
                    })
        
        return sorted(challenging, key=lambda x: x["mean_score"])
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get overall summary statistics across all experiments."""
        all_results = self.tracker.storage.get_results()
        all_runs = self.tracker.get_all_runs()
        
        if not all_results:
            return {"error": "No experimental data found"}
        
        scores = [r.final_score for r in all_results]
        models = [r.model_name for r in all_results]
        critics_used_counts = Counter()
        
        for result in all_results:
            for critic in result.critics_used:
                critics_used_counts[critic] += 1
        
        return {
            "experiment_summary": {
                "total_evaluations": len(all_results),
                "total_runs": len(all_runs),
                "unique_questions": len(set(r.question_id for r in all_results)),
                "unique_models": len(set(models)),
                "date_range": {
                    "earliest": min(r.timestamp for r in all_results) if all_results else None,
                    "latest": max(r.timestamp for r in all_results) if all_results else None
                }
            },
            "score_statistics": {
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min_score": min(scores),
                "max_score": max(scores),
                "score_distribution": {
                    "excellent (90-100)": len([s for s in scores if s >= 90]),
                    "good (75-89)": len([s for s in scores if 75 <= s < 90]),
                    "adequate (60-74)": len([s for s in scores if 60 <= s < 75]),
                    "poor (40-59)": len([s for s in scores if 40 <= s < 60]),
                    "inadequate (0-39)": len([s for s in scores if s < 40])
                }
            },
            "model_performance": self._summarize_model_performance(all_results),
            "critic_usage": dict(critics_used_counts.most_common()),
            "performance_insights": self._generate_performance_insights(all_results)
        }
    
    def _analyze_score_distribution(self, results: List[ExperimentRecord]) -> Dict[str, Any]:
        """Analyze score distribution for a set of results."""
        scores = [r.final_score for r in results]
        
        distribution = {
            "excellent (90-100)": len([s for s in scores if s >= 90]),
            "good (75-89)": len([s for s in scores if 75 <= s < 90]),
            "adequate (60-74)": len([s for s in scores if 60 <= s < 75]),
            "poor (40-59)": len([s for s in scores if 40 <= s < 60]),
            "inadequate (0-39)": len([s for s in scores if s < 40])
        }
        
        total = len(scores)
        percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in distribution.items()}
        
        return {
            "counts": distribution,
            "percentages": percentages,
            "quartiles": {
                "q1": scores[len(scores) // 4] if scores else 0,
                "q2": statistics.median(scores) if scores else 0,
                "q3": scores[3 * len(scores) // 4] if scores else 0
            } if len(scores) >= 4 else {}
        }
    
    def _analyze_critic_performance(self, results: List[ExperimentRecord]) -> Dict[str, Any]:
        """Analyze individual critic performance patterns."""
        critic_stats = defaultdict(list)
        
        for result in results:
            for critic_name, score in result.critic_scores.items():
                critic_stats[critic_name].append({
                    "score": score,
                    "final_score": result.final_score,
                    "question_id": result.question_id
                })
        
        analysis = {}
        for critic_name, scores_data in critic_stats.items():
            scores = [d["score"] for d in scores_data]
            final_scores = [d["final_score"] for d in scores_data]
            
            # Correlation with final score
            correlation = self._calculate_correlation(scores, final_scores) if len(scores) > 1 else 0
            
            analysis[critic_name] = {
                "mean_score": statistics.mean(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "correlation_with_final": correlation,
                "evaluations_count": len(scores),
                "score_range": (min(scores), max(scores)) if scores else (0, 0)
            }
        
        return analysis
    
    def _analyze_consensus_patterns(self, results: List[ExperimentRecord]) -> Dict[str, Any]:
        """Analyze consensus patterns across critics."""
        consensus_levels = [r.consensus_level for r in results]
        score_variances = [r.score_variance for r in results]
        
        consensus_counts = Counter(consensus_levels)
        
        return {
            "consensus_distribution": dict(consensus_counts),
            "mean_score_variance": statistics.mean(score_variances),
            "high_disagreement_questions": [
                r.question_id for r in results 
                if r.score_variance > 20  # High variance threshold
            ],
            "perfect_consensus_questions": [
                r.question_id for r in results
                if r.consensus_level == "high" and r.score_variance < 5
            ]
        }
    
    def _identify_challenging_questions(self, results: List[ExperimentRecord]) -> List[Dict[str, Any]]:
        """Identify challenging questions in this run."""
        challenging = []
        
        for result in results:
            if result.final_score < 60:  # Threshold for challenging
                challenging.append({
                    "question_id": result.question_id,
                    "score": result.final_score,
                    "tier": result.final_tier,
                    "consensus": result.consensus_level,
                    "variance": result.score_variance,
                    "question_preview": result.question_text[:100] + "..." if len(result.question_text) > 100 else result.question_text
                })
        
        return sorted(challenging, key=lambda x: x["score"])
    
    def _identify_high_performing_questions(self, results: List[ExperimentRecord]) -> List[Dict[str, Any]]:
        """Identify high-performing questions in this run."""
        high_performing = []
        
        for result in results:
            if result.final_score >= 85:  # Threshold for high performance
                high_performing.append({
                    "question_id": result.question_id,
                    "score": result.final_score,
                    "tier": result.final_tier,
                    "consensus": result.consensus_level,
                    "variance": result.score_variance,
                    "question_preview": result.question_text[:100] + "..." if len(result.question_text) > 100 else result.question_text
                })
        
        return sorted(high_performing, key=lambda x: x["score"], reverse=True)
    
    def _summarize_model_performance(self, results: List[ExperimentRecord]) -> Dict[str, Any]:
        """Summarize performance by model."""
        model_groups = defaultdict(list)
        for result in results:
            model_groups[result.model_name].append(result.final_score)
        
        model_summary = {}
        for model_name, scores in model_groups.items():
            model_summary[model_name] = {
                "evaluations": len(scores),
                "mean_score": statistics.mean(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min_score": min(scores),
                "max_score": max(scores)
            }
        
        return dict(sorted(model_summary.items(), key=lambda x: x[1]["mean_score"], reverse=True))
    
    def _generate_performance_insights(self, results: List[ExperimentRecord]) -> List[str]:
        """Generate actionable performance insights."""
        insights = []
        
        scores = [r.final_score for r in results]
        variances = [r.score_variance for r in results]
        
        # Score distribution insights
        excellent_count = len([s for s in scores if s >= 90])
        poor_count = len([s for s in scores if s < 60])
        
        if excellent_count / len(scores) > 0.5:
            insights.append(f"Strong overall performance: {excellent_count}/{len(scores)} evaluations scored 90+")
        
        if poor_count / len(scores) > 0.3:
            insights.append(f"Performance concerns: {poor_count}/{len(scores)} evaluations scored below 60")
        
        # Consensus insights
        high_variance_count = len([v for v in variances if v > 20])
        if high_variance_count > 0:
            insights.append(f"Critic disagreement detected in {high_variance_count} evaluations - review challenging questions")
        
        # Model comparison insights
        model_scores = defaultdict(list)
        for result in results:
            model_scores[result.model_name].append(result.final_score)
        
        if len(model_scores) > 1:
            model_means = {model: statistics.mean(scores) for model, scores in model_scores.items()}
            best_model = max(model_means, key=model_means.get)
            worst_model = min(model_means, key=model_means.get)
            
            if model_means[best_model] - model_means[worst_model] > 10:
                insights.append(f"Significant model performance gap: {best_model} outperforms {worst_model} by {model_means[best_model] - model_means[worst_model]:.1f} points")
        
        return insights
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0


def analyze_run(tracker: ExperimentTracker, run_id: str) -> Dict[str, Any]:
    """Convenience function to analyze a single run."""
    analyzer = PerformanceAnalyzer(tracker)
    return analyzer.analyze_run_performance(run_id)


def compare_runs(tracker: ExperimentTracker, run_id_1: str, run_id_2: str) -> Dict[str, Any]:
    """Convenience function to compare two runs."""
    analyzer = PerformanceAnalyzer(tracker)
    return analyzer.compare_runs(run_id_1, run_id_2)


def get_challenging_questions(tracker: ExperimentTracker, min_results: int = 3) -> List[Dict[str, Any]]:
    """Convenience function to identify challenging questions."""
    analyzer = PerformanceAnalyzer(tracker)
    return analyzer.identify_challenging_questions(min_results)


def get_summary_statistics(tracker: ExperimentTracker) -> Dict[str, Any]:
    """Convenience function to get overall summary statistics."""
    analyzer = PerformanceAnalyzer(tracker)
    return analyzer.get_summary_statistics()