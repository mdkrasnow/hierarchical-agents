"""
Calibration and quality assurance system for automated scoring.

This module provides functionality to:
1. Select representative question subsets for human labeling
2. Compute alignment between human and automated scores
3. Calibrate scoring parameters based on human feedback
4. Validate scoring consistency and reliability

The system uses the same rubric (0-100) as automated scoring to ensure
direct comparability between human and machine evaluations.
"""

import json
import logging
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr

try:
    # Try relative imports first (when used as module)
    from ..critics.models import CriticScore
    from ..critics.single_critic import SingleCriticAgent, score_answer
    from ..critics.orchestrator import MultiCriticOrchestrator
    from ..critics.debate_models import MultiCriticRequest
except ImportError:
    # Fall back to absolute imports (when used directly)
    from critics.models import CriticScore
    from critics.single_critic import SingleCriticAgent, score_answer
    from critics.orchestrator import MultiCriticOrchestrator
    from critics.debate_models import MultiCriticRequest


logger = logging.getLogger(__name__)


@dataclass
class HumanScore:
    """Represents a human-provided score for a question-answer pair."""
    question_id: str
    overall_score: int  # 0-100
    dimension_scores: Dict[str, int]  # dimension -> score 0-100
    justification: str
    evaluator_id: str
    evaluation_date: str
    confidence: float  # 0-1, evaluator's confidence in their scoring
    time_spent_minutes: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class AutomatedScore:
    """Represents an automated score (single or multi-critic)."""
    question_id: str
    overall_score: int
    dimension_scores: Dict[str, int]
    scoring_method: str  # "single_critic" or "multi_critic"
    evaluation_date: str
    execution_time_ms: float
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class AlignmentMetrics:
    """Metrics comparing human and automated scores."""
    
    # Correlation metrics
    overall_pearson: float
    overall_spearman: float
    overall_pearson_p_value: float
    overall_spearman_p_value: float
    
    # Dimension-wise correlations
    dimension_correlations: Dict[str, Dict[str, float]]
    
    # Error metrics
    mean_absolute_error: float
    root_mean_square_error: float
    mean_bias_error: float  # positive = automated scores higher
    
    # Agreement metrics
    exact_agreement_count: int
    within_5_points_count: int
    within_10_points_count: int
    agreement_percentage_5: float
    agreement_percentage_10: float
    
    # Distribution comparison
    human_score_stats: Dict[str, float]
    automated_score_stats: Dict[str, float]
    
    # Sample size
    n_samples: int
    
    # Quality indicators
    reliability_estimate: Optional[float] = None
    systematic_bias_detected: bool = False
    outlier_count: int = 0


@dataclass
class CalibrationResult:
    """Results from a calibration analysis."""
    
    alignment_metrics: AlignmentMetrics
    recommendations: List[str]
    areas_of_concern: List[str] 
    calibration_quality: str  # "excellent", "good", "needs_improvement", "poor"
    confidence_level: float  # 0-1
    
    # Detailed analysis
    score_distribution_analysis: Dict[str, Any]
    dimension_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    
    # Metadata
    analysis_date: str
    data_subset_info: Dict[str, Any]
    human_evaluators_count: int
    automated_method: str


class CalibrationDataSelector:
    """Selects representative subsets of questions for human labeling."""
    
    def __init__(self, questions_file: str = "data/questions.jsonl"):
        """
        Initialize with questions dataset.
        
        Args:
            questions_file: Path to JSONL file with questions
        """
        self.questions_file = Path(questions_file)
        self.questions = self._load_questions()
        
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from JSONL file."""
        questions = []
        try:
            with open(self.questions_file, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            logger.info(f"Loaded {len(questions)} questions from {self.questions_file}")
        except Exception as e:
            logger.error(f"Failed to load questions: {e}")
            raise
        return questions
    
    def select_calibration_subset(self, 
                                 target_size: int = 15,
                                 strategy: str = "balanced_sampling") -> List[Dict[str, Any]]:
        """
        Select a representative subset of questions for human evaluation.
        
        Args:
            target_size: Desired number of questions (10-20 range)
            strategy: Selection strategy ("balanced_sampling", "random", "stratified")
            
        Returns:
            List of selected question dictionaries
        """
        if target_size < 10 or target_size > 20:
            logger.warning(f"Target size {target_size} outside recommended range 10-20")
        
        if target_size > len(self.questions):
            logger.warning(f"Target size {target_size} exceeds available questions {len(self.questions)}")
            return self.questions.copy()
        
        if strategy == "balanced_sampling":
            return self._balanced_sampling(target_size)
        elif strategy == "random":
            return self._random_sampling(target_size)
        elif strategy == "stratified":
            return self._stratified_sampling(target_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _balanced_sampling(self, target_size: int) -> List[Dict[str, Any]]:
        """Select questions balancing across categories, difficulties, and rubric focus."""
        selected = []
        
        # Group by key characteristics
        by_category = {}
        by_difficulty = {}
        by_rubric_focus = {}
        
        for q in self.questions:
            # Category grouping
            cat = q.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(q)
            
            # Difficulty grouping
            diff = q.get('difficulty', 'unknown')
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(q)
            
            # Rubric focus grouping
            focus = tuple(sorted(q.get('rubric_focus', [])))
            if focus not in by_rubric_focus:
                by_rubric_focus[focus] = []
            by_rubric_focus[focus].append(q)
        
        logger.info(f"Found {len(by_category)} categories, {len(by_difficulty)} difficulties, "
                   f"{len(by_rubric_focus)} rubric focus patterns")
        
        # Sample proportionally from each group
        import random
        random.seed(42)  # For reproducible selection
        
        categories = list(by_category.keys())
        target_per_category = max(1, target_size // len(categories))
        
        for category in categories:
            category_questions = by_category[category]
            n_from_category = min(target_per_category, len(category_questions))
            
            # Within category, balance by difficulty
            selected_from_cat = random.sample(category_questions, n_from_category)
            selected.extend(selected_from_cat)
            
            if len(selected) >= target_size:
                break
        
        # Fill remaining slots if needed
        remaining_questions = [q for q in self.questions if q not in selected]
        while len(selected) < target_size and remaining_questions:
            selected.append(random.choice(remaining_questions))
            remaining_questions = [q for q in remaining_questions if q != selected[-1]]
        
        # Shuffle final selection
        random.shuffle(selected)
        selected = selected[:target_size]
        
        logger.info(f"Selected {len(selected)} questions using balanced sampling")
        self._log_selection_stats(selected)
        
        return selected
    
    def _random_sampling(self, target_size: int) -> List[Dict[str, Any]]:
        """Random selection of questions."""
        import random
        random.seed(42)
        selected = random.sample(self.questions, target_size)
        logger.info(f"Selected {len(selected)} questions using random sampling")
        self._log_selection_stats(selected)
        return selected
    
    def _stratified_sampling(self, target_size: int) -> List[Dict[str, Any]]:
        """Stratified sampling ensuring representation of key characteristics."""
        import random
        random.seed(42)
        
        # Define strata
        strata = {}
        
        for q in self.questions:
            # Create stratum key from category and difficulty
            key = (q.get('category', 'unknown'), q.get('difficulty', 'unknown'))
            if key not in strata:
                strata[key] = []
            strata[key].append(q)
        
        # Calculate samples per stratum
        selected = []
        strata_keys = list(strata.keys())
        target_per_stratum = max(1, target_size // len(strata_keys))
        
        for stratum_key in strata_keys:
            stratum_questions = strata[stratum_key]
            n_from_stratum = min(target_per_stratum, len(stratum_questions))
            selected.extend(random.sample(stratum_questions, n_from_stratum))
            
            if len(selected) >= target_size:
                break
        
        # Trim to exact size
        selected = selected[:target_size]
        
        logger.info(f"Selected {len(selected)} questions using stratified sampling")
        self._log_selection_stats(selected)
        
        return selected
    
    def _log_selection_stats(self, selected: List[Dict[str, Any]]):
        """Log statistics about the selected subset."""
        categories = {}
        difficulties = {}
        rubric_focuses = {}
        
        for q in selected:
            # Count categories
            cat = q.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count difficulties  
            diff = q.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
            
            # Count rubric focuses
            for focus in q.get('rubric_focus', []):
                rubric_focuses[focus] = rubric_focuses.get(focus, 0) + 1
        
        logger.info(f"Selection stats - Categories: {categories}")
        logger.info(f"Selection stats - Difficulties: {difficulties}")
        logger.info(f"Selection stats - Rubric focuses: {rubric_focuses}")


class ScoreAlignmentAnalyzer:
    """Analyzes alignment between human and automated scores."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def compute_alignment(self, 
                         human_scores: List[HumanScore],
                         automated_scores: List[AutomatedScore]) -> AlignmentMetrics:
        """
        Compute comprehensive alignment metrics between human and automated scores.
        
        Args:
            human_scores: List of human-provided scores
            automated_scores: List of automated scores
            
        Returns:
            AlignmentMetrics with detailed comparison
        """
        # Validate inputs
        if len(human_scores) != len(automated_scores):
            raise ValueError("Human and automated score lists must have same length")
        
        if len(human_scores) == 0:
            raise ValueError("Cannot compute alignment with empty score lists")
        
        # Align scores by question_id
        aligned_pairs = self._align_scores(human_scores, automated_scores)
        
        if len(aligned_pairs) == 0:
            raise ValueError("No matching question IDs found between human and automated scores")
        
        # Extract overall scores for analysis
        human_overall = [pair[0].overall_score for pair in aligned_pairs]
        auto_overall = [pair[1].overall_score for pair in aligned_pairs]
        
        # Compute correlation metrics
        overall_pearson, overall_pearson_p = pearsonr(human_overall, auto_overall)
        overall_spearman, overall_spearman_p = spearmanr(human_overall, auto_overall)
        
        # Compute dimension-wise correlations
        dimension_correlations = self._compute_dimension_correlations(aligned_pairs)
        
        # Compute error metrics
        errors = np.array(auto_overall) - np.array(human_overall)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mbe = np.mean(errors)
        
        # Compute agreement metrics
        exact_agreement = sum(1 for h, a in zip(human_overall, auto_overall) if h == a)
        within_5 = sum(1 for h, a in zip(human_overall, auto_overall) if abs(h - a) <= 5)
        within_10 = sum(1 for h, a in zip(human_overall, auto_overall) if abs(h - a) <= 10)
        
        n_samples = len(aligned_pairs)
        agreement_5_pct = (within_5 / n_samples) * 100
        agreement_10_pct = (within_10 / n_samples) * 100
        
        # Compute distribution statistics
        human_stats = {
            'mean': statistics.mean(human_overall),
            'median': statistics.median(human_overall), 
            'std_dev': statistics.stdev(human_overall) if len(human_overall) > 1 else 0,
            'min': min(human_overall),
            'max': max(human_overall)
        }
        
        auto_stats = {
            'mean': statistics.mean(auto_overall),
            'median': statistics.median(auto_overall),
            'std_dev': statistics.stdev(auto_overall) if len(auto_overall) > 1 else 0,
            'min': min(auto_overall),
            'max': max(auto_overall)
        }
        
        # Detect outliers and systematic bias
        outlier_count = self._count_outliers(errors)
        systematic_bias = abs(mbe) > 5.0  # Bias > 5 points considered systematic
        
        return AlignmentMetrics(
            overall_pearson=overall_pearson,
            overall_spearman=overall_spearman,
            overall_pearson_p_value=overall_pearson_p,
            overall_spearman_p_value=overall_spearman_p,
            dimension_correlations=dimension_correlations,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            mean_bias_error=mbe,
            exact_agreement_count=exact_agreement,
            within_5_points_count=within_5,
            within_10_points_count=within_10,
            agreement_percentage_5=agreement_5_pct,
            agreement_percentage_10=agreement_10_pct,
            human_score_stats=human_stats,
            automated_score_stats=auto_stats,
            n_samples=n_samples,
            systematic_bias_detected=systematic_bias,
            outlier_count=outlier_count
        )
    
    def _align_scores(self, human_scores: List[HumanScore], 
                     automated_scores: List[AutomatedScore]) -> List[Tuple[HumanScore, AutomatedScore]]:
        """Align human and automated scores by question_id."""
        human_by_id = {score.question_id: score for score in human_scores}
        auto_by_id = {score.question_id: score for score in automated_scores}
        
        aligned_pairs = []
        for q_id in human_by_id:
            if q_id in auto_by_id:
                aligned_pairs.append((human_by_id[q_id], auto_by_id[q_id]))
        
        self.logger.info(f"Aligned {len(aligned_pairs)} score pairs out of "
                        f"{len(human_scores)} human and {len(automated_scores)} automated scores")
        
        return aligned_pairs
    
    def _compute_dimension_correlations(self, 
                                      aligned_pairs: List[Tuple[HumanScore, AutomatedScore]]) -> Dict[str, Dict[str, float]]:
        """Compute correlations for each scoring dimension."""
        dimension_correlations = {}
        
        # Get all dimension names from both human and automated scores
        all_dimensions = set()
        for human_score, auto_score in aligned_pairs:
            all_dimensions.update(human_score.dimension_scores.keys())
            all_dimensions.update(auto_score.dimension_scores.keys())
        
        for dimension in all_dimensions:
            human_dim_scores = []
            auto_dim_scores = []
            
            # Extract dimension scores where both human and automated have the dimension
            for human_score, auto_score in aligned_pairs:
                if (dimension in human_score.dimension_scores and 
                    dimension in auto_score.dimension_scores):
                    human_dim_scores.append(human_score.dimension_scores[dimension])
                    auto_dim_scores.append(auto_score.dimension_scores[dimension])
            
            if len(human_dim_scores) >= 2:  # Need at least 2 points for correlation
                try:
                    pearson_corr, pearson_p = pearsonr(human_dim_scores, auto_dim_scores)
                    spearman_corr, spearman_p = spearmanr(human_dim_scores, auto_dim_scores)
                    
                    dimension_correlations[dimension] = {
                        'pearson': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'n_samples': len(human_dim_scores)
                    }
                except Exception as e:
                    self.logger.warning(f"Could not compute correlation for dimension {dimension}: {e}")
                    dimension_correlations[dimension] = {
                        'pearson': 0.0,
                        'pearson_p_value': 1.0,
                        'spearman': 0.0,
                        'spearman_p_value': 1.0,
                        'n_samples': len(human_dim_scores)
                    }
        
        return dimension_correlations
    
    def _count_outliers(self, errors: np.ndarray) -> int:
        """Count outliers using IQR method."""
        if len(errors) < 4:
            return 0
        
        q25 = np.percentile(errors, 25)
        q75 = np.percentile(errors, 75)
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = (errors < lower_bound) | (errors > upper_bound)
        return np.sum(outliers)


class CalibrationAnalyzer:
    """Main analyzer for calibration and quality assurance."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.logger = logging.getLogger(__name__)
        self.alignment_analyzer = ScoreAlignmentAnalyzer()
    
    def analyze_calibration(self,
                           human_scores: List[HumanScore],
                           automated_scores: List[AutomatedScore],
                           metadata: Optional[Dict[str, Any]] = None) -> CalibrationResult:
        """
        Perform comprehensive calibration analysis.
        
        Args:
            human_scores: Human-provided scores
            automated_scores: Automated scores
            metadata: Additional analysis metadata
            
        Returns:
            CalibrationResult with comprehensive analysis and recommendations
        """
        self.logger.info(f"Starting calibration analysis with {len(human_scores)} human scores "
                        f"and {len(automated_scores)} automated scores")
        
        # Compute alignment metrics
        alignment_metrics = self.alignment_analyzer.compute_alignment(human_scores, automated_scores)
        
        # Perform detailed analyses
        score_distribution_analysis = self._analyze_score_distributions(alignment_metrics)
        dimension_analysis = self._analyze_dimensions(alignment_metrics)
        outlier_analysis = self._analyze_outliers(human_scores, automated_scores, alignment_metrics)
        bias_analysis = self._analyze_bias(alignment_metrics)
        
        # Generate recommendations and assess quality
        recommendations = self._generate_recommendations(alignment_metrics)
        areas_of_concern = self._identify_concerns(alignment_metrics)
        calibration_quality, confidence_level = self._assess_quality(alignment_metrics)
        
        # Extract metadata
        human_evaluators = len(set(score.evaluator_id for score in human_scores))
        automated_method = self._determine_automated_method(automated_scores)
        
        return CalibrationResult(
            alignment_metrics=alignment_metrics,
            recommendations=recommendations,
            areas_of_concern=areas_of_concern,
            calibration_quality=calibration_quality,
            confidence_level=confidence_level,
            score_distribution_analysis=score_distribution_analysis,
            dimension_analysis=dimension_analysis,
            outlier_analysis=outlier_analysis,
            bias_analysis=bias_analysis,
            analysis_date=datetime.now().isoformat(),
            data_subset_info=metadata or {},
            human_evaluators_count=human_evaluators,
            automated_method=automated_method
        )
    
    def _analyze_score_distributions(self, metrics: AlignmentMetrics) -> Dict[str, Any]:
        """Analyze score distributions."""
        human_stats = metrics.human_score_stats
        auto_stats = metrics.automated_score_stats
        
        return {
            'mean_difference': auto_stats['mean'] - human_stats['mean'],
            'median_difference': auto_stats['median'] - human_stats['median'], 
            'variance_ratio': (auto_stats['std_dev']**2) / (human_stats['std_dev']**2) if human_stats['std_dev'] > 0 else 0,
            'range_comparison': {
                'human_range': human_stats['max'] - human_stats['min'],
                'automated_range': auto_stats['max'] - auto_stats['min']
            },
            'distribution_similarity': 'similar' if abs(auto_stats['mean'] - human_stats['mean']) < 3 else 'different'
        }
    
    def _analyze_dimensions(self, metrics: AlignmentMetrics) -> Dict[str, Any]:
        """Analyze dimension-wise performance."""
        dim_analysis = {}
        
        for dimension, corr_data in metrics.dimension_correlations.items():
            quality = 'poor'
            if corr_data['pearson'] > 0.8:
                quality = 'excellent'
            elif corr_data['pearson'] > 0.6:
                quality = 'good'
            elif corr_data['pearson'] > 0.4:
                quality = 'fair'
            
            dim_analysis[dimension] = {
                'correlation_quality': quality,
                'pearson_correlation': corr_data['pearson'],
                'significant': corr_data['pearson_p_value'] < 0.05,
                'sample_size': corr_data['n_samples']
            }
        
        # Find best and worst performing dimensions
        if dim_analysis:
            best_dim = max(dim_analysis.keys(), key=lambda k: dim_analysis[k]['pearson_correlation'])
            worst_dim = min(dim_analysis.keys(), key=lambda k: dim_analysis[k]['pearson_correlation'])
            
            dim_analysis['summary'] = {
                'best_dimension': best_dim,
                'worst_dimension': worst_dim,
                'avg_correlation': statistics.mean(d['pearson_correlation'] for d in dim_analysis.values() if isinstance(d, dict))
            }
        
        return dim_analysis
    
    def _analyze_outliers(self, human_scores: List[HumanScore], 
                         automated_scores: List[AutomatedScore], 
                         metrics: AlignmentMetrics) -> Dict[str, Any]:
        """Analyze outliers and extreme disagreements."""
        aligned_pairs = self.alignment_analyzer._align_scores(human_scores, automated_scores)
        
        large_disagreements = []
        for human_score, auto_score in aligned_pairs:
            diff = abs(human_score.overall_score - auto_score.overall_score)
            if diff > 15:  # Large disagreement threshold
                large_disagreements.append({
                    'question_id': human_score.question_id,
                    'human_score': human_score.overall_score,
                    'automated_score': auto_score.overall_score,
                    'difference': diff,
                    'human_justification': human_score.justification,
                    'human_confidence': human_score.confidence
                })
        
        return {
            'large_disagreements': large_disagreements,
            'large_disagreement_count': len(large_disagreements),
            'outlier_rate': (metrics.outlier_count / metrics.n_samples) * 100 if metrics.n_samples > 0 else 0,
            'worst_disagreement': max(large_disagreements, key=lambda x: x['difference']) if large_disagreements else None
        }
    
    def _analyze_bias(self, metrics: AlignmentMetrics) -> Dict[str, Any]:
        """Analyze systematic bias patterns."""
        bias_direction = 'none'
        bias_magnitude = 'small'
        
        if metrics.mean_bias_error > 5:
            bias_direction = 'automated_higher'
        elif metrics.mean_bias_error < -5:
            bias_direction = 'automated_lower'
        
        if abs(metrics.mean_bias_error) > 10:
            bias_magnitude = 'large'
        elif abs(metrics.mean_bias_error) > 5:
            bias_magnitude = 'moderate'
        
        return {
            'systematic_bias_detected': metrics.systematic_bias_detected,
            'bias_direction': bias_direction,
            'bias_magnitude': bias_magnitude,
            'mean_bias_error': metrics.mean_bias_error,
            'bias_assessment': f"{bias_magnitude} {bias_direction} bias" if bias_direction != 'none' else 'no significant bias'
        }
    
    def _generate_recommendations(self, metrics: AlignmentMetrics) -> List[str]:
        """Generate actionable recommendations based on alignment metrics."""
        recommendations = []
        
        # Correlation-based recommendations
        if metrics.overall_pearson < 0.6:
            recommendations.append(
                "Low overall correlation detected. Consider reviewing critic prompts and rubric weights."
            )
        
        # Bias-based recommendations  
        if metrics.systematic_bias_detected:
            if metrics.mean_bias_error > 0:
                recommendations.append(
                    "Automated system consistently scores higher than humans. Consider adjusting score calibration downward."
                )
            else:
                recommendations.append(
                    "Automated system consistently scores lower than humans. Consider adjusting score calibration upward."
                )
        
        # Agreement-based recommendations
        if metrics.agreement_percentage_10 < 70:
            recommendations.append(
                "Low score agreement (within 10 points). Review scoring criteria and consider additional human training."
            )
        
        # Error-based recommendations
        if metrics.mean_absolute_error > 8:
            recommendations.append(
                "High mean absolute error. Consider refining automated scoring algorithm or increasing training data."
            )
        
        # Sample size recommendations
        if metrics.n_samples < 15:
            recommendations.append(
                "Small sample size for calibration. Consider collecting more human evaluations for robust analysis."
            )
        
        # Dimension-specific recommendations
        for dimension, corr_data in metrics.dimension_correlations.items():
            if corr_data['pearson'] < 0.4:
                recommendations.append(
                    f"Poor correlation for {dimension} dimension ({corr_data['pearson']:.2f}). "
                    f"Review scoring criteria and prompts for this dimension."
                )
        
        return recommendations
    
    def _identify_concerns(self, metrics: AlignmentMetrics) -> List[str]:
        """Identify areas of concern in the calibration."""
        concerns = []
        
        if metrics.overall_pearson < 0.4:
            concerns.append("Very low correlation between human and automated scores")
        
        if metrics.systematic_bias_detected:
            concerns.append("Systematic bias detected in automated scoring")
        
        if metrics.outlier_count / metrics.n_samples > 0.2:
            concerns.append("High proportion of outlier scores (>20%)")
        
        if metrics.agreement_percentage_5 < 30:
            concerns.append("Very low exact agreement between human and automated scores")
        
        if any(corr_data['pearson'] < 0.3 for corr_data in metrics.dimension_correlations.values()):
            concerns.append("One or more dimensions show very poor correlation")
        
        return concerns
    
    def _assess_quality(self, metrics: AlignmentMetrics) -> Tuple[str, float]:
        """Assess overall calibration quality and confidence."""
        
        # Scoring criteria weights
        correlation_weight = 0.4
        agreement_weight = 0.3
        bias_weight = 0.2
        error_weight = 0.1
        
        # Score each component (0-1)
        correlation_score = max(0, min(1, metrics.overall_pearson))
        agreement_score = metrics.agreement_percentage_10 / 100
        bias_score = max(0, 1 - abs(metrics.mean_bias_error) / 20)  # Penalize bias >20 points
        error_score = max(0, 1 - metrics.mean_absolute_error / 20)  # Penalize MAE >20 points
        
        # Weighted composite score
        composite_score = (
            correlation_score * correlation_weight +
            agreement_score * agreement_weight +
            bias_score * bias_weight +
            error_score * error_weight
        )
        
        # Determine quality category
        if composite_score > 0.8:
            quality = "excellent"
        elif composite_score > 0.6:
            quality = "good"
        elif composite_score > 0.4:
            quality = "needs_improvement"
        else:
            quality = "poor"
        
        # Adjust confidence based on sample size
        base_confidence = composite_score
        if metrics.n_samples < 10:
            base_confidence *= 0.7
        elif metrics.n_samples < 15:
            base_confidence *= 0.85
        
        return quality, base_confidence
    
    def _determine_automated_method(self, automated_scores: List[AutomatedScore]) -> str:
        """Determine the automated scoring method used."""
        if not automated_scores:
            return "unknown"
        
        methods = [score.scoring_method for score in automated_scores]
        if len(set(methods)) == 1:
            return methods[0]
        else:
            return "mixed"


def save_human_scores(scores: List[HumanScore], filepath: str):
    """Save human scores to JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        for score in scores:
            score_dict = {
                'question_id': score.question_id,
                'overall_score': score.overall_score,
                'dimension_scores': score.dimension_scores,
                'justification': score.justification,
                'evaluator_id': score.evaluator_id,
                'evaluation_date': score.evaluation_date,
                'confidence': score.confidence,
                'time_spent_minutes': score.time_spent_minutes,
                'notes': score.notes
            }
            f.write(json.dumps(score_dict) + '\n')


def load_human_scores(filepath: str) -> List[HumanScore]:
    """Load human scores from JSONL file."""
    scores = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                score = HumanScore(
                    question_id=data['question_id'],
                    overall_score=data['overall_score'],
                    dimension_scores=data['dimension_scores'],
                    justification=data['justification'],
                    evaluator_id=data['evaluator_id'],
                    evaluation_date=data['evaluation_date'],
                    confidence=data['confidence'],
                    time_spent_minutes=data.get('time_spent_minutes'),
                    notes=data.get('notes')
                )
                scores.append(score)
    
    return scores


def save_calibration_result(result: CalibrationResult, filepath: str):
    """Save calibration result to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict for JSON serialization
    result_dict = {
        'alignment_metrics': {
            'overall_pearson': result.alignment_metrics.overall_pearson,
            'overall_spearman': result.alignment_metrics.overall_spearman,
            'overall_pearson_p_value': result.alignment_metrics.overall_pearson_p_value,
            'overall_spearman_p_value': result.alignment_metrics.overall_spearman_p_value,
            'dimension_correlations': result.alignment_metrics.dimension_correlations,
            'mean_absolute_error': result.alignment_metrics.mean_absolute_error,
            'root_mean_square_error': result.alignment_metrics.root_mean_square_error,
            'mean_bias_error': result.alignment_metrics.mean_bias_error,
            'exact_agreement_count': result.alignment_metrics.exact_agreement_count,
            'within_5_points_count': result.alignment_metrics.within_5_points_count,
            'within_10_points_count': result.alignment_metrics.within_10_points_count,
            'agreement_percentage_5': result.alignment_metrics.agreement_percentage_5,
            'agreement_percentage_10': result.alignment_metrics.agreement_percentage_10,
            'human_score_stats': result.alignment_metrics.human_score_stats,
            'automated_score_stats': result.alignment_metrics.automated_score_stats,
            'n_samples': result.alignment_metrics.n_samples,
            'systematic_bias_detected': result.alignment_metrics.systematic_bias_detected,
            'outlier_count': result.alignment_metrics.outlier_count
        },
        'recommendations': result.recommendations,
        'areas_of_concern': result.areas_of_concern,
        'calibration_quality': result.calibration_quality,
        'confidence_level': result.confidence_level,
        'score_distribution_analysis': result.score_distribution_analysis,
        'dimension_analysis': result.dimension_analysis,
        'outlier_analysis': result.outlier_analysis,
        'bias_analysis': result.bias_analysis,
        'analysis_date': result.analysis_date,
        'data_subset_info': result.data_subset_info,
        'human_evaluators_count': result.human_evaluators_count,
        'automated_method': result.automated_method
    }
    
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)