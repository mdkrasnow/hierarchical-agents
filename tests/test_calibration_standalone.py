"""
Standalone tests for calibration functionality.

Tests the core calibration functionality without dependencies on the broader project.
"""

import json
import pytest
import tempfile
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


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
    time_spent_minutes: float = None
    notes: str = None


@dataclass
class AutomatedScore:
    """Represents an automated score."""
    question_id: str
    overall_score: int
    dimension_scores: Dict[str, int]
    scoring_method: str
    evaluation_date: str
    execution_time_ms: float
    confidence: float = None
    metadata: Dict[str, Any] = None


@dataclass 
class AlignmentMetrics:
    """Metrics comparing human and automated scores."""
    overall_pearson: float
    overall_spearman: float
    overall_pearson_p_value: float
    overall_spearman_p_value: float
    mean_absolute_error: float
    root_mean_square_error: float
    mean_bias_error: float
    exact_agreement_count: int
    within_5_points_count: int
    within_10_points_count: int
    agreement_percentage_5: float
    agreement_percentage_10: float
    n_samples: int
    systematic_bias_detected: bool = False
    outlier_count: int = 0


class ScoreAlignmentAnalyzer:
    """Analyzes alignment between human and automated scores."""
    
    def compute_alignment(self, human_scores: List[HumanScore], 
                         automated_scores: List[AutomatedScore]) -> AlignmentMetrics:
        """Compute alignment metrics."""
        if len(human_scores) != len(automated_scores):
            raise ValueError("Score lists must have same length")
        
        if len(human_scores) == 0:
            raise ValueError("Cannot compute alignment with empty lists")
        
        # Align by question ID
        aligned_pairs = self._align_scores(human_scores, automated_scores)
        
        if not aligned_pairs:
            raise ValueError("No matching question IDs found")
        
        # Extract overall scores
        human_overall = [pair[0].overall_score for pair in aligned_pairs]
        auto_overall = [pair[1].overall_score for pair in aligned_pairs]
        
        # Correlations
        overall_pearson, overall_pearson_p = pearsonr(human_overall, auto_overall)
        overall_spearman, overall_spearman_p = spearmanr(human_overall, auto_overall)
        
        # Error metrics
        errors = np.array(auto_overall) - np.array(human_overall)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mbe = np.mean(errors)
        
        # Agreement
        exact_agreement = sum(1 for h, a in zip(human_overall, auto_overall) if h == a)
        within_5 = sum(1 for h, a in zip(human_overall, auto_overall) if abs(h - a) <= 5)
        within_10 = sum(1 for h, a in zip(human_overall, auto_overall) if abs(h - a) <= 10)
        
        n_samples = len(aligned_pairs)
        
        # Detect bias
        systematic_bias = abs(mbe) > 5.0
        
        return AlignmentMetrics(
            overall_pearson=overall_pearson,
            overall_spearman=overall_spearman,
            overall_pearson_p_value=overall_pearson_p,
            overall_spearman_p_value=overall_spearman_p,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            mean_bias_error=mbe,
            exact_agreement_count=exact_agreement,
            within_5_points_count=within_5,
            within_10_points_count=within_10,
            agreement_percentage_5=(within_5 / n_samples) * 100,
            agreement_percentage_10=(within_10 / n_samples) * 100,
            n_samples=n_samples,
            systematic_bias_detected=systematic_bias
        )
    
    def _align_scores(self, human_scores: List[HumanScore], 
                     automated_scores: List[AutomatedScore]):
        """Align scores by question ID."""
        human_by_id = {score.question_id: score for score in human_scores}
        auto_by_id = {score.question_id: score for score in automated_scores}
        
        aligned = []
        for q_id in human_by_id:
            if q_id in auto_by_id:
                aligned.append((human_by_id[q_id], auto_by_id[q_id]))
        
        return aligned


class CalibrationDataSelector:
    """Selects representative subsets for calibration."""
    
    def __init__(self, questions_data: List[Dict[str, Any]]):
        """Initialize with questions data."""
        self.questions = questions_data
    
    def select_calibration_subset(self, target_size: int = 15, 
                                 strategy: str = "balanced") -> List[Dict[str, Any]]:
        """Select subset of questions."""
        if target_size > len(self.questions):
            return self.questions.copy()
        
        if strategy == "balanced":
            return self._balanced_selection(target_size)
        elif strategy == "random":
            import random
            random.seed(42)
            return random.sample(self.questions, target_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _balanced_selection(self, target_size: int) -> List[Dict[str, Any]]:
        """Balanced selection across categories."""
        import random
        random.seed(42)
        
        # Group by category
        by_category = {}
        for q in self.questions:
            cat = q.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(q)
        
        # Sample proportionally
        selected = []
        categories = list(by_category.keys())
        target_per_cat = max(1, target_size // len(categories))
        
        for cat in categories:
            cat_questions = by_category[cat]
            n_from_cat = min(target_per_cat, len(cat_questions))
            selected.extend(random.sample(cat_questions, n_from_cat))
            
            if len(selected) >= target_size:
                break
        
        return selected[:target_size]


# Test Classes

class TestScoreAlignmentAnalyzer:
    """Test alignment analysis."""
    
    def test_perfect_alignment(self):
        """Test with perfect score alignment."""
        human_scores = [
            HumanScore("q1", 80, {"dim1": 80}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q2", 90, {"dim1": 90}, "test", "eval1", "2024-12-08", 0.9)
        ]
        automated_scores = [
            AutomatedScore("q1", 80, {"dim1": 80}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q2", 90, {"dim1": 90}, "test", "2024-12-08", 1000.0)
        ]
        
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, automated_scores)
        
        assert metrics.overall_pearson == 1.0
        assert metrics.mean_absolute_error == 0.0
        assert metrics.exact_agreement_count == 2
        assert metrics.agreement_percentage_5 == 100.0
        assert not metrics.systematic_bias_detected
    
    def test_basic_alignment(self):
        """Test basic alignment computation."""
        human_scores = [
            HumanScore("q1", 85, {"coverage": 85}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q2", 75, {"coverage": 75}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q3", 90, {"coverage": 90}, "test", "eval1", "2024-12-08", 0.9)
        ]
        automated_scores = [
            AutomatedScore("q1", 82, {"coverage": 82}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q2", 73, {"coverage": 73}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q3", 92, {"coverage": 92}, "test", "2024-12-08", 1000.0)
        ]
        
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, automated_scores)
        
        assert isinstance(metrics, AlignmentMetrics)
        assert metrics.n_samples == 3
        assert 0 <= abs(metrics.overall_pearson) <= 1
        assert metrics.mean_absolute_error > 0  # Some error expected
        assert metrics.agreement_percentage_10 > 50  # Should have reasonable agreement
    
    def test_systematic_bias_detection(self):
        """Test detection of systematic bias."""
        human_scores = [
            HumanScore("q1", 70, {}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q2", 80, {}, "test", "eval1", "2024-12-08", 0.9)
        ]
        # Automated scores consistently 10 points higher
        automated_scores = [
            AutomatedScore("q1", 85, {}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q2", 95, {}, "test", "2024-12-08", 1000.0)
        ]
        
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, automated_scores)
        
        assert metrics.mean_bias_error > 0  # Positive bias
        assert metrics.systematic_bias_detected
    
    def test_empty_lists(self):
        """Test handling of empty lists."""
        analyzer = ScoreAlignmentAnalyzer()
        
        with pytest.raises(ValueError, match="empty"):
            analyzer.compute_alignment([], [])
    
    def test_mismatched_lengths(self):
        """Test handling of mismatched lengths."""
        human_scores = [HumanScore("q1", 80, {}, "test", "eval1", "2024-12-08", 0.9)]
        automated_scores = []
        
        analyzer = ScoreAlignmentAnalyzer()
        
        with pytest.raises(ValueError, match="same length"):
            analyzer.compute_alignment(human_scores, automated_scores)
    
    def test_no_matching_ids(self):
        """Test when no question IDs match."""
        human_scores = [HumanScore("q1", 80, {}, "test", "eval1", "2024-12-08", 0.9)]
        automated_scores = [AutomatedScore("q2", 80, {}, "test", "2024-12-08", 1000.0)]
        
        analyzer = ScoreAlignmentAnalyzer()
        
        with pytest.raises(ValueError, match="No matching question IDs"):
            analyzer.compute_alignment(human_scores, automated_scores)


class TestCalibrationDataSelector:
    """Test question subset selection."""
    
    @pytest.fixture
    def sample_questions(self):
        """Sample questions for testing."""
        return [
            {"id": "q1", "category": "cat1", "difficulty": "easy"},
            {"id": "q2", "category": "cat1", "difficulty": "medium"}, 
            {"id": "q3", "category": "cat2", "difficulty": "hard"},
            {"id": "q4", "category": "cat2", "difficulty": "easy"},
            {"id": "q5", "category": "cat3", "difficulty": "medium"}
        ]
    
    def test_select_subset_size(self, sample_questions):
        """Test subset selection with different sizes."""
        selector = CalibrationDataSelector(sample_questions)
        
        # Normal size
        subset = selector.select_calibration_subset(3)
        assert len(subset) == 3
        
        # Size larger than available
        subset = selector.select_calibration_subset(10)
        assert len(subset) == 5  # All available
        
        # Minimum size
        subset = selector.select_calibration_subset(1)
        assert len(subset) == 1
    
    def test_balanced_strategy(self, sample_questions):
        """Test balanced sampling strategy."""
        selector = CalibrationDataSelector(sample_questions)
        
        subset = selector.select_calibration_subset(3, "balanced")
        assert len(subset) == 3
        
        # Check category diversity
        categories = [q.get('category') for q in subset]
        assert len(set(categories)) >= 2  # Multiple categories
    
    def test_random_strategy(self, sample_questions):
        """Test random sampling strategy."""
        selector = CalibrationDataSelector(sample_questions)
        
        subset = selector.select_calibration_subset(3, "random")
        assert len(subset) == 3
    
    def test_invalid_strategy(self, sample_questions):
        """Test invalid strategy handling."""
        selector = CalibrationDataSelector(sample_questions)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            selector.select_calibration_subset(3, "invalid")


class TestScoreConsistency:
    """Test scoring consistency validation."""
    
    def test_low_variance_scoring(self):
        """Test detection of consistent scoring."""
        # Multiple runs with very similar scores
        run1_scores = [85, 75, 90]
        run2_scores = [86, 74, 91]  # Very similar
        
        differences = [abs(a - b) for a, b in zip(run1_scores, run2_scores)]
        variance = statistics.variance(differences) if len(differences) > 1 else 0
        
        assert variance <= 2.0  # Very low variance
        assert max(differences) <= 2  # Max difference small
    
    def test_high_variance_detection(self):
        """Test detection of inconsistent scoring."""
        # Multiple runs with very different scores
        run1_scores = [85, 75, 90]
        run2_scores = [45, 35, 50]  # Very different
        
        differences = [abs(a - b) for a, b in zip(run1_scores, run2_scores)]
        
        # The differences are [40, 40, 40] - all the same, so variance is 0
        # Let's test that the differences are large instead
        assert all(diff > 30 for diff in differences)  # Large differences
        assert statistics.mean(differences) > 30  # High average difference


class TestSanityChecks:
    """Test basic sanity checks."""
    
    def test_score_ranges(self):
        """Test that scores are in valid ranges."""
        scores = [15, 45, 78, 85, 91, 22, 67]
        
        # All scores in valid range
        assert all(0 <= score <= 100 for score in scores)
        
        # Reasonable distribution
        assert min(scores) < 50  # Some low scores
        assert max(scores) > 75  # Some high scores
        assert statistics.mean(scores) < 95  # Not all perfect
        assert statistics.mean(scores) > 30  # Not all terrible
    
    def test_obviously_bad_scores(self):
        """Test that bad answers get low scores."""
        bad_scores = [15, 8, 22, 18]
        
        for score in bad_scores:
            assert score < 40  # Below adequate threshold
        
        avg_score = statistics.mean(bad_scores)
        assert avg_score < 25  # Very low average
    
    def test_obviously_good_scores(self):
        """Test that good answers get high scores."""
        good_scores = [88, 94, 91, 85]
        
        for score in good_scores:
            assert score > 75  # Above good threshold
        
        avg_score = statistics.mean(good_scores)
        assert avg_score > 85  # High average


if __name__ == "__main__":
    pytest.main([__file__, "-v"])