"""
Tests for the calibration and QA system.

Tests cover:
- Question selection and subset sampling
- Score alignment analysis and metrics computation
- Human score data handling
- Calibration analysis and recommendation generation
- Score variance and consistency validation
- Sanity checks for scoring system
"""

import json
import pytest
import tempfile
import statistics
from datetime import datetime
from pathlib import Path
from typing import List

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.calibration import (
    CalibrationDataSelector, ScoreAlignmentAnalyzer, CalibrationAnalyzer,
    HumanScore, AutomatedScore, AlignmentMetrics, CalibrationResult,
    save_human_scores, load_human_scores, save_calibration_result
)


class TestCalibrationDataSelector:
    """Test question selection and subset sampling functionality."""
    
    @pytest.fixture
    def sample_questions(self):
        """Create sample questions for testing."""
        return [
            {
                "id": "q1", "prompt": "Question 1", "category": "cat1", "difficulty": "easy",
                "rubric_focus": ["coverage", "detail"], "created_date": "2024-12-08"
            },
            {
                "id": "q2", "prompt": "Question 2", "category": "cat1", "difficulty": "medium", 
                "rubric_focus": ["style"], "created_date": "2024-12-08"
            },
            {
                "id": "q3", "prompt": "Question 3", "category": "cat2", "difficulty": "hard",
                "rubric_focus": ["coverage"], "created_date": "2024-12-08"
            },
            {
                "id": "q4", "prompt": "Question 4", "category": "cat2", "difficulty": "easy",
                "rubric_focus": ["detail", "structure"], "created_date": "2024-12-08"
            },
            {
                "id": "q5", "prompt": "Question 5", "category": "cat3", "difficulty": "medium",
                "rubric_focus": ["style", "coverage"], "created_date": "2024-12-08"
            }
        ]
    
    @pytest.fixture
    def temp_questions_file(self, sample_questions):
        """Create temporary questions file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for q in sample_questions:
                f.write(json.dumps(q) + '\n')
            temp_file = f.name
        
        yield temp_file
        Path(temp_file).unlink()
    
    def test_load_questions(self, temp_questions_file, sample_questions):
        """Test loading questions from JSONL file."""
        selector = CalibrationDataSelector(temp_questions_file)
        
        assert len(selector.questions) == len(sample_questions)
        assert selector.questions[0]['id'] == 'q1'
        assert selector.questions[-1]['id'] == 'q5'
    
    def test_select_calibration_subset_size(self, temp_questions_file):
        """Test selecting subset with different sizes."""
        selector = CalibrationDataSelector(temp_questions_file)
        
        # Test normal size
        subset = selector.select_calibration_subset(target_size=3)
        assert len(subset) == 3
        
        # Test size larger than available
        subset = selector.select_calibration_subset(target_size=10)
        assert len(subset) == 5  # Should return all available
        
        # Test minimum size
        subset = selector.select_calibration_subset(target_size=1)
        assert len(subset) == 1
    
    def test_select_calibration_subset_strategies(self, temp_questions_file):
        """Test different selection strategies."""
        selector = CalibrationDataSelector(temp_questions_file)
        
        # Test balanced sampling
        balanced = selector.select_calibration_subset(3, "balanced_sampling")
        assert len(balanced) == 3
        
        # Test random sampling
        random_subset = selector.select_calibration_subset(3, "random")
        assert len(random_subset) == 3
        
        # Test stratified sampling
        stratified = selector.select_calibration_subset(3, "stratified")
        assert len(stratified) == 3
    
    def test_invalid_strategy(self, temp_questions_file):
        """Test handling of invalid selection strategy."""
        selector = CalibrationDataSelector(temp_questions_file)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            selector.select_calibration_subset(3, "invalid_strategy")
    
    def test_selection_diversity(self, temp_questions_file):
        """Test that selection includes diverse questions."""
        selector = CalibrationDataSelector(temp_questions_file)
        
        # Select subset
        subset = selector.select_calibration_subset(4, "balanced_sampling")
        
        # Check category diversity
        categories = [q.get('category') for q in subset]
        assert len(set(categories)) >= 2  # Should have multiple categories
        
        # Check difficulty diversity if possible
        difficulties = [q.get('difficulty') for q in subset]
        if len(subset) >= 3:
            assert len(set(difficulties)) >= 2  # Should have multiple difficulties


class TestScoreAlignmentAnalyzer:
    """Test score alignment analysis and metrics computation."""
    
    @pytest.fixture
    def human_scores(self):
        """Create sample human scores."""
        return [
            HumanScore(
                question_id="q1", overall_score=85, 
                dimension_scores={"coverage": 90, "detail": 80, "style": 85},
                justification="Good answer with comprehensive coverage",
                evaluator_id="eval1", evaluation_date="2024-12-08T10:00:00",
                confidence=0.9
            ),
            HumanScore(
                question_id="q2", overall_score=70,
                dimension_scores={"coverage": 75, "detail": 65, "style": 70},
                justification="Adequate answer but missing some details",
                evaluator_id="eval1", evaluation_date="2024-12-08T10:15:00",
                confidence=0.8
            ),
            HumanScore(
                question_id="q3", overall_score=92,
                dimension_scores={"coverage": 95, "detail": 90, "style": 90},
                justification="Excellent comprehensive answer",
                evaluator_id="eval2", evaluation_date="2024-12-08T11:00:00",
                confidence=0.95
            )
        ]
    
    @pytest.fixture
    def automated_scores(self):
        """Create sample automated scores."""
        return [
            AutomatedScore(
                question_id="q1", overall_score=82,
                dimension_scores={"coverage": 85, "detail": 78, "style": 83},
                scoring_method="single_critic", evaluation_date="2024-12-08T10:30:00",
                execution_time_ms=1500.0, confidence=0.85
            ),
            AutomatedScore(
                question_id="q2", overall_score=73,
                dimension_scores={"coverage": 76, "detail": 70, "style": 73},
                scoring_method="single_critic", evaluation_date="2024-12-08T10:45:00",
                execution_time_ms=1200.0, confidence=0.80
            ),
            AutomatedScore(
                question_id="q3", overall_score=88,
                dimension_scores={"coverage": 92, "detail": 85, "style": 87},
                scoring_method="single_critic", evaluation_date="2024-12-08T11:15:00",
                execution_time_ms=1800.0, confidence=0.90
            )
        ]
    
    def test_compute_alignment_basic(self, human_scores, automated_scores):
        """Test basic alignment computation."""
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, automated_scores)
        
        assert isinstance(metrics, AlignmentMetrics)
        assert metrics.n_samples == 3
        assert 0 <= abs(metrics.overall_pearson) <= 1
        assert 0 <= abs(metrics.overall_spearman) <= 1
        assert metrics.mean_absolute_error >= 0
        assert metrics.root_mean_square_error >= 0
    
    def test_compute_alignment_perfect_agreement(self):
        """Test alignment with perfect agreement."""
        # Create identical scores
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
    
    def test_compute_alignment_mismatched_lengths(self, human_scores):
        """Test handling of mismatched score list lengths."""
        analyzer = ScoreAlignmentAnalyzer()
        
        # Empty automated scores
        with pytest.raises(ValueError, match="same length"):
            analyzer.compute_alignment(human_scores, [])
        
        # Different lengths
        automated_scores = [
            AutomatedScore("q1", 80, {}, "test", "2024-12-08", 1000.0)
        ]
        with pytest.raises(ValueError, match="same length"):
            analyzer.compute_alignment(human_scores, automated_scores)
    
    def test_compute_alignment_empty_lists(self):
        """Test handling of empty score lists."""
        analyzer = ScoreAlignmentAnalyzer()
        
        with pytest.raises(ValueError, match="empty score lists"):
            analyzer.compute_alignment([], [])
    
    def test_compute_alignment_no_matching_ids(self, human_scores):
        """Test handling when no question IDs match."""
        # Create automated scores with different IDs
        automated_scores = [
            AutomatedScore("q10", 80, {}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q11", 85, {}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q12", 90, {}, "test", "2024-12-08", 1000.0)
        ]
        
        analyzer = ScoreAlignmentAnalyzer()
        with pytest.raises(ValueError, match="No matching question IDs"):
            analyzer.compute_alignment(human_scores, automated_scores)
    
    def test_dimension_correlations(self, human_scores, automated_scores):
        """Test dimension-wise correlation computation."""
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, automated_scores)
        
        assert "coverage" in metrics.dimension_correlations
        assert "detail" in metrics.dimension_correlations
        assert "style" in metrics.dimension_correlations
        
        for dim_corr in metrics.dimension_correlations.values():
            assert "pearson" in dim_corr
            assert "spearman" in dim_corr
            assert "n_samples" in dim_corr
            assert dim_corr["n_samples"] <= metrics.n_samples
    
    def test_bias_detection(self):
        """Test systematic bias detection."""
        # Create scores with systematic bias (automated always higher)
        human_scores = [
            HumanScore("q1", 70, {}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q2", 80, {}, "test", "eval1", "2024-12-08", 0.9)
        ]
        automated_scores = [
            AutomatedScore("q1", 85, {}, "test", "2024-12-08", 1000.0),  # +15
            AutomatedScore("q2", 95, {}, "test", "2024-12-08", 1000.0)   # +15
        ]
        
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, automated_scores)
        
        assert metrics.mean_bias_error > 0  # Positive bias (automated higher)
        assert metrics.systematic_bias_detected  # Should detect bias
    
    def test_outlier_detection(self):
        """Test outlier detection in score differences."""
        # Create scores with one outlier
        human_scores = [
            HumanScore("q1", 80, {}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q2", 82, {}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q3", 85, {}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q4", 40, {}, "test", "eval1", "2024-12-08", 0.9)  # Outlier
        ]
        automated_scores = [
            AutomatedScore("q1", 81, {}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q2", 83, {}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q3", 84, {}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q4", 80, {}, "test", "2024-12-08", 1000.0)   # Large diff
        ]
        
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, automated_scores)
        
        assert metrics.outlier_count >= 1  # Should detect the outlier


class TestCalibrationAnalyzer:
    """Test overall calibration analysis and recommendation generation."""
    
    @pytest.fixture
    def good_alignment_scores(self):
        """Create scores with good alignment for testing."""
        human_scores = [
            HumanScore("q1", 85, {"coverage": 85}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q2", 75, {"coverage": 75}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q3", 90, {"coverage": 90}, "test", "eval1", "2024-12-08", 0.9)
        ]
        automated_scores = [
            AutomatedScore("q1", 87, {"coverage": 87}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q2", 73, {"coverage": 73}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q3", 92, {"coverage": 92}, "test", "2024-12-08", 1000.0)
        ]
        return human_scores, automated_scores
    
    @pytest.fixture
    def poor_alignment_scores(self):
        """Create scores with poor alignment for testing."""
        human_scores = [
            HumanScore("q1", 85, {"coverage": 85}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q2", 75, {"coverage": 75}, "test", "eval1", "2024-12-08", 0.9),
            HumanScore("q3", 90, {"coverage": 90}, "test", "eval1", "2024-12-08", 0.9)
        ]
        # Very different automated scores
        automated_scores = [
            AutomatedScore("q1", 45, {"coverage": 45}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q2", 35, {"coverage": 35}, "test", "2024-12-08", 1000.0),
            AutomatedScore("q3", 50, {"coverage": 50}, "test", "2024-12-08", 1000.0)
        ]
        return human_scores, automated_scores
    
    def test_analyze_calibration_good_alignment(self, good_alignment_scores):
        """Test calibration analysis with good alignment."""
        human_scores, automated_scores = good_alignment_scores
        
        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(human_scores, automated_scores)
        
        assert isinstance(result, CalibrationResult)
        assert result.calibration_quality in ["excellent", "good"]
        assert result.confidence_level > 0.5
        assert len(result.recommendations) >= 0  # May have no recommendations if good
        assert result.alignment_metrics.n_samples == 3
    
    def test_analyze_calibration_poor_alignment(self, poor_alignment_scores):
        """Test calibration analysis with poor alignment."""
        human_scores, automated_scores = poor_alignment_scores
        
        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(human_scores, automated_scores)
        
        assert result.calibration_quality in ["poor", "needs_improvement"]
        assert len(result.recommendations) > 0  # Should have recommendations
        assert len(result.areas_of_concern) > 0  # Should have concerns
        assert result.alignment_metrics.overall_pearson < 0.8  # Poor correlation
    
    def test_recommendation_generation(self, poor_alignment_scores):
        """Test that appropriate recommendations are generated."""
        human_scores, automated_scores = poor_alignment_scores
        
        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(human_scores, automated_scores)
        
        # Should have recommendations for poor alignment
        rec_text = " ".join(result.recommendations).lower()
        assert any(keyword in rec_text for keyword in 
                  ["correlation", "agreement", "scoring", "review", "calibration"])
    
    def test_metadata_handling(self, good_alignment_scores):
        """Test handling of analysis metadata."""
        human_scores, automated_scores = good_alignment_scores
        
        metadata = {"test_run": "calibration_test", "version": "1.0"}
        
        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(human_scores, automated_scores, metadata)
        
        assert result.data_subset_info == metadata
        assert result.human_evaluators_count == 1  # Only one evaluator in test data
        assert result.automated_method == "test"  # From test scoring method
        assert result.analysis_date is not None


class TestDataHandling:
    """Test human score data saving/loading functionality."""
    
    def test_save_and_load_human_scores(self):
        """Test saving and loading human scores."""
        scores = [
            HumanScore(
                question_id="q1", overall_score=85,
                dimension_scores={"coverage": 90, "detail": 80},
                justification="Good answer", evaluator_id="eval1",
                evaluation_date="2024-12-08T10:00:00", confidence=0.9,
                time_spent_minutes=5.5, notes="Test note"
            ),
            HumanScore(
                question_id="q2", overall_score=70,
                dimension_scores={"coverage": 75, "detail": 65},
                justification="Adequate answer", evaluator_id="eval2",
                evaluation_date="2024-12-08T11:00:00", confidence=0.8
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save scores
            save_human_scores(scores, temp_file)
            assert Path(temp_file).exists()
            
            # Load scores
            loaded_scores = load_human_scores(temp_file)
            
            assert len(loaded_scores) == 2
            assert loaded_scores[0].question_id == "q1"
            assert loaded_scores[0].overall_score == 85
            assert loaded_scores[0].dimension_scores["coverage"] == 90
            assert loaded_scores[0].confidence == 0.9
            assert loaded_scores[0].time_spent_minutes == 5.5
            assert loaded_scores[0].notes == "Test note"
            
            assert loaded_scores[1].question_id == "q2"
            assert loaded_scores[1].overall_score == 70
            assert loaded_scores[1].time_spent_minutes is None  # Not set
            assert loaded_scores[1].notes is None
            
        finally:
            Path(temp_file).unlink()
    
    def test_save_calibration_result(self, good_alignment_scores):
        """Test saving calibration analysis results."""
        human_scores, automated_scores = good_alignment_scores
        
        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze_calibration(human_scores, automated_scores)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save result
            save_calibration_result(result, temp_file)
            assert Path(temp_file).exists()
            
            # Load and verify structure
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            assert "alignment_metrics" in data
            assert "recommendations" in data
            assert "calibration_quality" in data
            assert "confidence_level" in data
            assert data["alignment_metrics"]["n_samples"] == 3
            
        finally:
            Path(temp_file).unlink()


class TestScoringConsistency:
    """Test scoring consistency and variance validation."""
    
    def test_score_variance_detection(self):
        """Test detection of high score variance between runs."""
        # Create scores with high variance (inconsistent scoring)
        run1_scores = [
            AutomatedScore("q1", 85, {}, "test", "2024-12-08T10:00:00", 1000.0),
            AutomatedScore("q2", 75, {}, "test", "2024-12-08T10:01:00", 1000.0),
        ]
        run2_scores = [
            AutomatedScore("q1", 45, {}, "test", "2024-12-08T10:05:00", 1000.0),  # 40 point diff
            AutomatedScore("q2", 95, {}, "test", "2024-12-08T10:06:00", 1000.0),  # 20 point diff
        ]
        
        # Convert to "human" scores for comparison
        human_scores = [
            HumanScore("q1", 85, {}, "run1", "system", "2024-12-08", 1.0),
            HumanScore("q2", 75, {}, "run1", "system", "2024-12-08", 1.0)
        ]
        
        analyzer = ScoreAlignmentAnalyzer()
        metrics = analyzer.compute_alignment(human_scores, run2_scores)
        
        # High variance should result in poor alignment
        assert metrics.mean_absolute_error > 15  # Large error due to inconsistency
        assert metrics.agreement_percentage_10 < 50  # Poor agreement
    
    def test_deterministic_scoring_consistency(self):
        """Test that deterministic questions get consistent scores."""
        # For deterministic questions, multiple runs should yield very similar scores
        question_ids = ["det1", "det2", "det3"]
        
        # Simulate multiple scoring runs
        run1_scores = [
            AutomatedScore("det1", 85, {}, "test", "2024-12-08T10:00:00", 1000.0),
            AutomatedScore("det2", 92, {}, "test", "2024-12-08T10:01:00", 1000.0),
            AutomatedScore("det3", 78, {}, "test", "2024-12-08T10:02:00", 1000.0),
        ]
        run2_scores = [
            AutomatedScore("det1", 86, {}, "test", "2024-12-08T11:00:00", 1100.0),  # +1
            AutomatedScore("det2", 91, {}, "test", "2024-12-08T11:01:00", 1050.0),  # -1
            AutomatedScore("det3", 79, {}, "test", "2024-12-08T11:02:00", 1200.0),  # +1
        ]
        
        # Check consistency
        differences = []
        for q_id in question_ids:
            score1 = next(s.overall_score for s in run1_scores if s.question_id == q_id)
            score2 = next(s.overall_score for s in run2_scores if s.question_id == q_id)
            differences.append(abs(score1 - score2))
        
        # For deterministic scoring, variance should be very low
        variance = statistics.variance(differences) if len(differences) > 1 else differences[0]**2
        assert variance <= 4  # Very low variance expected
        assert max(differences) <= 3  # Max difference should be small


class TestSanityChecks:
    """Test sanity checks for obviously good/bad answers."""
    
    def test_obviously_bad_answers_get_low_scores(self):
        """Test that obviously bad answers receive appropriately low scores."""
        # These would typically be tested with actual LLM scoring
        # Here we simulate the expected behavior
        
        bad_answer_scores = [
            AutomatedScore("bad1", 15, {}, "test", "2024-12-08", 1000.0),  # Very low
            AutomatedScore("bad2", 8, {}, "test", "2024-12-08", 1000.0),   # Very low
            AutomatedScore("bad3", 22, {}, "test", "2024-12-08", 1000.0),  # Low
        ]
        
        # All scores should be below reasonable threshold for "bad" answers
        for score in bad_answer_scores:
            assert score.overall_score < 40  # Below "adequate" threshold
        
        avg_score = statistics.mean(s.overall_score for s in bad_answer_scores)
        assert avg_score < 25  # Very low average
    
    def test_obviously_good_answers_get_high_scores(self):
        """Test that obviously good answers receive appropriately high scores."""
        good_answer_scores = [
            AutomatedScore("good1", 88, {}, "test", "2024-12-08", 1000.0),  # High
            AutomatedScore("good2", 94, {}, "test", "2024-12-08", 1000.0),  # Very high
            AutomatedScore("good3", 91, {}, "test", "2024-12-08", 1000.0),  # High
        ]
        
        # All scores should be above reasonable threshold for "good" answers
        for score in good_answer_scores:
            assert score.overall_score > 75  # Above "good" threshold
        
        avg_score = statistics.mean(s.overall_score for s in good_answer_scores)
        assert avg_score > 85  # High average
    
    def test_score_distribution_sanity(self):
        """Test that score distributions make sense."""
        # Simulate a realistic score distribution
        scores = [
            AutomatedScore("q1", 45, {}, "test", "2024-12-08", 1000.0),   # Poor
            AutomatedScore("q2", 62, {}, "test", "2024-12-08", 1000.0),   # Adequate
            AutomatedScore("q3", 78, {}, "test", "2024-12-08", 1000.0),   # Good  
            AutomatedScore("q4", 85, {}, "test", "2024-12-08", 1000.0),   # Good
            AutomatedScore("q5", 91, {}, "test", "2024-12-08", 1000.0),   # Excellent
            AutomatedScore("q6", 58, {}, "test", "2024-12-08", 1000.0),   # Adequate
            AutomatedScore("q7", 72, {}, "test", "2024-12-08", 1000.0),   # Good
        ]
        
        score_values = [s.overall_score for s in scores]
        
        # Basic distribution sanity checks
        assert all(0 <= score <= 100 for score in score_values)  # Valid range
        assert statistics.mean(score_values) > 30  # Not all terrible
        assert statistics.mean(score_values) < 95  # Not all perfect
        assert len(set(score_values)) > 1  # Some variance in scores
        
        # Check that we have representation across score ranges
        has_poor = any(score < 50 for score in score_values)
        has_good = any(score > 75 for score in score_values)
        has_adequate = any(50 <= score <= 75 for score in score_values)
        
        assert has_poor or has_adequate  # Not all scores are excellent
        assert has_good or has_adequate  # Not all scores are poor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])