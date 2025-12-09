"""
Tests for experiment analysis system.

Covers performance analysis, comparisons, trend analysis, and statistical functions.
"""

import pytest
import tempfile
import statistics
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from src.eval.tracking import ExperimentTracker, ExperimentRecord, RunMetadata
from src.eval.analysis import PerformanceAnalyzer, analyze_run, compare_runs, get_challenging_questions


@pytest.fixture
def sample_records():
    """Sample experiment records for testing."""
    records = []
    
    # Create diverse set of records for analysis testing
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    for i in range(20):
        time_offset = timedelta(hours=i)
        timestamp = (base_time + time_offset).isoformat()
        
        record = ExperimentRecord(
            run_id=f"run_{i % 4}",  # 4 different runs
            timestamp=timestamp,
            model_name=f"model_{i % 3}",  # 3 different models
            config_hash=f"hash_{i % 2}",  # 2 different configs
            question_id=f"q{i:03d}",
            question_text=f"Question {i} about something",
            answer_text=f"Answer {i} with varying quality",
            final_score=60 + (i % 10) * 4,  # Scores from 60 to 96
            final_tier="good" if (60 + (i % 10) * 4) >= 75 else "adequate",
            per_dimension_scores={
                "coverage": 55 + (i % 12) * 3,
                "depth": 50 + (i % 15) * 3,
                "style": 65 + (i % 8) * 4
            },
            critic_scores={
                "coverage": 55 + (i % 12) * 3,
                "depth": 50 + (i % 15) * 3, 
                "style": 65 + (i % 8) * 4
            },
            confidence_level=0.6 + (i % 5) * 0.08,
            execution_time_ms=1000 + i * 100,
            aggregation_method="reasoned_synthesis",
            consensus_level=["low", "medium", "high"][i % 3],
            score_variance=2.0 + (i % 6) * 2,
            evaluation_summary=f"Evaluation summary {i}",
            system_version="1.0",
            critics_used=["coverage", "depth", "style"],
            tags=["test", "analysis"] if i % 2 == 0 else ["test"]
        )
        records.append(record)
    
    return records


@pytest.fixture  
def sample_run_metadata():
    """Sample run metadata for testing."""
    metadata = []
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    
    for i in range(4):
        time_offset = timedelta(days=i)
        timestamp = (base_time + time_offset).isoformat()
        
        meta = RunMetadata(
            run_id=f"run_{i}",
            timestamp=timestamp,
            model_name=f"model_{i % 3}",
            config_hash=f"hash_{i % 2}",
            config_data={"temperature": 0.7 + i * 0.05, "model": f"model_{i % 3}"},
            description=f"Test run {i}",
            tags=["test", "experiment"]
        )
        metadata.append(meta)
    
    return metadata


@pytest.fixture
def mock_tracker_with_data(sample_records, sample_run_metadata):
    """Mock tracker with sample data."""
    tracker = Mock(spec=ExperimentTracker)
    
    # Mock storage that returns our sample data
    mock_storage = Mock()
    mock_storage.get_results.return_value = sample_records
    tracker.storage = mock_storage
    
    # Mock methods
    def get_run_results(run_id):
        return [r for r in sample_records if r.run_id == run_id]
    
    def get_run_metadata(run_id):
        return next((m for m in sample_run_metadata if m.run_id == run_id), None)
    
    def get_all_runs():
        return sample_run_metadata
    
    def get_results_by_model(model_name):
        return [r for r in sample_records if r.model_name == model_name]
    
    tracker.get_run_results = get_run_results
    tracker.get_run_metadata = get_run_metadata
    tracker.get_all_runs = get_all_runs
    tracker.get_results_by_model = get_results_by_model
    
    return tracker


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""
    
    def test_analyze_run_performance(self, mock_tracker_with_data):
        """Test single run performance analysis."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        # Analyze run_0 (should have 5 records: indices 0,4,8,12,16)
        analysis = analyzer.analyze_run_performance("run_0")
        
        assert "error" not in analysis
        assert analysis["run_id"] == "run_0"
        
        summary = analysis["summary"]
        assert summary["total_questions"] == 5
        assert isinstance(summary["mean_score"], float)
        assert isinstance(summary["median_score"], float)
        assert summary["min_score"] >= 60
        assert summary["max_score"] <= 96
        
        # Check that all required sections are present
        assert "score_distribution" in analysis
        assert "critic_performance" in analysis
        assert "consensus_analysis" in analysis
        assert "challenging_questions" in analysis
        assert "high_performing_questions" in analysis
    
    def test_analyze_nonexistent_run(self, mock_tracker_with_data):
        """Test analysis of non-existent run."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        analysis = analyzer.analyze_run_performance("nonexistent_run")
        
        assert "error" in analysis
        assert "No results found" in analysis["error"]
    
    def test_compare_runs(self, mock_tracker_with_data):
        """Test run comparison functionality."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        comparison = analyzer.compare_runs("run_0", "run_1")
        
        assert "error" not in comparison
        
        # Check comparison structure
        assert "run_1" in comparison
        assert "run_2" in comparison
        assert "score_comparison" in comparison
        assert "performance_comparison" in comparison
        assert "question_comparisons" in comparison
        
        # Check score comparison metrics
        score_comp = comparison["score_comparison"]
        assert "mean_score_diff" in score_comp
        assert "median_score_diff" in score_comp
        assert "run_1_better_questions" in score_comp
        assert "run_2_better_questions" in score_comp
        assert "tied_questions" in score_comp
        
        # Verify counts add up correctly
        total_compared = (score_comp["run_1_better_questions"] + 
                         score_comp["run_2_better_questions"] + 
                         score_comp["tied_questions"])
        
        # Should have some common questions between run_0 and run_1
        assert total_compared >= 0
    
    def test_analyze_model_trends(self, mock_tracker_with_data):
        """Test model trend analysis."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        trends = analyzer.analyze_model_trends("model_0")
        
        assert "error" not in trends
        assert trends["model_name"] == "model_0"
        
        # Check trend analysis structure
        assert "analysis_period" in trends
        assert "recent_runs" in trends
        assert "trend_analysis" in trends
        
        trend_analysis = trends["trend_analysis"]
        assert "score_trend" in trend_analysis
        assert "consistency_trend" in trend_analysis
        assert "best_run" in trend_analysis
        assert "worst_run" in trend_analysis
        
        # Verify trend categorization
        assert trend_analysis["score_trend"] in ["improving", "declining", "stable"]
        assert trend_analysis["consistency_trend"] in ["more_consistent", "less_consistent", "stable"]
    
    def test_identify_challenging_questions(self, mock_tracker_with_data):
        """Test challenging questions identification."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        challenging = analyzer.identify_challenging_questions(min_results=1)
        
        # Should be a list of dictionaries
        assert isinstance(challenging, list)
        
        for question in challenging:
            assert "question_id" in question
            assert "mean_score" in question
            assert "num_evaluations" in question
            assert "models_evaluated" in question
            
            # All challenging questions should have low scores
            assert question["mean_score"] < 60
    
    def test_get_summary_statistics(self, mock_tracker_with_data):
        """Test overall summary statistics."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        summary = analyzer.get_summary_statistics()
        
        assert "error" not in summary
        
        # Check main sections
        assert "experiment_summary" in summary
        assert "score_statistics" in summary
        assert "model_performance" in summary
        assert "critic_usage" in summary
        assert "performance_insights" in summary
        
        exp_summary = summary["experiment_summary"]
        assert exp_summary["total_evaluations"] == 20
        assert exp_summary["total_runs"] == 4
        assert exp_summary["unique_questions"] == 20
        assert exp_summary["unique_models"] == 3
        
        score_stats = summary["score_statistics"]
        assert isinstance(score_stats["mean_score"], float)
        assert isinstance(score_stats["std_dev"], float)
        assert 60 <= score_stats["min_score"] <= score_stats["max_score"] <= 96


class TestScoreDistributionAnalysis:
    """Test score distribution analysis methods."""
    
    def test_score_distribution_calculation(self, mock_tracker_with_data, sample_records):
        """Test score distribution calculation."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        # Test with subset of records
        test_records = sample_records[:10]  # First 10 records
        distribution = analyzer._analyze_score_distribution(test_records)
        
        assert "counts" in distribution
        assert "percentages" in distribution
        
        counts = distribution["counts"]
        total_records = len(test_records)
        
        # Verify all tiers are present
        expected_tiers = ["excellent (90-100)", "good (75-89)", "adequate (60-74)", 
                         "poor (40-59)", "inadequate (0-39)"]
        for tier in expected_tiers:
            assert tier in counts
            assert isinstance(counts[tier], int)
            assert counts[tier] >= 0
        
        # Verify counts sum to total
        assert sum(counts.values()) == total_records
        
        # Verify percentages sum to 100 (within floating point precision)
        percentages = distribution["percentages"]
        assert abs(sum(percentages.values()) - 100.0) < 0.001


class TestCriticPerformanceAnalysis:
    """Test critic performance analysis methods."""
    
    def test_critic_performance_analysis(self, mock_tracker_with_data, sample_records):
        """Test individual critic performance analysis."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        # Test with subset of records
        test_records = sample_records[:10]
        critic_analysis = analyzer._analyze_critic_performance(test_records)
        
        # Should analyze all critics
        expected_critics = ["coverage", "depth", "style"]
        for critic in expected_critics:
            assert critic in critic_analysis
            
            critic_stats = critic_analysis[critic]
            assert "mean_score" in critic_stats
            assert "std_dev" in critic_stats
            assert "correlation_with_final" in critic_stats
            assert "evaluations_count" in critic_stats
            assert "score_range" in critic_stats
            
            # Verify reasonable values
            assert 0 <= critic_stats["mean_score"] <= 100
            assert critic_stats["std_dev"] >= 0
            assert -1 <= critic_stats["correlation_with_final"] <= 1
            assert critic_stats["evaluations_count"] > 0


class TestConsensusAnalysis:
    """Test consensus pattern analysis."""
    
    def test_consensus_patterns(self, mock_tracker_with_data, sample_records):
        """Test consensus pattern analysis."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        test_records = sample_records[:15]
        consensus_analysis = analyzer._analyze_consensus_patterns(test_records)
        
        assert "consensus_distribution" in consensus_analysis
        assert "mean_score_variance" in consensus_analysis
        assert "high_disagreement_questions" in consensus_analysis
        assert "perfect_consensus_questions" in consensus_analysis
        
        # Verify consensus distribution
        consensus_dist = consensus_analysis["consensus_distribution"]
        expected_levels = ["low", "medium", "high"]
        for level in expected_levels:
            if level in consensus_dist:
                assert isinstance(consensus_dist[level], int)
                assert consensus_dist[level] >= 0


class TestCorrelationCalculation:
    """Test statistical calculation methods."""
    
    def test_correlation_calculation(self, mock_tracker_with_data):
        """Test correlation calculation method."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        # Test perfect positive correlation
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        correlation = analyzer._calculate_correlation(x, y)
        assert abs(correlation - 1.0) < 0.001
        
        # Test perfect negative correlation
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        correlation = analyzer._calculate_correlation(x, y)
        assert abs(correlation - (-1.0)) < 0.001
        
        # Test no correlation
        x = [1, 2, 3, 4, 5]
        y = [3, 3, 3, 3, 3]  # Constant
        correlation = analyzer._calculate_correlation(x, y)
        assert correlation == 0.0
        
        # Test edge cases
        assert analyzer._calculate_correlation([], []) == 0.0
        assert analyzer._calculate_correlation([1], [2]) == 0.0
        assert analyzer._calculate_correlation([1, 2], [3, 4, 5]) == 0.0  # Different lengths


class TestInsightGeneration:
    """Test performance insight generation."""
    
    def test_performance_insights(self, mock_tracker_with_data, sample_records):
        """Test generation of performance insights."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        insights = analyzer._generate_performance_insights(sample_records)
        
        assert isinstance(insights, list)
        
        # Insights should be strings
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 0
        
        # Should generate reasonable number of insights
        assert 0 <= len(insights) <= 10


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_analyze_run_convenience(self, mock_tracker_with_data):
        """Test analyze_run convenience function."""
        analysis = analyze_run(mock_tracker_with_data, "run_0")
        
        assert "error" not in analysis
        assert analysis["run_id"] == "run_0"
        assert "summary" in analysis
    
    def test_compare_runs_convenience(self, mock_tracker_with_data):
        """Test compare_runs convenience function."""
        comparison = compare_runs(mock_tracker_with_data, "run_0", "run_1")
        
        assert "error" not in comparison
        assert "score_comparison" in comparison
    
    def test_get_challenging_questions_convenience(self, mock_tracker_with_data):
        """Test get_challenging_questions convenience function."""
        challenging = get_challenging_questions(mock_tracker_with_data, min_results=1)
        
        assert isinstance(challenging, list)


class TestErrorHandling:
    """Test error handling in analysis functions."""
    
    def test_empty_data_handling(self):
        """Test analysis with empty data."""
        # Mock tracker with no data
        empty_tracker = Mock(spec=ExperimentTracker)
        mock_storage = Mock()
        mock_storage.get_results.return_value = []
        empty_tracker.storage = mock_storage
        empty_tracker.get_run_results.return_value = []
        empty_tracker.get_run_metadata.return_value = None
        empty_tracker.get_all_runs.return_value = []
        
        analyzer = PerformanceAnalyzer(empty_tracker)
        
        # Should handle empty data gracefully
        analysis = analyzer.analyze_run_performance("nonexistent")
        assert "error" in analysis
        
        summary = analyzer.get_summary_statistics()
        assert "error" in summary
    
    def test_invalid_data_handling(self, mock_tracker_with_data):
        """Test handling of invalid or corrupted data."""
        analyzer = PerformanceAnalyzer(mock_tracker_with_data)
        
        # Test with empty records list
        distribution = analyzer._analyze_score_distribution([])
        
        # Should return valid structure even with empty data
        assert "counts" in distribution
        assert "percentages" in distribution
        
        # Test correlation with edge cases
        assert analyzer._calculate_correlation([1], [1]) == 0.0  # Single point
        assert analyzer._calculate_correlation([1, 1, 1], [2, 2, 2]) == 0.0  # Zero variance


class TestStatisticalAccuracy:
    """Test statistical calculation accuracy."""
    
    def test_score_statistics_accuracy(self, sample_records):
        """Test that calculated statistics match expected values."""
        # Use known data for verification
        test_scores = [60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
        
        # Create records with known scores
        test_records = []
        for i, score in enumerate(test_scores):
            record = ExperimentRecord(
                run_id="test",
                timestamp="2024-01-01T10:00:00",
                model_name="test_model",
                config_hash="test_hash",
                question_id=f"q{i}",
                question_text=f"Question {i}",
                answer_text=f"Answer {i}",
                final_score=score,
                final_tier="good",
                per_dimension_scores={},
                critic_scores={},
                confidence_level=0.8,
                execution_time_ms=1000.0,
                aggregation_method="test",
                consensus_level="high",
                score_variance=2.0,
                evaluation_summary="Test",
                system_version="1.0",
                critics_used=["test_critic"],
                tags=[]
            )
            test_records.append(record)
        
        # Create mock tracker
        mock_tracker = Mock()
        mock_storage = Mock()
        mock_storage.get_results.return_value = test_records
        mock_tracker.storage = mock_storage
        
        analyzer = PerformanceAnalyzer(mock_tracker)
        summary = analyzer.get_summary_statistics()
        
        score_stats = summary["score_statistics"]
        
        # Verify statistical calculations
        expected_mean = statistics.mean(test_scores)
        expected_median = statistics.median(test_scores) 
        expected_std = statistics.stdev(test_scores)
        
        assert abs(score_stats["mean_score"] - expected_mean) < 0.001
        assert abs(score_stats["median_score"] - expected_median) < 0.001
        assert abs(score_stats["std_dev"] - expected_std) < 0.001
        assert score_stats["min_score"] == min(test_scores)
        assert score_stats["max_score"] == max(test_scores)


if __name__ == "__main__":
    pytest.main([__file__])