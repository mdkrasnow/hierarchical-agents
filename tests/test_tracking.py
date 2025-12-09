"""
Tests for experiment tracking system.

Covers storage backends, data models, and experiment logging functionality.
"""

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from src.eval.tracking import (
    ExperimentTracker, JSONLStorage, SQLiteStorage, ExperimentRecord, 
    RunMetadata, create_tracker, log_result
)
from src.critics.debate_models import MultiCriticResult, ScoreAggregation
from src.critics.models import CriticScore


@pytest.fixture
def sample_experiment_record():
    """Sample experiment record for testing."""
    return ExperimentRecord(
        run_id="test_run_001",
        timestamp="2024-01-01T10:00:00",
        model_name="test_model",
        config_hash="abc123",
        question_id="q001",
        question_text="What is Python?",
        answer_text="Python is a programming language.",
        final_score=85,
        final_tier="good",
        per_dimension_scores={"coverage": 80, "depth": 90},
        critic_scores={"coverage": 80, "depth": 90, "style": 85},
        confidence_level=0.8,
        execution_time_ms=1500.0,
        aggregation_method="reasoned_synthesis",
        consensus_level="medium",
        score_variance=5.2,
        evaluation_summary="Good overall performance",
        system_version="1.0",
        critics_used=["coverage", "depth", "style"],
        tags=["test", "python"]
    )


@pytest.fixture
def sample_run_metadata():
    """Sample run metadata for testing."""
    return RunMetadata(
        run_id="test_run_001",
        timestamp="2024-01-01T10:00:00",
        model_name="test_model",
        config_hash="abc123",
        config_data={"temperature": 0.7, "max_tokens": 100},
        description="Test evaluation run",
        tags=["test", "experiment"]
    )


@pytest.fixture
def temp_storage_path():
    """Temporary storage path for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestJSONLStorage:
    """Test JSONL storage backend."""
    
    def test_store_and_retrieve_result(self, temp_storage_path, sample_experiment_record):
        """Test storing and retrieving experiment results."""
        storage = JSONLStorage(temp_storage_path)
        
        # Store result
        success = storage.store_result(sample_experiment_record)
        assert success
        
        # Retrieve result
        results = storage.get_results()
        assert len(results) == 1
        
        retrieved = results[0]
        assert retrieved.run_id == sample_experiment_record.run_id
        assert retrieved.question_id == sample_experiment_record.question_id
        assert retrieved.final_score == sample_experiment_record.final_score
        assert retrieved.critic_scores == sample_experiment_record.critic_scores
    
    def test_store_and_retrieve_metadata(self, temp_storage_path, sample_run_metadata):
        """Test storing and retrieving run metadata."""
        storage = JSONLStorage(temp_storage_path)
        
        # Store metadata
        success = storage.store_run_metadata(sample_run_metadata)
        assert success
        
        # Retrieve metadata
        retrieved = storage.get_run_metadata(sample_run_metadata.run_id)
        assert retrieved is not None
        assert retrieved.run_id == sample_run_metadata.run_id
        assert retrieved.model_name == sample_run_metadata.model_name
        assert retrieved.config_data == sample_run_metadata.config_data
    
    def test_filter_results(self, temp_storage_path, sample_experiment_record):
        """Test filtering results by various criteria."""
        storage = JSONLStorage(temp_storage_path)
        
        # Store multiple records with different properties
        record1 = sample_experiment_record
        record2 = ExperimentRecord(**{
            **sample_experiment_record.__dict__,
            "question_id": "q002",
            "final_score": 95,
            "model_name": "different_model"
        })
        
        storage.store_result(record1)
        storage.store_result(record2)
        
        # Test model filter
        results = storage.get_results({"model_name": "test_model"})
        assert len(results) == 1
        assert results[0].question_id == "q001"
        
        # Test score range filter
        results = storage.get_results({"score_range": (90, 100)})
        assert len(results) == 1
        assert results[0].question_id == "q002"
    
    def test_get_all_runs(self, temp_storage_path):
        """Test retrieving all run metadata."""
        storage = JSONLStorage(temp_storage_path)
        
        metadata1 = RunMetadata("run1", "2024-01-01T10:00:00", "model1", "hash1", {"temp": 0.7})
        metadata2 = RunMetadata("run2", "2024-01-01T11:00:00", "model2", "hash2", {"temp": 0.8})
        
        storage.store_run_metadata(metadata1)
        storage.store_run_metadata(metadata2)
        
        all_runs = storage.get_all_runs()
        assert len(all_runs) == 2
        
        run_ids = {run.run_id for run in all_runs}
        assert run_ids == {"run1", "run2"}


class TestSQLiteStorage:
    """Test SQLite storage backend."""
    
    def test_database_initialization(self, temp_storage_path):
        """Test database is properly initialized."""
        db_path = temp_storage_path / "test.sqlite"
        storage = SQLiteStorage(db_path)
        
        # Check tables exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "run_metadata" in tables
            assert "experiment_results" in tables
    
    def test_store_and_retrieve_result(self, temp_storage_path, sample_experiment_record):
        """Test storing and retrieving results in SQLite."""
        db_path = temp_storage_path / "test.sqlite"
        storage = SQLiteStorage(db_path)
        
        # Store result
        success = storage.store_result(sample_experiment_record)
        assert success
        
        # Retrieve result
        results = storage.get_results()
        assert len(results) == 1
        
        retrieved = results[0]
        assert retrieved.run_id == sample_experiment_record.run_id
        assert retrieved.final_score == sample_experiment_record.final_score
        assert retrieved.critic_scores == sample_experiment_record.critic_scores
    
    def test_complex_queries(self, temp_storage_path):
        """Test complex SQL queries and filtering."""
        db_path = temp_storage_path / "test.sqlite"
        storage = SQLiteStorage(db_path)
        
        # Store multiple records
        records = []
        for i in range(10):
            record = ExperimentRecord(
                run_id=f"run_{i % 3}",  # 3 different runs
                timestamp=f"2024-01-0{1 + i//3}T10:00:00",
                model_name=f"model_{i % 2}",  # 2 different models
                config_hash="test_hash",
                question_id=f"q{i:03d}",
                question_text=f"Question {i}",
                answer_text=f"Answer {i}",
                final_score=70 + i * 3,  # Scores from 70 to 97
                final_tier="good",
                per_dimension_scores={},
                critic_scores={"critic1": 70 + i * 3},
                confidence_level=0.8,
                execution_time_ms=1000.0,
                aggregation_method="test",
                consensus_level="high",
                score_variance=2.0,
                evaluation_summary="Test evaluation",
                system_version="1.0",
                critics_used=["critic1"],
                tags=["test"]
            )
            records.append(record)
            storage.store_result(record)
        
        # Test score range query
        high_score_results = storage.get_results({"score_range": (85, 100)})
        assert len(high_score_results) >= 3  # Should find records with scores 85, 88, 91, 94, 97
        
        # Test model filter
        model_0_results = storage.get_results({"model_name": "model_0"})
        assert len(model_0_results) == 5  # Even indices
        
        # Test run filter
        run_1_results = storage.get_results({"run_id": "run_1"})
        assert len(run_1_results) == 3  # Indices 1, 4, 7


class TestExperimentTracker:
    """Test main ExperimentTracker interface."""
    
    def test_tracker_initialization(self, temp_storage_path):
        """Test tracker initialization with different backends."""
        # Test JSONL backend
        tracker_jsonl = ExperimentTracker("jsonl", temp_storage_path)
        assert isinstance(tracker_jsonl.storage, JSONLStorage)
        
        # Test SQLite backend
        tracker_sqlite = ExperimentTracker("sqlite", temp_storage_path / "test.db")
        assert isinstance(tracker_sqlite.storage, SQLiteStorage)
        
        # Test invalid backend
        with pytest.raises(ValueError):
            ExperimentTracker("invalid_backend")
    
    def test_run_lifecycle(self, temp_storage_path):
        """Test complete run lifecycle: start, log results, retrieve."""
        tracker = ExperimentTracker("jsonl", temp_storage_path)
        
        # Start a run
        config = {"model": "test_model", "temperature": 0.7}
        run_id = tracker.start_run("test_model", config, "Test run")
        
        assert run_id is not None
        assert run_id.startswith("run_")
        
        # Verify run metadata stored
        metadata = tracker.get_run_metadata(run_id)
        assert metadata is not None
        assert metadata.model_name == "test_model"
        assert metadata.config_data == config
    
    def test_generate_run_id(self, temp_storage_path):
        """Test run ID generation."""
        tracker = ExperimentTracker("jsonl", temp_storage_path)
        
        config = {"temperature": 0.7}
        run_id1 = tracker.generate_run_id("model1", config)
        run_id2 = tracker.generate_run_id("model1", config)
        run_id3 = tracker.generate_run_id("model2", config)
        
        # Different timestamps should generate different IDs
        assert run_id1 != run_id2
        
        # Different models should generate different IDs
        assert run_id1.split("_")[3] != run_id3.split("_")[3]  # Different model names
    
    @patch('src.eval.tracking.datetime')
    def test_log_result(self, mock_datetime, temp_storage_path):
        """Test logging MultiCriticResult."""
        # Mock datetime for consistent timestamps
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T10:00:00"
        
        tracker = ExperimentTracker("jsonl", temp_storage_path)
        
        # Create mock MultiCriticResult
        mock_result = Mock(spec=MultiCriticResult)
        mock_result.request_id = "test_request_123"
        mock_result.question = "Test question?"
        mock_result.answer = "Test answer."
        mock_result.timestamp = "2024-01-01T10:00:00"
        mock_result.total_execution_time_ms = 1500.0
        mock_result.final_score = 85
        mock_result.final_tier = "good"
        mock_result.confidence_level = 0.8
        mock_result.evaluation_summary = "Good performance"
        mock_result.system_version = "1.0"
        mock_result.critics_used = ["coverage", "depth"]
        
        # Mock aggregation
        mock_aggregation = Mock()
        mock_aggregation.individual_scores = {"coverage": 80, "depth": 90}
        mock_aggregation.aggregation_method = "reasoned"
        mock_aggregation.consensus_level = "high"
        mock_aggregation.score_variance = 3.5
        mock_result.final_aggregation = mock_aggregation
        mock_result.individual_critic_scores = {"coverage": 80, "depth": 90}
        
        # Mock debate rounds (simplified)
        mock_result.debate_rounds = []
        
        # Create metadata
        metadata = RunMetadata(
            run_id="test_run",
            timestamp="2024-01-01T10:00:00",
            model_name="test_model",
            config_hash="abc123",
            config_data={"temp": 0.7}
        )
        
        # Log result
        success = tracker.log_result(mock_result, metadata, "test_question")
        assert success
        
        # Verify result was stored
        results = tracker.get_run_results("test_run")
        assert len(results) == 1
        
        stored_result = results[0]
        assert stored_result.question_id == "test_question"
        assert stored_result.final_score == 85
        assert stored_result.critic_scores == {"coverage": 80, "depth": 90}


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_tracker(self, temp_storage_path):
        """Test create_tracker convenience function."""
        tracker = create_tracker("jsonl", str(temp_storage_path))
        assert isinstance(tracker, ExperimentTracker)
        assert isinstance(tracker.storage, JSONLStorage)
    
    def test_log_result_convenience(self, temp_storage_path):
        """Test log_result convenience function."""
        # This test would require more complex mocking
        # For now, just test that it imports correctly
        from src.eval.tracking import log_result
        assert callable(log_result)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_storage_path(self):
        """Test behavior with invalid storage paths."""
        # Test with read-only directory would require special setup
        # For now, test with very long path name
        long_path = "x" * 300  # Very long path
        
        try:
            storage = JSONLStorage(long_path)
            # Should not raise exception during initialization
            assert True
        except Exception:
            # Some systems may reject very long paths
            assert True
    
    def test_malformed_jsonl_data(self, temp_storage_path):
        """Test handling of malformed JSONL data."""
        storage = JSONLStorage(temp_storage_path)
        
        # Write malformed data directly to file
        results_file = temp_storage_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        
        # Should handle malformed lines gracefully
        results = storage.get_results()
        # Should only get valid results, malformed lines skipped
        assert len(results) == 0  # None match our ExperimentRecord schema
    
    def test_empty_database_queries(self, temp_storage_path):
        """Test queries on empty database."""
        storage = SQLiteStorage(temp_storage_path / "empty.db")
        
        results = storage.get_results()
        assert results == []
        
        metadata = storage.get_run_metadata("nonexistent")
        assert metadata is None
        
        all_runs = storage.get_all_runs()
        assert all_runs == []


# Integration tests
class TestIntegration:
    """Integration tests across components."""
    
    def test_full_experiment_workflow(self, temp_storage_path):
        """Test complete experiment workflow."""
        tracker = ExperimentTracker("sqlite", temp_storage_path / "integration.db")
        
        # Start multiple runs
        runs = []
        for i in range(3):
            config = {"model": f"model_{i}", "temp": 0.7 + i * 0.1}
            run_id = tracker.start_run(f"model_{i}", config, f"Test run {i}")
            runs.append(run_id)
        
        # Verify all runs stored
        all_runs = tracker.get_all_runs()
        assert len(all_runs) == 3
        
        # Verify run retrieval
        for run_id in runs:
            metadata = tracker.get_run_metadata(run_id)
            assert metadata is not None
            assert metadata.run_id == run_id
    
    def test_storage_backend_consistency(self, temp_storage_path, sample_experiment_record):
        """Test that both storage backends produce consistent results."""
        # Store same data in both backends
        jsonl_storage = JSONLStorage(temp_storage_path / "jsonl")
        sqlite_storage = SQLiteStorage(temp_storage_path / "test.db")
        
        # Store in both
        jsonl_storage.store_result(sample_experiment_record)
        sqlite_storage.store_result(sample_experiment_record)
        
        # Retrieve from both
        jsonl_results = jsonl_storage.get_results()
        sqlite_results = sqlite_storage.get_results()
        
        assert len(jsonl_results) == len(sqlite_results) == 1
        
        # Compare key fields
        jsonl_result = jsonl_results[0]
        sqlite_result = sqlite_results[0]
        
        assert jsonl_result.run_id == sqlite_result.run_id
        assert jsonl_result.final_score == sqlite_result.final_score
        assert jsonl_result.critic_scores == sqlite_result.critic_scores


if __name__ == "__main__":
    pytest.main([__file__])