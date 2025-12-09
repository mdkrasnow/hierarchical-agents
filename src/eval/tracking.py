"""
Experiment Tracking System for Multi-Agent Scoring Results.

Provides persistent storage for multi-critic evaluation results with support for
both JSONL and SQLite backends. Enables experiment logging, run metadata tracking,
and performance analysis across evaluation runs.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import hashlib
import logging
from dataclasses import dataclass

try:
    # Try relative import first (when used as part of package)
    from ..critics.debate_models import MultiCriticResult, ScoreAggregation
    from ..critics.models import CriticScore
except ImportError:
    # Fallback to absolute import (when used standalone)
    from critics.debate_models import MultiCriticResult, ScoreAggregation
    from critics.models import CriticScore

logger = logging.getLogger(__name__)


@dataclass
class RunMetadata:
    """Metadata about an evaluation run."""
    run_id: str
    timestamp: str
    model_name: str
    config_hash: str
    config_data: Dict[str, Any]
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ExperimentRecord:
    """Single experimental result record combining MultiCriticResult with metadata."""
    run_id: str
    timestamp: str
    model_name: str
    config_hash: str
    question_id: str
    question_text: str
    answer_text: str
    final_score: int
    final_tier: str
    per_dimension_scores: Dict[str, int]
    critic_scores: Dict[str, int]
    confidence_level: float
    execution_time_ms: float
    aggregation_method: str
    consensus_level: str
    score_variance: float
    
    # Additional metadata
    evaluation_summary: str
    system_version: str
    critics_used: List[str]
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ResultStorage:
    """Abstract base class for result storage backends."""
    
    def store_result(self, record: ExperimentRecord) -> bool:
        """Store a single experiment record."""
        raise NotImplementedError
    
    def store_run_metadata(self, metadata: RunMetadata) -> bool:
        """Store run-level metadata."""
        raise NotImplementedError
    
    def get_results(self, filters: Optional[Dict] = None) -> List[ExperimentRecord]:
        """Retrieve experiment records with optional filtering."""
        raise NotImplementedError
    
    def get_run_metadata(self, run_id: str) -> Optional[RunMetadata]:
        """Retrieve metadata for a specific run."""
        raise NotImplementedError
    
    def get_all_runs(self) -> List[RunMetadata]:
        """Get all run metadata."""
        raise NotImplementedError


class JSONLStorage(ResultStorage):
    """JSONL-based storage implementation."""
    
    def __init__(self, base_path: Union[str, Path] = "runs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.results_path = self.base_path / "results.jsonl"
        self.metadata_path = self.base_path / "metadata.jsonl"
    
    def store_result(self, record: ExperimentRecord) -> bool:
        """Store experiment record as JSONL entry."""
        try:
            # Convert record to dict for JSON serialization
            record_dict = {
                "run_id": record.run_id,
                "timestamp": record.timestamp,
                "model_name": record.model_name,
                "config_hash": record.config_hash,
                "question_id": record.question_id,
                "question_text": record.question_text,
                "answer_text": record.answer_text,
                "final_score": record.final_score,
                "final_tier": record.final_tier,
                "per_dimension_scores": record.per_dimension_scores,
                "critic_scores": record.critic_scores,
                "confidence_level": record.confidence_level,
                "execution_time_ms": record.execution_time_ms,
                "aggregation_method": record.aggregation_method,
                "consensus_level": record.consensus_level,
                "score_variance": record.score_variance,
                "evaluation_summary": record.evaluation_summary,
                "system_version": record.system_version,
                "critics_used": record.critics_used,
                "tags": record.tags
            }
            
            # Atomic write to JSONL file
            with open(self.results_path, "a") as f:
                f.write(json.dumps(record_dict) + "\n")
            
            logger.debug(f"Stored result for question {record.question_id} in run {record.run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
            return False
    
    def store_run_metadata(self, metadata: RunMetadata) -> bool:
        """Store run metadata as JSONL entry."""
        try:
            metadata_dict = {
                "run_id": metadata.run_id,
                "timestamp": metadata.timestamp,
                "model_name": metadata.model_name,
                "config_hash": metadata.config_hash,
                "config_data": metadata.config_data,
                "description": metadata.description,
                "tags": metadata.tags
            }
            
            with open(self.metadata_path, "a") as f:
                f.write(json.dumps(metadata_dict) + "\n")
            
            logger.info(f"Stored metadata for run {metadata.run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store metadata: {e}")
            return False
    
    def get_results(self, filters: Optional[Dict] = None) -> List[ExperimentRecord]:
        """Read experiment records from JSONL file with optional filtering."""
        if not self.results_path.exists():
            return []
        
        results = []
        try:
            with open(self.results_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        record = ExperimentRecord(**data)
                        
                        # Apply filters if specified
                        if filters:
                            if not self._matches_filters(record, filters):
                                continue
                        
                        results.append(record)
                        
        except Exception as e:
            logger.error(f"Failed to read results: {e}")
            
        return results
    
    def get_run_metadata(self, run_id: str) -> Optional[RunMetadata]:
        """Get metadata for specific run."""
        if not self.metadata_path.exists():
            return None
        
        try:
            with open(self.metadata_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data["run_id"] == run_id:
                            return RunMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
        
        return None
    
    def get_all_runs(self) -> List[RunMetadata]:
        """Get all run metadata."""
        if not self.metadata_path.exists():
            return []
        
        runs = []
        try:
            with open(self.metadata_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        runs.append(RunMetadata(**data))
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
        
        return runs
    
    def _matches_filters(self, record: ExperimentRecord, filters: Dict) -> bool:
        """Check if record matches filter criteria."""
        for key, value in filters.items():
            if hasattr(record, key):
                record_value = getattr(record, key)
                if isinstance(value, list):
                    if record_value not in value:
                        return False
                elif record_value != value:
                    return False
            else:
                # Handle nested filters (e.g., score ranges)
                if key == "score_range":
                    min_score, max_score = value
                    if not (min_score <= record.final_score <= max_score):
                        return False
                elif key == "time_range":
                    start_time, end_time = value
                    if not (start_time <= record.timestamp <= end_time):
                        return False
        return True


class SQLiteStorage(ResultStorage):
    """SQLite-based storage implementation for better querying performance."""
    
    def __init__(self, db_path: Union[str, Path] = "runs/results.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create run metadata table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS run_metadata (
                        run_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        config_hash TEXT NOT NULL,
                        config_data TEXT NOT NULL,  -- JSON
                        description TEXT,
                        tags TEXT,  -- JSON array
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create experiment results table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS experiment_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        config_hash TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        question_text TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        final_score INTEGER NOT NULL,
                        final_tier TEXT NOT NULL,
                        per_dimension_scores TEXT,  -- JSON
                        critic_scores TEXT,  -- JSON
                        confidence_level REAL NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        aggregation_method TEXT NOT NULL,
                        consensus_level TEXT NOT NULL,
                        score_variance REAL NOT NULL,
                        evaluation_summary TEXT,
                        system_version TEXT NOT NULL,
                        critics_used TEXT,  -- JSON array
                        tags TEXT,  -- JSON array
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (run_id) REFERENCES run_metadata (run_id)
                    )
                """)
                
                # Create indexes for common queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON experiment_results (run_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON experiment_results (timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON experiment_results (model_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_final_score ON experiment_results (final_score)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_question_id ON experiment_results (question_id)")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_result(self, record: ExperimentRecord) -> bool:
        """Store experiment record in SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experiment_results (
                        run_id, timestamp, model_name, config_hash, question_id,
                        question_text, answer_text, final_score, final_tier,
                        per_dimension_scores, critic_scores, confidence_level,
                        execution_time_ms, aggregation_method, consensus_level,
                        score_variance, evaluation_summary, system_version,
                        critics_used, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.run_id, record.timestamp, record.model_name,
                    record.config_hash, record.question_id, record.question_text,
                    record.answer_text, record.final_score, record.final_tier,
                    json.dumps(record.per_dimension_scores),
                    json.dumps(record.critic_scores), record.confidence_level,
                    record.execution_time_ms, record.aggregation_method,
                    record.consensus_level, record.score_variance,
                    record.evaluation_summary, record.system_version,
                    json.dumps(record.critics_used), json.dumps(record.tags)
                ))
            
            logger.debug(f"Stored result for question {record.question_id} in SQLite")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store result in SQLite: {e}")
            return False
    
    def store_run_metadata(self, metadata: RunMetadata) -> bool:
        """Store run metadata in SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO run_metadata (
                        run_id, timestamp, model_name, config_hash, 
                        config_data, description, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.run_id, metadata.timestamp, metadata.model_name,
                    metadata.config_hash, json.dumps(metadata.config_data),
                    metadata.description, json.dumps(metadata.tags)
                ))
            
            logger.info(f"Stored metadata for run {metadata.run_id} in SQLite")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store metadata in SQLite: {e}")
            return False
    
    def get_results(self, filters: Optional[Dict] = None) -> List[ExperimentRecord]:
        """Query experiment results from SQLite with optional filtering."""
        try:
            query = "SELECT * FROM experiment_results"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    if key == "score_range":
                        min_score, max_score = value
                        conditions.append("final_score BETWEEN ? AND ?")
                        params.extend([min_score, max_score])
                    elif key == "time_range":
                        start_time, end_time = value
                        conditions.append("timestamp BETWEEN ? AND ?")
                        params.extend([start_time, end_time])
                    elif isinstance(value, list):
                        placeholders = ",".join(["?"] * len(value))
                        conditions.append(f"{key} IN ({placeholders})")
                        params.extend(value)
                    else:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                results = []
                
                for row in cursor.fetchall():
                    record = ExperimentRecord(
                        run_id=row["run_id"],
                        timestamp=row["timestamp"],
                        model_name=row["model_name"],
                        config_hash=row["config_hash"],
                        question_id=row["question_id"],
                        question_text=row["question_text"],
                        answer_text=row["answer_text"],
                        final_score=row["final_score"],
                        final_tier=row["final_tier"],
                        per_dimension_scores=json.loads(row["per_dimension_scores"]),
                        critic_scores=json.loads(row["critic_scores"]),
                        confidence_level=row["confidence_level"],
                        execution_time_ms=row["execution_time_ms"],
                        aggregation_method=row["aggregation_method"],
                        consensus_level=row["consensus_level"],
                        score_variance=row["score_variance"],
                        evaluation_summary=row["evaluation_summary"],
                        system_version=row["system_version"],
                        critics_used=json.loads(row["critics_used"]),
                        tags=json.loads(row["tags"]) if row["tags"] else []
                    )
                    results.append(record)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to query results: {e}")
            return []
    
    def get_run_metadata(self, run_id: str) -> Optional[RunMetadata]:
        """Get metadata for specific run from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM run_metadata WHERE run_id = ?", 
                    (run_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return RunMetadata(
                        run_id=row["run_id"],
                        timestamp=row["timestamp"],
                        model_name=row["model_name"],
                        config_hash=row["config_hash"],
                        config_data=json.loads(row["config_data"]),
                        description=row["description"] or "",
                        tags=json.loads(row["tags"]) if row["tags"] else []
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get run metadata: {e}")
        
        return None
    
    def get_all_runs(self) -> List[RunMetadata]:
        """Get all run metadata from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM run_metadata ORDER BY timestamp DESC"
                )
                
                runs = []
                for row in cursor.fetchall():
                    runs.append(RunMetadata(
                        run_id=row["run_id"],
                        timestamp=row["timestamp"],
                        model_name=row["model_name"],
                        config_hash=row["config_hash"],
                        config_data=json.loads(row["config_data"]),
                        description=row["description"] or "",
                        tags=json.loads(row["tags"]) if row["tags"] else []
                    ))
                
                return runs
                
        except Exception as e:
            logger.error(f"Failed to get all runs: {e}")
            return []


class ExperimentTracker:
    """Main interface for experiment tracking."""
    
    def __init__(self, storage_type: str = "jsonl", storage_path: Union[str, Path] = "runs"):
        """
        Initialize experiment tracker.
        
        Args:
            storage_type: Either "jsonl" or "sqlite"
            storage_path: Base path or database file path
        """
        self.storage_path = Path(storage_path)
        
        if storage_type == "jsonl":
            self.storage = JSONLStorage(storage_path)
        elif storage_type == "sqlite":
            if storage_path == "runs":
                storage_path = "runs/results.sqlite"
            self.storage = SQLiteStorage(storage_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        logger.info(f"Initialized experiment tracker with {storage_type} storage at {storage_path}")
    
    def generate_run_id(self, model_name: str, config_data: Dict) -> str:
        """Generate unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate config hash for reproducibility
        config_str = json.dumps(config_data, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"run_{timestamp}_{model_name}_{config_hash}"
    
    def start_run(self, model_name: str, config_data: Dict, description: str = "", tags: List[str] = None) -> str:
        """
        Start a new evaluation run.
        
        Args:
            model_name: Name/identifier of model being evaluated
            config_data: Configuration parameters for the run
            description: Human-readable description
            tags: Optional tags for categorization
            
        Returns:
            Unique run ID for this evaluation run
        """
        run_id = self.generate_run_id(model_name, config_data)
        timestamp = datetime.now().isoformat()
        config_hash = hashlib.md5(json.dumps(config_data, sort_keys=True).encode()).hexdigest()[:8]
        
        metadata = RunMetadata(
            run_id=run_id,
            timestamp=timestamp,
            model_name=model_name,
            config_hash=config_hash,
            config_data=config_data,
            description=description,
            tags=tags or []
        )
        
        success = self.storage.store_run_metadata(metadata)
        if success:
            logger.info(f"Started evaluation run: {run_id}")
            return run_id
        else:
            raise RuntimeError(f"Failed to start run {run_id}")
    
    def log_result(self, result: MultiCriticResult, metadata: RunMetadata, question_id: str = None) -> bool:
        """
        Log a multi-critic evaluation result.
        
        Args:
            result: MultiCriticResult object from orchestrator
            metadata: Run metadata
            question_id: Optional question identifier (extracted from result if not provided)
            
        Returns:
            True if successfully logged
        """
        try:
            # Extract question ID if not provided
            if question_id is None:
                question_id = getattr(result, 'question_id', f"q_{result.request_id[:8]}")
            
            # Extract per-dimension scores from individual critics
            per_dimension_scores = {}
            for round_data in result.debate_rounds:
                for critic_result in round_data.results:
                    if critic_result.critic_role != "aggregator":  # Skip aggregator
                        critic_score = critic_result.critic_score
                        if hasattr(critic_score, 'dimension_scores') and critic_score.dimension_scores:
                            for dim_name, dim_score in critic_score.dimension_scores.items():
                                if isinstance(dim_score, dict):
                                    per_dimension_scores[f"{critic_result.critic_role}_{dim_name}"] = dim_score.get('score', 0)
                                else:
                                    per_dimension_scores[f"{critic_result.critic_role}_{dim_name}"] = getattr(dim_score, 'score', 0)
            
            # Create experiment record
            record = ExperimentRecord(
                run_id=metadata.run_id,
                timestamp=result.timestamp,
                model_name=metadata.model_name,
                config_hash=metadata.config_hash,
                question_id=question_id,
                question_text=result.question,
                answer_text=result.answer,
                final_score=result.final_score,
                final_tier=result.final_tier,
                per_dimension_scores=per_dimension_scores,
                critic_scores=result.individual_critic_scores,
                confidence_level=result.confidence_level,
                execution_time_ms=result.total_execution_time_ms,
                aggregation_method=result.final_aggregation.aggregation_method,
                consensus_level=result.final_aggregation.consensus_level,
                score_variance=result.final_aggregation.score_variance,
                evaluation_summary=result.evaluation_summary,
                system_version=result.system_version,
                critics_used=result.critics_used,
                tags=metadata.tags
            )
            
            success = self.storage.store_result(record)
            if success:
                logger.debug(f"Logged result for question {question_id} in run {metadata.run_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
            return False
    
    def get_run_results(self, run_id: str) -> List[ExperimentRecord]:
        """Get all results for a specific run."""
        return self.storage.get_results(filters={"run_id": run_id})
    
    def get_results_by_model(self, model_name: str) -> List[ExperimentRecord]:
        """Get all results for a specific model."""
        return self.storage.get_results(filters={"model_name": model_name})
    
    def get_results_by_score_range(self, min_score: int, max_score: int) -> List[ExperimentRecord]:
        """Get results within a score range."""
        return self.storage.get_results(filters={"score_range": (min_score, max_score)})
    
    def get_recent_results(self, limit: int = 100) -> List[ExperimentRecord]:
        """Get most recent results."""
        all_results = self.storage.get_results()
        return sorted(all_results, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_run_metadata(self, run_id: str) -> Optional[RunMetadata]:
        """Get metadata for a specific run."""
        return self.storage.get_run_metadata(run_id)
    
    def get_all_runs(self) -> List[RunMetadata]:
        """Get metadata for all runs."""
        return self.storage.get_all_runs()


# Convenience functions for direct usage

def create_tracker(storage_type: str = "jsonl", storage_path: str = "runs") -> ExperimentTracker:
    """Create an experiment tracker instance."""
    return ExperimentTracker(storage_type=storage_type, storage_path=storage_path)


def log_result(result: MultiCriticResult, metadata: RunMetadata, 
               tracker: Optional[ExperimentTracker] = None, question_id: str = None) -> bool:
    """
    Convenience function to log a result.
    
    Args:
        result: MultiCriticResult to log
        metadata: Run metadata
        tracker: Optional tracker instance (creates default if None)
        question_id: Optional question ID
        
    Returns:
        True if successfully logged
    """
    if tracker is None:
        tracker = create_tracker()
    
    return tracker.log_result(result, metadata, question_id)