"""
Experiment evaluation and tracking system.

Provides comprehensive experiment tracking, analysis, and comparison capabilities
for multi-critic evaluation results.
"""

from .tracking import (
    ExperimentTracker, 
    ExperimentRecord, 
    RunMetadata, 
    JSONLStorage, 
    SQLiteStorage,
    create_tracker,
    log_result
)

from .analysis import (
    PerformanceAnalyzer,
    analyze_run,
    compare_runs, 
    get_challenging_questions,
    get_summary_statistics
)


__all__ = [
    # Tracking
    'ExperimentTracker',
    'ExperimentRecord', 
    'RunMetadata',
    'JSONLStorage',
    'SQLiteStorage',
    'create_tracker',
    'log_result',
    
    # Analysis
    'PerformanceAnalyzer',
    'analyze_run',
    'compare_runs',
    'get_challenging_questions', 
    'get_summary_statistics'
]