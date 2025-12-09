"""
Orchestration package for coordinating hierarchical agent execution.

This package provides:
- HierarchicalOrchestrator for coordinating multi-layer agent chains
- Role-based execution routing for principals vs superintendents
- Parallel execution coordination with concurrency limits
"""

from .hierarchical import HierarchicalOrchestrator, OrchestrationConfig, OrchestrationResult

__all__ = [
    "HierarchicalOrchestrator",
    "OrchestrationConfig", 
    "OrchestrationResult"
]