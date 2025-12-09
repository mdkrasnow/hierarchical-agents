"""
Scoring System for Hierarchical Multi-Agent Evaluation

This package provides comprehensive functionality for loading, filtering, and managing
evaluation questions used in the hierarchical agents scoring system.

Main components:
- Question: Individual evaluation question with metadata
- QuestionDataset: Collection of questions with filtering capabilities  
- QuestionLoader: High-level interface for loading and managing questions
"""

from .dataset import Question, QuestionDataset, load_questions_from_jsonl, load_categories_config
from .loaders import (
    QuestionLoader, 
    get_default_loader, 
    load_all_questions,
    get_questions_for_testing,
    search_questions
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    'Question',
    'QuestionDataset', 
    'QuestionLoader',
    
    # Loading functions
    'load_questions_from_jsonl',
    'load_categories_config',
    'get_default_loader',
    'load_all_questions',
    'get_questions_for_testing',
    'search_questions',
]