"""
Question Dataset Loaders and Utilities

This module provides high-level utilities for loading and working with 
evaluation questions, including convenience functions for common filtering
operations and dataset management tasks.
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
import logging
from datetime import datetime, timedelta

from .dataset import Question, QuestionDataset, load_questions_from_jsonl, load_categories_config

logger = logging.getLogger(__name__)

class QuestionLoader:
    """High-level interface for loading and managing evaluation questions."""
    
    def __init__(self, 
                 questions_path: str = None, 
                 categories_config_path: str = None,
                 base_dir: str = None):
        """
        Initialize the question loader.
        
        Args:
            questions_path: Path to questions JSONL file
            categories_config_path: Path to categories config YAML file
            base_dir: Base directory to resolve relative paths from
        """
        if base_dir is None:
            # Default to project root (assuming we're in src/scoring/)
            base_dir = Path(__file__).parent.parent.parent
        else:
            base_dir = Path(base_dir)
        
        # Set default paths if not provided
        if questions_path is None:
            questions_path = base_dir / "data" / "questions.jsonl"
        if categories_config_path is None:
            categories_config_path = base_dir / "configs" / "question_categories.yaml"
        
        self.questions_path = Path(questions_path)
        self.categories_config_path = Path(categories_config_path)
        
        self._dataset = None
        self._categories_config = None
    
    @property
    def dataset(self) -> QuestionDataset:
        """Get the loaded dataset, loading it if necessary."""
        if self._dataset is None:
            self._dataset = self.load_dataset()
        return self._dataset
    
    @property
    def categories_config(self) -> Optional[Dict[str, Any]]:
        """Get the categories configuration, loading it if necessary."""
        if self._categories_config is None:
            self._categories_config = load_categories_config(str(self.categories_config_path))
        return self._categories_config
    
    def load_dataset(self) -> QuestionDataset:
        """Load the complete dataset from file."""
        dataset = load_questions_from_jsonl(str(self.questions_path))
        dataset.categories_config = self.categories_config
        return dataset
    
    def reload_dataset(self) -> QuestionDataset:
        """Force reload of the dataset from file."""
        self._dataset = None
        self._categories_config = None
        return self.dataset
    
    def get_all_questions(self) -> QuestionDataset:
        """Get all questions in the dataset."""
        return self.dataset
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get a specific question by ID."""
        return self.dataset.get_question(question_id)
    
    # Category-based filtering
    def get_hierarchical_agents_questions(self) -> QuestionDataset:
        """Get all hierarchical agents questions."""
        return self.dataset.get_by_category("hierarchical-agents")
    
    def get_supabase_sql_questions(self) -> QuestionDataset:
        """Get all Supabase/SQL questions."""
        return self.dataset.get_by_category("supabase-sql")
    
    def get_writing_questions(self) -> QuestionDataset:
        """Get all general writing questions."""
        return self.dataset.get_by_category("general-writing")
    
    # Difficulty-based filtering
    def get_beginner_questions(self) -> QuestionDataset:
        """Get all beginner-level questions."""
        return self.dataset.get_by_difficulty("beginner")
    
    def get_intermediate_questions(self) -> QuestionDataset:
        """Get all intermediate-level questions."""
        return self.dataset.get_by_difficulty("intermediate")
    
    def get_advanced_questions(self) -> QuestionDataset:
        """Get all advanced-level questions."""
        return self.dataset.get_by_difficulty("advanced")
    
    def get_expert_questions(self) -> QuestionDataset:
        """Get all expert-level questions."""
        return self.dataset.get_by_difficulty("expert")
    
    # Rubric focus filtering
    def get_coverage_focused_questions(self) -> QuestionDataset:
        """Get questions that focus on coverage evaluation."""
        return self.dataset.get_by_rubric_focus("coverage")
    
    def get_detail_focused_questions(self) -> QuestionDataset:
        """Get questions that focus on detail and specificity."""
        return self.dataset.get_by_rubric_focus("detail_specificity")
    
    def get_structure_focused_questions(self) -> QuestionDataset:
        """Get questions that focus on structure and coherence."""
        return self.dataset.get_by_rubric_focus("structure_coherence")
    
    def get_style_focused_questions(self) -> QuestionDataset:
        """Get questions that focus on style and tone."""
        return self.dataset.get_by_rubric_focus("style_tone")
    
    def get_instruction_focused_questions(self) -> QuestionDataset:
        """Get questions that focus on instruction following."""
        return self.dataset.get_by_rubric_focus("instruction_following")
    
    # Tag-based filtering
    def get_questions_by_tag(self, tag: str) -> QuestionDataset:
        """Get all questions with a specific tag."""
        return self.dataset.get_by_tags(tag)
    
    def get_questions_by_tags(self, tags: List[str]) -> QuestionDataset:
        """Get all questions that have any of the specified tags."""
        return self.dataset.get_by_tags(tags)
    
    # Advanced filtering
    def filter_questions(self, **filters) -> QuestionDataset:
        """Apply custom filters to questions."""
        return self.dataset.filter_questions(**filters)
    
    def get_recent_questions(self, days: int = 30) -> QuestionDataset:
        """Get questions created within the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.dataset.filter_questions(created_after=cutoff_date.isoformat())
    
    # Sampling and selection
    def sample_balanced_questions(self, n_per_category: int = 3) -> QuestionDataset:
        """Sample questions with balanced representation across categories."""
        sampled_questions = []
        
        categories = self.get_available_categories()
        for category in categories:
            category_questions = self.dataset.get_by_category(category)
            if category_questions.questions:
                sample_size = min(n_per_category, len(category_questions.questions))
                sampled = category_questions.sample_questions(sample_size)
                sampled_questions.extend(sampled.questions)
        
        return QuestionDataset(
            questions=sampled_questions,
            categories_config=self.categories_config
        )
    
    def sample_by_difficulty_distribution(self, 
                                        total_questions: int,
                                        beginner_pct: float = 0.2,
                                        intermediate_pct: float = 0.4,
                                        advanced_pct: float = 0.3,
                                        expert_pct: float = 0.1) -> QuestionDataset:
        """Sample questions with specified difficulty distribution."""
        if abs((beginner_pct + intermediate_pct + advanced_pct + expert_pct) - 1.0) > 0.01:
            raise ValueError("Difficulty percentages must sum to 1.0")
        
        n_beginner = int(total_questions * beginner_pct)
        n_intermediate = int(total_questions * intermediate_pct)
        n_advanced = int(total_questions * advanced_pct)
        n_expert = total_questions - n_beginner - n_intermediate - n_advanced
        
        sampled_questions = []
        
        # Sample from each difficulty level
        difficulty_samples = [
            (n_beginner, "beginner"),
            (n_intermediate, "intermediate"),
            (n_advanced, "advanced"),
            (n_expert, "expert")
        ]
        
        for n, difficulty in difficulty_samples:
            if n > 0:
                difficulty_questions = self.dataset.get_by_difficulty(difficulty)
                if difficulty_questions.questions:
                    sampled = difficulty_questions.sample_questions(n)
                    sampled_questions.extend(sampled.questions)
        
        return QuestionDataset(
            questions=sampled_questions,
            categories_config=self.categories_config
        )
    
    def create_test_set(self, 
                       n_questions: int = 10,
                       include_all_categories: bool = True,
                       include_all_difficulties: bool = True) -> QuestionDataset:
        """Create a balanced test set for evaluation."""
        filters = {}
        
        if include_all_categories:
            # Ensure we have representation from each category
            categories = self.get_available_categories()
            questions_per_category = max(1, n_questions // len(categories))
            
            sampled_questions = []
            remaining = n_questions
            
            for i, category in enumerate(categories):
                # For the last category, take all remaining questions
                if i == len(categories) - 1:
                    n_from_category = remaining
                else:
                    n_from_category = min(questions_per_category, remaining)
                
                if n_from_category > 0:
                    category_questions = self.dataset.get_by_category(category)
                    if category_questions.questions:
                        sampled = category_questions.sample_questions(n_from_category)
                        sampled_questions.extend(sampled.questions)
                        remaining -= len(sampled.questions)
            
            return QuestionDataset(
                questions=sampled_questions,
                categories_config=self.categories_config
            )
        
        else:
            return self.dataset.sample_questions(n_questions, **filters)
    
    # Information and statistics
    def get_available_categories(self) -> List[str]:
        """Get list of all available categories in the dataset."""
        categories = set()
        for question in self.dataset.questions:
            categories.add(question.category)
        return sorted(list(categories))
    
    def get_available_tags(self) -> List[str]:
        """Get list of all available tags in the dataset."""
        tags = set()
        for question in self.dataset.questions:
            tags.update(question.tags)
        return sorted(list(tags))
    
    def get_available_difficulties(self) -> List[str]:
        """Get list of all available difficulty levels in the dataset."""
        difficulties = set()
        for question in self.dataset.questions:
            difficulties.add(question.difficulty)
        return sorted(list(difficulties))
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset."""
        return self.dataset.get_statistics()
    
    def print_dataset_summary(self):
        """Print a human-readable summary of the dataset."""
        stats = self.get_dataset_statistics()
        
        print("=== Question Dataset Summary ===")
        print(f"Total Questions: {stats['total_questions']}")
        print()
        
        print("Categories:")
        for category, count in stats['categories'].items():
            print(f"  {category}: {count}")
        print()
        
        print("Difficulty Levels:")
        for difficulty, count in stats['difficulties'].items():
            print(f"  {difficulty}: {count}")
        print()
        
        print("Most Common Tags:")
        sorted_tags = sorted(stats['tags'].items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:10]:
            print(f"  {tag}: {count}")
        
        if 'rubric_focus_distribution' in stats:
            print()
            print("Rubric Focus Distribution:")
            for focus, count in stats['rubric_focus_distribution'].items():
                print(f"  {focus}: {count}")

# Convenience functions for common operations
def get_default_loader() -> QuestionLoader:
    """Get a QuestionLoader with default paths."""
    return QuestionLoader()

def load_all_questions() -> QuestionDataset:
    """Quick function to load all questions with default settings."""
    loader = get_default_loader()
    return loader.get_all_questions()

def get_questions_for_testing(n: int = 5) -> QuestionDataset:
    """Quick function to get a small set of questions for testing."""
    loader = get_default_loader()
    return loader.create_test_set(n_questions=n)

def search_questions(search_term: str) -> QuestionDataset:
    """Search for questions containing the search term in prompt or tags."""
    loader = get_default_loader()
    all_questions = loader.get_all_questions()
    
    matching_questions = []
    search_term_lower = search_term.lower()
    
    for question in all_questions.questions:
        # Search in prompt
        if search_term_lower in question.prompt.lower():
            matching_questions.append(question)
        # Search in tags
        elif any(search_term_lower in tag.lower() for tag in question.tags):
            matching_questions.append(question)
        # Search in expected key points
        elif any(search_term_lower in point.lower() for point in question.expected_key_points):
            matching_questions.append(question)
    
    return QuestionDataset(
        questions=matching_questions,
        categories_config=loader.categories_config
    )

# Export main classes and functions
__all__ = [
    'QuestionLoader',
    'get_default_loader',
    'load_all_questions', 
    'get_questions_for_testing',
    'search_questions'
]