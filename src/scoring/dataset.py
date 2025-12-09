"""
Question Dataset Management for Multi-Agent Scoring System

This module provides core functionality for loading, filtering, and managing 
evaluation questions used in the hierarchical agents scoring system.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Question:
    """Represents a single evaluation question with all metadata."""
    
    id: str
    prompt: str
    category: str
    tags: List[str]
    expected_key_points: List[str]
    style_requirements: Dict[str, Any]
    difficulty: str
    rubric_focus: List[str]
    created_date: str
    
    # Optional fields with defaults
    description: Optional[str] = None
    notes: Optional[str] = None
    last_modified: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize question data after initialization."""
        self._validate_required_fields()
        self._normalize_data()
    
    def _validate_required_fields(self):
        """Validate that all required fields are present and valid."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Question ID must be a non-empty string")
        
        if not self.prompt or not isinstance(self.prompt, str):
            raise ValueError("Question prompt must be a non-empty string")
        
        if not self.category or not isinstance(self.category, str):
            raise ValueError("Question category must be a non-empty string")
        
        if not isinstance(self.tags, list) or not self.tags:
            raise ValueError("Question tags must be a non-empty list")
        
        if not isinstance(self.expected_key_points, list) or not self.expected_key_points:
            raise ValueError("Expected key points must be a non-empty list")
        
        if not isinstance(self.style_requirements, dict):
            raise ValueError("Style requirements must be a dictionary")
        
        if self.difficulty not in ['beginner', 'intermediate', 'advanced', 'expert']:
            raise ValueError("Difficulty must be one of: beginner, intermediate, advanced, expert")
    
    def _normalize_data(self):
        """Normalize data formats for consistency."""
        # Ensure tags are lowercase and unique
        self.tags = list(set(tag.lower().strip() for tag in self.tags))
        
        # Ensure category is lowercase
        self.category = self.category.lower().strip()
        
        # Ensure difficulty is lowercase
        self.difficulty = self.difficulty.lower().strip()
        
        # Normalize rubric_focus
        if isinstance(self.rubric_focus, list):
            self.rubric_focus = [focus.lower().strip() for focus in self.rubric_focus]
        else:
            self.rubric_focus = []
    
    def matches_filter(self, **filters) -> bool:
        """Check if this question matches the given filters."""
        for key, value in filters.items():
            if not self._field_matches(key, value):
                return False
        return True
    
    def _field_matches(self, field: str, filter_value: Any) -> bool:
        """Check if a specific field matches the filter value."""
        if field == 'category':
            return self.category == filter_value.lower() if filter_value else True
        
        elif field == 'difficulty':
            return self.difficulty == filter_value.lower() if filter_value else True
        
        elif field == 'tags':
            if isinstance(filter_value, str):
                return filter_value.lower() in self.tags
            elif isinstance(filter_value, list):
                return any(tag.lower() in self.tags for tag in filter_value)
            return True
        
        elif field == 'rubric_focus':
            if isinstance(filter_value, str):
                return filter_value.lower() in self.rubric_focus
            elif isinstance(filter_value, list):
                return any(focus.lower() in self.rubric_focus for focus in filter_value)
            return True
        
        elif field == 'has_tag':
            return filter_value.lower() in self.tags if filter_value else True
        
        elif field == 'created_after':
            try:
                question_date = datetime.fromisoformat(self.created_date)
                filter_date = datetime.fromisoformat(filter_value) if isinstance(filter_value, str) else filter_value
                return question_date >= filter_date
            except (ValueError, TypeError):
                return True
        
        elif field == 'created_before':
            try:
                question_date = datetime.fromisoformat(self.created_date)
                filter_date = datetime.fromisoformat(filter_value) if isinstance(filter_value, str) else filter_value
                return question_date <= filter_date
            except (ValueError, TypeError):
                return True
        
        # Default: check if the field exists and matches
        elif hasattr(self, field):
            field_value = getattr(self, field)
            if isinstance(field_value, str):
                return filter_value.lower() in field_value.lower() if filter_value else True
            return field_value == filter_value
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert question to dictionary format."""
        return {
            'id': self.id,
            'prompt': self.prompt,
            'category': self.category,
            'tags': self.tags,
            'expected_key_points': self.expected_key_points,
            'style_requirements': self.style_requirements,
            'difficulty': self.difficulty,
            'rubric_focus': self.rubric_focus,
            'created_date': self.created_date,
            'description': self.description,
            'notes': self.notes,
            'last_modified': self.last_modified
        }

@dataclass
class QuestionDataset:
    """Manages a collection of evaluation questions with filtering and analysis capabilities."""
    
    questions: List[Question] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    categories_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize dataset with validation."""
        self._validate_questions()
        self._update_metadata()
    
    def _validate_questions(self):
        """Validate all questions and check for duplicates."""
        ids = set()
        for question in self.questions:
            if question.id in ids:
                raise ValueError(f"Duplicate question ID found: {question.id}")
            ids.add(question.id)
    
    def _update_metadata(self):
        """Update dataset metadata with current statistics."""
        if not self.questions:
            return
        
        categories = {}
        difficulties = {}
        tags = {}
        
        for question in self.questions:
            # Count categories
            categories[question.category] = categories.get(question.category, 0) + 1
            
            # Count difficulties
            difficulties[question.difficulty] = difficulties.get(question.difficulty, 0) + 1
            
            # Count tags
            for tag in question.tags:
                tags[tag] = tags.get(tag, 0) + 1
        
        self.metadata.update({
            'total_questions': len(self.questions),
            'categories': categories,
            'difficulties': difficulties,
            'tags': tags,
            'last_updated': datetime.now().isoformat()
        })
    
    def add_question(self, question: Question):
        """Add a new question to the dataset."""
        if any(q.id == question.id for q in self.questions):
            raise ValueError(f"Question with ID {question.id} already exists")
        
        self.questions.append(question)
        self._update_metadata()
    
    def remove_question(self, question_id: str) -> bool:
        """Remove a question by ID. Returns True if question was found and removed."""
        for i, question in enumerate(self.questions):
            if question.id == question_id:
                del self.questions[i]
                self._update_metadata()
                return True
        return False
    
    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a specific question by ID."""
        for question in self.questions:
            if question.id == question_id:
                return question
        return None
    
    def filter_questions(self, **filters) -> 'QuestionDataset':
        """Filter questions based on provided criteria."""
        filtered = [q for q in self.questions if q.matches_filter(**filters)]
        
        return QuestionDataset(
            questions=filtered,
            categories_config=self.categories_config
        )
    
    def get_by_category(self, category: str) -> 'QuestionDataset':
        """Get all questions from a specific category."""
        return self.filter_questions(category=category)
    
    def get_by_difficulty(self, difficulty: str) -> 'QuestionDataset':
        """Get all questions of a specific difficulty."""
        return self.filter_questions(difficulty=difficulty)
    
    def get_by_tags(self, tags: Union[str, List[str]]) -> 'QuestionDataset':
        """Get all questions that have any of the specified tags."""
        return self.filter_questions(tags=tags)
    
    def get_by_rubric_focus(self, focus: Union[str, List[str]]) -> 'QuestionDataset':
        """Get all questions that focus on specific rubric dimensions."""
        return self.filter_questions(rubric_focus=focus)
    
    def sample_questions(self, n: int, **filters) -> 'QuestionDataset':
        """Sample n questions, optionally with filters applied."""
        filtered = self.filter_questions(**filters) if filters else self
        
        if n >= len(filtered.questions):
            return filtered
        
        import random
        sampled = random.sample(filtered.questions, n)
        
        return QuestionDataset(
            questions=sampled,
            categories_config=self.categories_config
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset."""
        if not self.questions:
            return {'total_questions': 0}
        
        stats = self.metadata.copy()
        
        # Add more detailed statistics
        rubric_focus_counts = {}
        tag_co_occurrence = {}
        
        for question in self.questions:
            # Rubric focus statistics
            for focus in question.rubric_focus:
                rubric_focus_counts[focus] = rubric_focus_counts.get(focus, 0) + 1
            
            # Tag co-occurrence (for the first 5 most common combinations)
            for i, tag1 in enumerate(question.tags):
                for tag2 in question.tags[i+1:]:
                    pair = tuple(sorted([tag1, tag2]))
                    tag_co_occurrence[pair] = tag_co_occurrence.get(pair, 0) + 1
        
        stats.update({
            'rubric_focus_distribution': rubric_focus_counts,
            'common_tag_pairs': sorted(tag_co_occurrence.items(), key=lambda x: x[1], reverse=True)[:10]
        })
        
        return stats
    
    def to_jsonl(self) -> str:
        """Convert dataset to JSONL format string."""
        lines = []
        for question in self.questions:
            lines.append(json.dumps(question.to_dict(), ensure_ascii=False))
        return '\n'.join(lines)
    
    def save_to_file(self, filepath: str):
        """Save dataset to JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_jsonl())
        
        logger.info(f"Saved {len(self.questions)} questions to {filepath}")

def load_questions_from_jsonl(filepath: str) -> QuestionDataset:
    """Load questions from a JSONL file."""
    questions = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    question = Question(**data)
                    questions.append(question)
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid question on line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(questions)} questions from {filepath}")
        
    except FileNotFoundError:
        logger.error(f"Questions file not found: {filepath}")
        return QuestionDataset()
    except Exception as e:
        logger.error(f"Error loading questions from {filepath}: {e}")
        return QuestionDataset()
    
    return QuestionDataset(questions=questions)

def load_categories_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load question categories configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded categories configuration from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.warning(f"Categories config file not found: {config_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading categories config from {config_path}: {e}")
        return None