"""
Answer generation framework for meta-evaluation system.

This module provides a flexible architecture for generating candidate answers
to evaluation questions using various strategies including generic LLMs and
hierarchical agents.
"""

from .answer_generator import (
    AnswerGenerator, 
    GenerationConfig, 
    GenerationResult, 
    GenerationStrategy,
    create_generator,
    generate_answer_simple
)
from .llm_generator import LLMGenerator  
from .hierarchical_generator import HierarchicalGenerator

__all__ = [
    'AnswerGenerator',
    'GenerationConfig', 
    'GenerationResult',
    'GenerationStrategy',
    'LLMGenerator',
    'HierarchicalGenerator',
    'create_generator',
    'generate_answer_simple'
]