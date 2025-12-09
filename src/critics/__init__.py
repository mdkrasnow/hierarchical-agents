"""
Critics module for multi-agent review system.

Provides scoring agents that evaluate answers based on coverage, detail,
and style rather than factual correctness.
"""

from .models import CriticScore, DimensionScore, CriticRequest
from .single_critic import SingleCriticAgent, score_answer, score_answers_batch

__all__ = [
    'CriticScore',
    'DimensionScore', 
    'CriticRequest',
    'SingleCriticAgent',
    'score_answer',
    'score_answers_batch'
]