"""
Core data models for hierarchical agents system.

This package contains:
- Database models mapping to Supabase tables
- Hierarchical agent output schemas 
- Permission and role models
"""

from .database import User, Organization, DanielsonEvaluation, Role
from .agent_outputs import (
    EvaluationSummary,
    TeacherSummary, 
    SchoolSummary,
    DistrictSummary,
    DomainSummary,
    DomainStatus,
    RiskLevel,
    TrendDirection,
    PDCohort,
    SchoolRanking,
    BoardStory
)
from .permissions import PermissionScope, UserScope
from . import utils

__all__ = [
    # Database models
    "User",
    "Organization", 
    "DanielsonEvaluation",
    "Role",
    
    # Agent output schemas
    "EvaluationSummary",
    "TeacherSummary",
    "SchoolSummary", 
    "DistrictSummary",
    "DomainSummary",
    "DomainStatus",
    "RiskLevel",
    "TrendDirection",
    "PDCohort",
    "SchoolRanking",
    "BoardStory",
    
    # Permission models
    "PermissionScope",
    "UserScope",
    
    # Utilities
    "utils"
]