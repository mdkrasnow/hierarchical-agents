"""
Database models mapping to existing Supabase tables.

These Pydantic models map to the existing schema:
- public.users 
- public.organizations
- public.danielson
- public.roles

Handles JSONB fields and maintains compatibility with existing data.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class Role(BaseModel):
    """Maps to public.roles table."""
    id: UUID
    name: str
    description: Optional[str] = None
    permissions: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Organization(BaseModel):
    """Maps to public.organizations table."""
    id: UUID
    name: str
    performance_level_config: Optional[Dict[str, Any]] = None  # JSONB field
    created_at: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    
    # Additional org metadata
    type: Optional[str] = None  # district, school, ESC, etc.
    parent_organization_id: Optional[UUID] = None

    class Config:
        from_attributes = True

    @validator('performance_level_config')
    def validate_performance_config(cls, v):
        """Validate performance level configuration structure."""
        if v is None:
            return v
        
        # Basic validation - should contain domain thresholds
        # Example structure: {"domains": {"I-A": {"green": 3, "yellow": 2}}}
        if not isinstance(v, dict):
            raise ValueError("performance_level_config must be a dictionary")
        
        return v


class User(BaseModel):
    """Maps to public.users table."""
    id: UUID  # Primary key field
    user_id: Optional[UUID] = None  # For backward compatibility if needed
    email: str
    role_id: UUID
    class_role: Optional[str] = None  # "Principal", "Superintendent", etc.
    organization_id: UUID
    school_id: Optional[List[UUID]] = None  # Array field - schools this user has access to
    created_at: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    
    # Additional user fields that might exist
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    active: bool = True

    class Config:
        from_attributes = True
        validate_assignment = True
        
    def get_user_id(self) -> UUID:
        """Get the user ID, handling both field names."""
        return self.user_id or self.id

    @validator('school_id')
    def validate_school_ids(cls, v):
        """Handle school_id array field."""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle case where single UUID comes as string
            try:
                return [UUID(v)]
            except ValueError:
                return []
        return v


class DomainScore(BaseModel):
    """Individual domain score within an evaluation."""
    domain_id: str  # e.g., "I-A", "II-B", etc.
    component_scores: Dict[str, Union[int, float]] = {}
    overall_score: Optional[Union[int, float]] = None
    notes: Optional[str] = None


class EvaluationMetadata(BaseModel):
    """Metadata fields from danielson evaluation."""
    framework_id: Optional[str] = None
    is_informal: bool = False
    is_archived: bool = False
    evaluator: Optional[str] = None
    teacher_name: Optional[str] = None
    school_name: Optional[str] = None


class DanielsonEvaluation(BaseModel):
    """Maps to public.danielson table."""
    id: UUID
    evaluation: Optional[Dict[str, Any]] = None  # Main evaluation JSON
    ai_evaluation: Optional[Dict[str, Any]] = None  # AI analysis JSONB
    low_inference_notes: Optional[str] = None
    
    # Metadata fields
    framework_id: Optional[str] = None
    is_informal: bool = False
    is_archived: bool = False
    evaluator: Optional[str] = None
    teacher_name: Optional[str] = None
    school_name: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    created_by: Optional[UUID] = None

    class Config:
        from_attributes = True

    @validator('evaluation')
    def validate_evaluation_structure(cls, v):
        """Validate evaluation JSON structure."""
        if v is None:
            return v
            
        if not isinstance(v, dict):
            raise ValueError("evaluation must be a dictionary")
            
        # Basic validation - should contain domain information
        return v

    @validator('ai_evaluation') 
    def validate_ai_evaluation(cls, v):
        """Validate AI evaluation JSONB structure."""
        if v is None:
            return v
            
        if not isinstance(v, dict):
            raise ValueError("ai_evaluation must be a dictionary")
            
        return v

    def get_domain_scores(self) -> List[DomainScore]:
        """Extract domain scores from evaluation JSON."""
        if not self.evaluation:
            return []
            
        domain_scores = []
        
        # Handle different possible structures in evaluation JSON
        if 'domains' in self.evaluation:
            domains_data = self.evaluation['domains']
        elif 'scores' in self.evaluation:
            domains_data = self.evaluation['scores']
        else:
            # Fallback - try to find domain-like keys
            domains_data = {k: v for k, v in self.evaluation.items() 
                          if isinstance(k, str) and '-' in k}  # I-A, II-B pattern
        
        for domain_id, domain_data in domains_data.items():
            if isinstance(domain_data, dict):
                domain_score = DomainScore(
                    domain_id=domain_id,
                    component_scores=domain_data.get('components', {}),
                    overall_score=domain_data.get('score'),
                    notes=domain_data.get('notes')
                )
                domain_scores.append(domain_score)
        
        return domain_scores

    def get_metadata(self) -> EvaluationMetadata:
        """Get structured metadata for this evaluation."""
        return EvaluationMetadata(
            framework_id=self.framework_id,
            is_informal=self.is_informal,
            is_archived=self.is_archived,
            evaluator=self.evaluator,
            teacher_name=self.teacher_name,
            school_name=self.school_name
        )