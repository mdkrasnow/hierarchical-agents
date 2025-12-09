"""
Role-based permission models for data access control.

Handles scope definition for principals vs superintendents and determines
what evaluations, teachers, and schools each user can access.
"""

from enum import Enum
from typing import List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, validator

from .database import User, Organization


class UserRole(str, Enum):
    """Standard user roles in the system."""
    SUPERINTENDENT = "superintendent"
    PRINCIPAL = "principal" 
    EVALUATOR = "evaluator"
    ESC_LEADER = "esc_leader"
    DISTRICT_ADMIN = "district_admin"
    TEACHER = "teacher"  # For future use


class AccessLevel(str, Enum):
    """Levels of data access."""
    FULL = "full"  # All data in scope
    READ_ONLY = "read_only"  # View but not modify
    SUMMARY_ONLY = "summary_only"  # Aggregated data only
    LIMITED = "limited"  # Restricted subset


class PermissionScope(BaseModel):
    """Defines what data a user can access."""
    
    # Organizational scope
    organization_ids: Set[UUID] = set()
    school_ids: Set[UUID] = set()
    
    # Data type permissions
    can_view_evaluations: bool = True
    can_view_teacher_details: bool = True
    can_view_raw_notes: bool = True
    can_view_ai_analysis: bool = True
    
    # Functional permissions
    can_run_district_analysis: bool = False
    can_run_school_analysis: bool = False
    can_create_pd_cohorts: bool = False
    can_export_data: bool = False
    
    # Access level
    access_level: AccessLevel = AccessLevel.READ_ONLY
    
    # Evaluation filtering
    can_see_informal_evals: bool = True
    can_see_archived_evals: bool = False
    max_history_months: Optional[int] = 24  # How far back they can look
    
    @validator('organization_ids', 'school_ids')
    def validate_non_empty_for_access(cls, v, values):
        """Ensure user has some scope defined."""
        # This validation would depend on access level and role
        return v


class UserScope(BaseModel):
    """Complete scope and permissions for a specific user."""
    
    user_id: UUID
    role: UserRole
    permissions: PermissionScope
    
    # Cached organizational info
    organization_names: List[str] = []
    school_names: List[str] = []
    
    # Role-specific settings
    role_specific_config: dict = {}

    @classmethod
    def from_user(cls, user: User, organization: Organization) -> "UserScope":
        """Create UserScope from User and Organization models."""
        
        # Determine role from user.class_role
        role = cls._parse_user_role(user.class_role)
        
        # Build permissions based on role
        permissions = cls._build_permissions_for_role(role, user, organization)
        
        return cls(
            user_id=user.get_user_id(),
            role=role,
            permissions=permissions,
            organization_names=[organization.name] if organization else [],
            school_names=[],  # Would be populated from school lookup
            role_specific_config=cls._get_role_config(role)
        )
    
    @staticmethod
    def _parse_user_role(class_role: Optional[str]) -> UserRole:
        """Parse user.class_role into standardized UserRole enum."""
        if not class_role:
            return UserRole.EVALUATOR
            
        class_role_lower = class_role.lower()
        
        if "superintendent" in class_role_lower:
            return UserRole.SUPERINTENDENT
        elif "principal" in class_role_lower:
            return UserRole.PRINCIPAL
        elif "esc" in class_role_lower or "educational service" in class_role_lower:
            return UserRole.ESC_LEADER
        elif "district" in class_role_lower and "admin" in class_role_lower:
            return UserRole.DISTRICT_ADMIN
        elif "teacher" in class_role_lower:
            return UserRole.TEACHER
        else:
            return UserRole.EVALUATOR
    
    @staticmethod 
    def _build_permissions_for_role(
        role: UserRole, 
        user: User, 
        organization: Organization
    ) -> PermissionScope:
        """Build PermissionScope based on user role."""
        
        base_permissions = PermissionScope(
            organization_ids={user.organization_id},
            school_ids=set(user.school_id or [])
        )
        
        if role == UserRole.SUPERINTENDENT:
            # Superintendents see everything in their organization
            base_permissions.can_run_district_analysis = True
            base_permissions.can_run_school_analysis = True
            base_permissions.can_create_pd_cohorts = True
            base_permissions.can_export_data = True
            base_permissions.access_level = AccessLevel.FULL
            base_permissions.can_see_archived_evals = True
            base_permissions.max_history_months = None  # No limit
            
        elif role == UserRole.PRINCIPAL:
            # Principals see their school(s) only
            base_permissions.can_run_school_analysis = True
            base_permissions.can_create_pd_cohorts = True
            base_permissions.can_export_data = True
            base_permissions.access_level = AccessLevel.FULL
            base_permissions.can_see_archived_evals = True
            
        elif role == UserRole.ESC_LEADER:
            # ESC leaders have multi-organization access
            base_permissions.can_run_district_analysis = True
            base_permissions.can_run_school_analysis = True
            base_permissions.access_level = AccessLevel.FULL
            base_permissions.max_history_months = None
            
        elif role == UserRole.EVALUATOR:
            # Evaluators have limited scope
            base_permissions.can_view_teacher_details = True
            base_permissions.access_level = AccessLevel.LIMITED
            base_permissions.max_history_months = 12
            
        else:
            # Default minimal permissions
            base_permissions.access_level = AccessLevel.SUMMARY_ONLY
            base_permissions.can_view_raw_notes = False
            base_permissions.max_history_months = 6
            
        return base_permissions
    
    @staticmethod
    def _get_role_config(role: UserRole) -> dict:
        """Get role-specific configuration settings."""
        configs = {
            UserRole.SUPERINTENDENT: {
                "default_view": "district_dashboard",
                "preferred_aggregation": "district",
                "show_school_comparisons": True,
                "board_story_focus": True
            },
            UserRole.PRINCIPAL: {
                "default_view": "school_dashboard", 
                "preferred_aggregation": "school",
                "show_teacher_lists": True,
                "pd_planning_focus": True
            },
            UserRole.ESC_LEADER: {
                "default_view": "multi_district",
                "preferred_aggregation": "district",
                "show_cross_district_patterns": True
            },
            UserRole.EVALUATOR: {
                "default_view": "evaluation_list",
                "preferred_aggregation": "teacher",
                "detail_level": "individual"
            }
        }
        
        return configs.get(role, {})
    
    def can_access_evaluation(self, evaluation_school_id: UUID, evaluation_org_id: UUID) -> bool:
        """Check if user can access a specific evaluation."""
        
        # Check organizational scope
        if evaluation_org_id not in self.permissions.organization_ids:
            return False
            
        # For school-level users, check school scope
        if self.role == UserRole.PRINCIPAL:
            if evaluation_school_id not in self.permissions.school_ids:
                return False
                
        return True
    
    def get_evaluation_filter_params(self) -> dict:
        """Get database filter parameters for this user's scope."""
        params = {
            "organization_ids": list(self.permissions.organization_ids),
            "include_archived": self.permissions.can_see_archived_evals,
            "include_informal": self.permissions.can_see_informal_evals
        }
        
        if self.permissions.school_ids:
            params["school_ids"] = list(self.permissions.school_ids)
            
        if self.permissions.max_history_months:
            params["max_history_months"] = self.permissions.max_history_months
            
        return params
    
    def get_agent_hierarchy_config(self) -> dict:
        """Get configuration for which agents this user should run."""
        if self.role == UserRole.SUPERINTENDENT:
            return {
                "run_evaluation_agent": True,
                "run_teacher_agent": True, 
                "run_school_agent": True,
                "run_district_agent": True,
                "primary_output": "district_summary"
            }
        elif self.role == UserRole.PRINCIPAL:
            return {
                "run_evaluation_agent": True,
                "run_teacher_agent": True,
                "run_school_agent": True, 
                "run_district_agent": False,
                "primary_output": "school_summary"
            }
        else:
            return {
                "run_evaluation_agent": True,
                "run_teacher_agent": True,
                "run_school_agent": False,
                "run_district_agent": False, 
                "primary_output": "teacher_summary"
            }