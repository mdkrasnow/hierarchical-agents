"""
Data access layer for evaluations with permission filtering and organizational queries.

Provides async queries for the hierarchical agents system with proper permission
filtering by organization_id, school_id[], and evaluator access patterns.
"""

import json
from typing import List, Optional, Dict, Any, Set, Union
from datetime import datetime, date
from dataclasses import dataclass
import asyncpg
import logging

from .connection import get_database_pool


logger = logging.getLogger(__name__)


@dataclass
class UserPermissions:
    """User permissions and access scope."""
    user_id: str
    email: str
    role_id: str
    class_role: Optional[str]
    organization_id: str
    school_ids: List[str]  # List of school IDs the user can access
    is_evaluator: bool = False
    is_principal: bool = False
    is_superintendent: bool = False


@dataclass
class EvaluationRecord:
    """Represents a Danielson evaluation record."""
    id: str
    teacher_name: str
    school_name: str
    organization_id: str
    evaluation_json: Dict[str, Any]
    ai_evaluation: Optional[Dict[str, Any]]
    low_inference_notes: Optional[str]
    framework_id: str
    is_informal: bool
    is_archived: bool
    evaluator: Optional[str]
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]


@dataclass
class TeacherSummary:
    """Summary of a teacher and their evaluations."""
    teacher_name: str
    school_name: str
    organization_id: str
    total_evaluations: int
    latest_evaluation_date: Optional[datetime]
    evaluation_ids: List[str]


@dataclass
class SchoolSummary:
    """Summary of a school and its teachers/evaluations."""
    school_name: str
    organization_id: str
    total_teachers: int
    total_evaluations: int
    teacher_names: List[str]


@dataclass
class OrganizationTree:
    """Hierarchical organization structure for a user."""
    organization_id: str
    organization_name: str
    accessible_schools: List[str]
    accessible_teachers: Dict[str, str]  # teacher_name -> school_name
    total_evaluations: int


class EvaluationQueries:
    """Data access layer for evaluation operations."""

    @staticmethod
    async def get_user_permissions(user_email: str) -> Optional[UserPermissions]:
        """
        Get user permissions and access scope.
        
        Args:
            user_email: Email address of the user
            
        Returns:
            UserPermissions object or None if user not found
        """
        query = """
        SELECT 
            u.id as user_id,
            u.email,
            u.role_id,
            u.class_role,
            u.organization_id,
            u.school_id as school_ids,
            r.name as role_name
        FROM public.users u
        LEFT JOIN public.roles r ON u.role_id = r.id
        WHERE u.email = $1 AND u.deleted_at IS NULL
        """
        
        pool = await get_database_pool()
        result = await pool.execute_query_one(query, user_email)
        
        if not result:
            return None
        
        # Parse school_ids (stored as JSONB array)
        school_ids = []
        if result['school_ids']:
            try:
                school_ids = json.loads(result['school_ids']) if isinstance(result['school_ids'], str) else result['school_ids']
                if not isinstance(school_ids, list):
                    school_ids = [school_ids] if school_ids else []
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse school_ids for user {user_email}: {result['school_ids']}")
                school_ids = []
        
        # Determine role flags
        role_name = result['role_name'] or ''
        class_role = result['class_role'] or ''
        
        is_evaluator = 'evaluator' in role_name.lower() or 'evaluator' in class_role.lower()
        is_principal = 'principal' in role_name.lower() or 'principal' in class_role.lower()
        is_superintendent = 'superintendent' in role_name.lower() or 'superintendent' in class_role.lower()
        
        return UserPermissions(
            user_id=result['user_id'],
            email=result['email'],
            role_id=result['role_id'],
            class_role=result['class_role'],
            organization_id=result['organization_id'],
            school_ids=school_ids,
            is_evaluator=is_evaluator,
            is_principal=is_principal,
            is_superintendent=is_superintendent
        )

    @staticmethod
    async def get_evaluations_for_user(
        permissions: UserPermissions,
        teacher_name: Optional[str] = None,
        school_name: Optional[str] = None,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
        include_informal: bool = True,
        include_archived: bool = False,
        limit: Optional[int] = None
    ) -> List[EvaluationRecord]:
        """
        Get evaluations accessible to a user with permission filtering.
        
        Args:
            permissions: User permissions object
            teacher_name: Filter by specific teacher name
            school_name: Filter by specific school name
            start_date: Filter evaluations after this date
            end_date: Filter evaluations before this date
            include_informal: Include informal evaluations
            include_archived: Include archived evaluations
            limit: Maximum number of evaluations to return
            
        Returns:
            List of EvaluationRecord objects
        """
        # Build WHERE conditions based on permissions and filters
        where_conditions = ["d.deleted_at IS NULL"]
        params = []
        param_counter = 1

        # Organization-level filtering
        where_conditions.append(f"d.organization_id = ${param_counter}")
        params.append(permissions.organization_id)
        param_counter += 1

        # School-level filtering for principals
        if permissions.is_principal and permissions.school_ids:
            school_placeholders = [f"${param_counter + i}" for i in range(len(permissions.school_ids))]
            where_conditions.append(f"d.school_name = ANY(ARRAY[{','.join(school_placeholders)}])")
            params.extend(permissions.school_ids)
            param_counter += len(permissions.school_ids)

        # Evaluator-based filtering - users can see evaluations they created
        if permissions.is_evaluator:
            where_conditions.append(f"(d.created_by = ${param_counter} OR d.evaluator = ${param_counter})")
            params.append(permissions.user_id)
            param_counter += 1

        # Additional filters
        if teacher_name:
            where_conditions.append(f"d.teacher_name ILIKE ${param_counter}")
            params.append(f"%{teacher_name}%")
            param_counter += 1

        if school_name:
            where_conditions.append(f"d.school_name ILIKE ${param_counter}")
            params.append(f"%{school_name}%")
            param_counter += 1

        if start_date:
            where_conditions.append(f"d.created_at >= ${param_counter}")
            params.append(start_date)
            param_counter += 1

        if end_date:
            where_conditions.append(f"d.created_at <= ${param_counter}")
            params.append(end_date)
            param_counter += 1

        if not include_informal:
            where_conditions.append("d.is_informal = false")

        if not include_archived:
            where_conditions.append("d.is_archived = false")

        # Construct query
        query = f"""
        SELECT 
            d.id,
            d.teacher_name,
            d.school_name,
            d.organization_id,
            d.evaluation,
            d.ai_evaluation,
            d.low_inference_notes,
            d.framework_id,
            d.is_informal,
            d.is_archived,
            d.evaluator,
            d.created_by,
            d.created_at,
            d.updated_at,
            d.deleted_at
        FROM public.danielson d
        WHERE {' AND '.join(where_conditions)}
        ORDER BY d.created_at DESC
        """

        if limit:
            query += f" LIMIT ${param_counter}"
            params.append(limit)

        pool = await get_database_pool()
        results = await pool.execute_query(query, *params)

        # Convert to EvaluationRecord objects
        evaluations = []
        for row in results:
            # Parse JSON fields
            evaluation_json = {}
            if row['evaluation']:
                try:
                    evaluation_json = json.loads(row['evaluation']) if isinstance(row['evaluation'], str) else row['evaluation']
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse evaluation JSON for evaluation {row['id']}")

            ai_evaluation = None
            if row['ai_evaluation']:
                try:
                    ai_evaluation = json.loads(row['ai_evaluation']) if isinstance(row['ai_evaluation'], str) else row['ai_evaluation']
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse ai_evaluation JSON for evaluation {row['id']}")

            evaluations.append(EvaluationRecord(
                id=row['id'],
                teacher_name=row['teacher_name'],
                school_name=row['school_name'],
                organization_id=row['organization_id'],
                evaluation_json=evaluation_json,
                ai_evaluation=ai_evaluation,
                low_inference_notes=row['low_inference_notes'],
                framework_id=row['framework_id'],
                is_informal=row['is_informal'],
                is_archived=row['is_archived'],
                evaluator=row['evaluator'],
                created_by=row['created_by'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                deleted_at=row['deleted_at']
            ))

        return evaluations

    @staticmethod
    async def get_evaluation_by_id(
        evaluation_id: str,
        permissions: UserPermissions
    ) -> Optional[EvaluationRecord]:
        """
        Get a specific evaluation by ID with permission checking.
        
        Args:
            evaluation_id: ID of the evaluation
            permissions: User permissions object
            
        Returns:
            EvaluationRecord or None if not found/not accessible
        """
        query = """
        SELECT 
            d.id,
            d.teacher_name,
            d.school_name,
            d.organization_id,
            d.evaluation,
            d.ai_evaluation,
            d.low_inference_notes,
            d.framework_id,
            d.is_informal,
            d.is_archived,
            d.evaluator,
            d.created_by,
            d.created_at,
            d.updated_at,
            d.deleted_at
        FROM public.danielson d
        WHERE d.id = $1 
            AND d.deleted_at IS NULL
            AND d.organization_id = $2
        """

        params = [evaluation_id, permissions.organization_id]

        # Add school-level filtering for principals
        if permissions.is_principal and permissions.school_ids:
            school_placeholders = [f"${3 + i}" for i in range(len(permissions.school_ids))]
            query += f" AND d.school_name = ANY(ARRAY[{','.join(school_placeholders)}])"
            params.extend(permissions.school_ids)

        # Add evaluator filtering
        if permissions.is_evaluator:
            evaluator_param = f"${len(params) + 1}"
            query += f" AND (d.created_by = {evaluator_param} OR d.evaluator = {evaluator_param})"
            params.append(permissions.user_id)

        pool = await get_database_pool()
        result = await pool.execute_query_one(query, *params)

        if not result:
            return None

        # Parse JSON fields
        evaluation_json = {}
        if result['evaluation']:
            try:
                evaluation_json = json.loads(result['evaluation']) if isinstance(result['evaluation'], str) else result['evaluation']
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse evaluation JSON for evaluation {evaluation_id}")

        ai_evaluation = None
        if result['ai_evaluation']:
            try:
                ai_evaluation = json.loads(result['ai_evaluation']) if isinstance(result['ai_evaluation'], str) else result['ai_evaluation']
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse ai_evaluation JSON for evaluation {evaluation_id}")

        return EvaluationRecord(
            id=result['id'],
            teacher_name=result['teacher_name'],
            school_name=result['school_name'],
            organization_id=result['organization_id'],
            evaluation_json=evaluation_json,
            ai_evaluation=ai_evaluation,
            low_inference_notes=result['low_inference_notes'],
            framework_id=result['framework_id'],
            is_informal=result['is_informal'],
            is_archived=result['is_archived'],
            evaluator=result['evaluator'],
            created_by=result['created_by'],
            created_at=result['created_at'],
            updated_at=result['updated_at'],
            deleted_at=result['deleted_at']
        )

    @staticmethod
    async def get_teachers_for_user(permissions: UserPermissions) -> List[TeacherSummary]:
        """
        Get all teachers accessible to a user with evaluation counts.
        
        Args:
            permissions: User permissions object
            
        Returns:
            List of TeacherSummary objects
        """
        where_conditions = ["d.deleted_at IS NULL"]
        params = []
        param_counter = 1

        # Organization-level filtering
        where_conditions.append(f"d.organization_id = ${param_counter}")
        params.append(permissions.organization_id)
        param_counter += 1

        # School-level filtering for principals
        if permissions.is_principal and permissions.school_ids:
            school_placeholders = [f"${param_counter + i}" for i in range(len(permissions.school_ids))]
            where_conditions.append(f"d.school_name = ANY(ARRAY[{','.join(school_placeholders)}])")
            params.extend(permissions.school_ids)
            param_counter += len(permissions.school_ids)

        # Evaluator-based filtering
        if permissions.is_evaluator:
            where_conditions.append(f"(d.created_by = ${param_counter} OR d.evaluator = ${param_counter})")
            params.append(permissions.user_id)
            param_counter += 1

        query = f"""
        SELECT 
            d.teacher_name,
            d.school_name,
            d.organization_id,
            COUNT(d.id) as total_evaluations,
            MAX(d.created_at) as latest_evaluation_date,
            ARRAY_AGG(d.id ORDER BY d.created_at DESC) as evaluation_ids
        FROM public.danielson d
        WHERE {' AND '.join(where_conditions)}
        GROUP BY d.teacher_name, d.school_name, d.organization_id
        ORDER BY d.teacher_name
        """

        pool = await get_database_pool()
        results = await pool.execute_query(query, *params)

        teachers = []
        for row in results:
            teachers.append(TeacherSummary(
                teacher_name=row['teacher_name'],
                school_name=row['school_name'],
                organization_id=row['organization_id'],
                total_evaluations=row['total_evaluations'],
                latest_evaluation_date=row['latest_evaluation_date'],
                evaluation_ids=row['evaluation_ids'] or []
            ))

        return teachers

    @staticmethod
    async def get_schools_for_user(permissions: UserPermissions) -> List[SchoolSummary]:
        """
        Get all schools accessible to a user with teacher/evaluation counts.
        
        Args:
            permissions: User permissions object
            
        Returns:
            List of SchoolSummary objects
        """
        where_conditions = ["d.deleted_at IS NULL"]
        params = []
        param_counter = 1

        # Organization-level filtering
        where_conditions.append(f"d.organization_id = ${param_counter}")
        params.append(permissions.organization_id)
        param_counter += 1

        # School-level filtering for principals
        if permissions.is_principal and permissions.school_ids:
            school_placeholders = [f"${param_counter + i}" for i in range(len(permissions.school_ids))]
            where_conditions.append(f"d.school_name = ANY(ARRAY[{','.join(school_placeholders)}])")
            params.extend(permissions.school_ids)
            param_counter += len(permissions.school_ids)

        # Evaluator-based filtering
        if permissions.is_evaluator:
            where_conditions.append(f"(d.created_by = ${param_counter} OR d.evaluator = ${param_counter})")
            params.append(permissions.user_id)
            param_counter += 1

        query = f"""
        SELECT 
            d.school_name,
            d.organization_id,
            COUNT(DISTINCT d.teacher_name) as total_teachers,
            COUNT(d.id) as total_evaluations,
            ARRAY_AGG(DISTINCT d.teacher_name ORDER BY d.teacher_name) as teacher_names
        FROM public.danielson d
        WHERE {' AND '.join(where_conditions)}
        GROUP BY d.school_name, d.organization_id
        ORDER BY d.school_name
        """

        pool = await get_database_pool()
        results = await pool.execute_query(query, *params)

        schools = []
        for row in results:
            schools.append(SchoolSummary(
                school_name=row['school_name'],
                organization_id=row['organization_id'],
                total_teachers=row['total_teachers'],
                total_evaluations=row['total_evaluations'],
                teacher_names=row['teacher_names'] or []
            ))

        return schools

    @staticmethod
    async def get_organization_tree(permissions: UserPermissions) -> OrganizationTree:
        """
        Build organizational tree structure for a user showing their access scope.
        
        Args:
            permissions: User permissions object
            
        Returns:
            OrganizationTree object
        """
        # Get organization name
        org_query = """
        SELECT name 
        FROM public.organizations 
        WHERE id = $1 AND deleted_at IS NULL
        """
        
        pool = await get_database_pool()
        org_result = await pool.execute_query_one(org_query, permissions.organization_id)
        org_name = org_result['name'] if org_result else 'Unknown Organization'

        # Get accessible schools and teachers
        where_conditions = ["d.deleted_at IS NULL"]
        params = []
        param_counter = 1

        # Organization-level filtering
        where_conditions.append(f"d.organization_id = ${param_counter}")
        params.append(permissions.organization_id)
        param_counter += 1

        # School-level filtering for principals
        if permissions.is_principal and permissions.school_ids:
            school_placeholders = [f"${param_counter + i}" for i in range(len(permissions.school_ids))]
            where_conditions.append(f"d.school_name = ANY(ARRAY[{','.join(school_placeholders)}])")
            params.extend(permissions.school_ids)
            param_counter += len(permissions.school_ids)

        # Evaluator-based filtering
        if permissions.is_evaluator:
            where_conditions.append(f"(d.created_by = ${param_counter} OR d.evaluator = ${param_counter})")
            params.append(permissions.user_id)
            param_counter += 1

        tree_query = f"""
        SELECT 
            DISTINCT d.school_name,
            d.teacher_name,
            COUNT(d.id) OVER() as total_evaluations
        FROM public.danielson d
        WHERE {' AND '.join(where_conditions)}
        ORDER BY d.school_name, d.teacher_name
        """

        results = await pool.execute_query(tree_query, *params)

        accessible_schools = set()
        accessible_teachers = {}
        total_evaluations = 0

        for row in results:
            accessible_schools.add(row['school_name'])
            accessible_teachers[row['teacher_name']] = row['school_name']
            if total_evaluations == 0:  # Set once from the window function
                total_evaluations = row['total_evaluations']

        return OrganizationTree(
            organization_id=permissions.organization_id,
            organization_name=org_name,
            accessible_schools=list(accessible_schools),
            accessible_teachers=accessible_teachers,
            total_evaluations=total_evaluations
        )