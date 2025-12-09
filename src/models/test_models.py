"""
Simple validation tests for the data models.

This module provides basic validation to ensure our Pydantic models
work correctly and can handle the expected data structures.
"""

from datetime import datetime
from uuid import uuid4, UUID
from typing import Dict, Any

from .database import User, Organization, DanielsonEvaluation, Role
from .agent_outputs import (
    EvaluationSummary, TeacherSummary, SchoolSummary, DistrictSummary,
    DomainStatus, RiskLevel, DomainSummary, TrendDirection
)
from .permissions import UserScope, PermissionScope, UserRole


def test_database_models():
    """Test basic database model functionality."""
    
    # Test Role
    role = Role(
        id=uuid4(),
        name="Principal",
        description="School Principal",
        created_at=datetime.now()
    )
    assert role.name == "Principal"
    
    # Test Organization with performance config
    perf_config = {
        "domains": {
            "I-A": {"green": 3, "yellow": 2, "red": 1},
            "II-B": {"green": 3, "yellow": 2, "red": 1}
        },
        "thresholds": {
            "exemplar": 3.5,
            "proficient": 2.5
        }
    }
    
    org = Organization(
        id=uuid4(),
        name="Test District",
        performance_level_config=perf_config,
        created_at=datetime.now()
    )
    assert org.name == "Test District"
    assert "domains" in org.performance_level_config
    
    # Test User
    user = User(
        id=uuid4(),
        email="principal@test.edu",
        role_id=role.id,
        class_role="Principal",
        organization_id=org.id,
        school_id=[uuid4(), uuid4()],  # Multiple schools
        created_at=datetime.now()
    )
    assert user.email == "principal@test.edu"
    assert len(user.school_id) == 2
    
    # Test Evaluation
    eval_data = {
        "domains": {
            "I-A": {
                "score": 2.5,
                "components": {"1a": 3, "1b": 2},
                "notes": "Shows progress in planning"
            },
            "II-B": {
                "score": 3.0,
                "components": {"2b": 3, "2c": 3},
                "notes": "Strong questioning techniques"
            }
        }
    }
    
    evaluation = DanielsonEvaluation(
        id=uuid4(),
        evaluation=eval_data,
        teacher_name="Jane Smith",
        school_name="Test Elementary",
        framework_id="danielson_2023",
        created_at=datetime.now()
    )
    
    # Test domain extraction
    domain_scores = evaluation.get_domain_scores()
    assert len(domain_scores) == 2
    assert domain_scores[0].domain_id in ["I-A", "II-B"]
    
    print("✓ Database models validation passed")


def test_agent_output_schemas():
    """Test agent output schema functionality."""
    
    # Test DomainSummary
    domain_summary = DomainSummary(
        domain_id="II-B",
        score=2.5,
        status_color=DomainStatus.YELLOW,
        trend=TrendDirection.IMPROVING,
        summary="Questioning skills developing",
        growth_signals=["Using wait time", "More open-ended questions"],
        evidence_quotes=["Students discussed for 10 minutes about the main idea"]
    )
    assert domain_summary.status_color == DomainStatus.YELLOW
    
    # Test EvaluationSummary
    eval_summary = EvaluationSummary(
        teacher_name="Jane Smith",
        school_name="Test Elementary", 
        evaluation_id=uuid4(),
        date=datetime.now(),
        per_domain={
            "II-B": domain_summary
        },
        evidence_snippets=["Strong evidence of growth"],
        flags={"needs_PD": ["II-B"], "exemplar": False}
    )
    assert "II-B" in eval_summary.per_domain
    assert eval_summary.flags["needs_PD"] == ["II-B"]
    
    # Test TeacherSummary
    teacher_summary = TeacherSummary(
        teacher_name="Jane Smith",
        school_name="Test Elementary",
        num_evaluations=3,
        per_domain_overview={"II-B": domain_summary},
        recommended_PD_focus=["Questioning strategies"],
        risk_level=RiskLevel.LOW,
        overall_short_summary="Teacher shows consistent growth in questioning with targeted support needed.",
        domain_distribution={
            DomainStatus.GREEN: 2,
            DomainStatus.YELLOW: 1, 
            DomainStatus.RED: 0
        }
    )
    assert teacher_summary.risk_level == RiskLevel.LOW
    assert teacher_summary.domain_distribution[DomainStatus.GREEN] == 2
    
    # Test SchoolSummary
    school_summary = SchoolSummary(
        school_name="Test Elementary",
        num_teachers_analyzed=15,
        domain_stats={
            "II-B": {DomainStatus.GREEN: 5, DomainStatus.YELLOW: 8, DomainStatus.RED: 2}
        },
        school_strengths=["Strong collaborative culture"],
        school_needs=["Focus on questioning strategies"],
        overall_performance_level=DomainStatus.YELLOW
    )
    assert school_summary.domain_stats["II-B"][DomainStatus.GREEN] == 5
    
    # Test DistrictSummary
    district_summary = DistrictSummary(
        organization_id=uuid4(),
        organization_name="Test District",
        num_schools_analyzed=5,
        num_teachers_analyzed=75,
        priority_domains=["II-B", "I-A"],
        district_strengths=["Strong leadership support"],
        executive_summary="District shows progress in most domains with targeted needs in questioning strategies.",
        overall_district_health=DomainStatus.YELLOW
    )
    assert len(district_summary.priority_domains) == 2
    assert "II-B" in district_summary.priority_domains
    
    print("✓ Agent output schemas validation passed")


def test_permission_models():
    """Test permission and scope functionality."""
    
    # Create test data
    org_id = uuid4()
    school_ids = [uuid4(), uuid4()]
    
    user = User(
        id=uuid4(),
        email="principal@test.edu", 
        role_id=uuid4(),
        class_role="Principal",
        organization_id=org_id,
        school_id=school_ids,
        created_at=datetime.now()
    )
    
    org = Organization(
        id=org_id,
        name="Test District",
        created_at=datetime.now()
    )
    
    # Test UserScope creation
    user_scope = UserScope.from_user(user, org)
    assert user_scope.role == UserRole.PRINCIPAL
    assert org_id in user_scope.permissions.organization_ids
    assert user_scope.permissions.can_run_school_analysis
    
    # Test access checks
    eval_org_id = org_id
    eval_school_id = school_ids[0]
    assert user_scope.can_access_evaluation(eval_school_id, eval_org_id)
    
    # Test filter params
    filter_params = user_scope.get_evaluation_filter_params()
    assert org_id in filter_params["organization_ids"]
    assert eval_school_id in filter_params["school_ids"]
    
    # Test agent config
    agent_config = user_scope.get_agent_hierarchy_config()
    assert agent_config["run_school_agent"] == True
    assert agent_config["run_district_agent"] == False  # Principal doesn't run district
    assert agent_config["primary_output"] == "school_summary"
    
    print("✓ Permission models validation passed")


def run_all_tests():
    """Run all validation tests."""
    print("Running model validation tests...\n")
    
    test_database_models()
    test_agent_output_schemas()
    test_permission_models()
    
    print("\n✅ All model validation tests passed!")
    print("\nModels are ready for:")
    print("- Mapping to Supabase tables")
    print("- Hierarchical agent outputs")
    print("- Role-based permissions")
    print("- Red/yellow/green status system")


if __name__ == "__main__":
    run_all_tests()