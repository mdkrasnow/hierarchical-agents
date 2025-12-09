"""
Tests for the hierarchical orchestrator module.

This module tests the core orchestration logic, role-based execution,
and integration between different agent layers.
"""

import asyncio
import pytest
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestration.hierarchical import (
    HierarchicalOrchestrator, 
    OrchestrationConfig, 
    OrchestrationResult
)
from models import (
    User, Organization, DanielsonEvaluation,
    EvaluationSummary, TeacherSummary, SchoolSummary, DistrictSummary
)
from models.permissions import UserScope, UserRole
from utils.llm import LLMClient


class TestOrchestrationConfig:
    """Test configuration validation and defaults."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestrationConfig()
        assert config.max_concurrent_evaluations == 10
        assert config.max_concurrent_teachers == 5
        assert config.max_concurrent_schools == 3
        assert config.include_informal_evaluations is True
        assert config.show_progress is True
    
    def test_concurrency_validation(self):
        """Test validation of concurrency limits."""
        # Valid values
        config = OrchestrationConfig(max_concurrent_evaluations=5)
        assert config.max_concurrent_evaluations == 5
        
        # Invalid values should raise ValidationError
        with pytest.raises(ValueError):
            OrchestrationConfig(max_concurrent_evaluations=0)
        
        with pytest.raises(ValueError):
            OrchestrationConfig(max_concurrent_evaluations=100)


class TestOrchestrationResult:
    """Test result object structure and validation."""
    
    def test_basic_result_creation(self):
        """Test creating a basic result object."""
        result = OrchestrationResult(
            success=True,
            user_id=uuid4(),
            user_role=UserRole.PRINCIPAL,
            execution_time_ms=1500.0,
            start_time=datetime.now() - timedelta(seconds=2),
            end_time=datetime.now()
        )
        
        assert result.success is True
        assert result.user_role == UserRole.PRINCIPAL
        assert result.execution_time_ms == 1500.0
        assert result.num_evaluations_processed == 0  # Default
        assert result.primary_result_type == "none"  # Default


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_client = AsyncMock(spec=LLMClient)
    return mock_client


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(
        id=uuid4(),
        email="test@example.com",
        role_id=uuid4(),
        class_role="Principal",
        organization_id=uuid4(),
        school_id=[uuid4()],
        created_at=datetime.now(),
        first_name="Test",
        last_name="User",
        active=True
    )


@pytest.fixture
def sample_organization():
    """Create a sample organization for testing."""
    return Organization(
        id=uuid4(),
        name="Test School District",
        performance_level_config={
            "domains": {
                "I-A": {"green": 3.0, "yellow": 2.0},
                "I-B": {"green": 3.0, "yellow": 2.0}
            }
        },
        created_at=datetime.now(),
        type="district"
    )


@pytest.fixture
def sample_evaluations():
    """Create sample evaluations for testing."""
    evaluations = []
    for i in range(3):
        eval = DanielsonEvaluation(
            id=uuid4(),
            evaluation={
                "domains": {
                    "I-A": {"score": 2.5, "components": {}},
                    "I-B": {"score": 3.0, "components": {}}
                }
            },
            framework_id="danielson",
            teacher_name=f"Teacher {i}",
            school_name="Test School",
            created_at=datetime.now() - timedelta(days=i*30)
        )
        evaluations.append(eval)
    return evaluations


@pytest.fixture
def sample_evaluation_summaries():
    """Create sample evaluation summaries for testing."""
    summaries = []
    for i in range(3):
        summary = EvaluationSummary(
            teacher_name=f"Teacher {i}",
            school_name="Test School",
            evaluation_id=uuid4(),
            date=datetime.now() - timedelta(days=i*30),
            per_domain={},
            evidence_snippets=[f"Evidence {i}"],
            key_strengths=[f"Strength {i}"]
        )
        summaries.append(summary)
    return summaries


class TestHierarchicalOrchestrator:
    """Test the main orchestrator functionality."""
    
    def test_orchestrator_initialization(self, mock_llm_client):
        """Test orchestrator initialization with default config."""
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client
        )
        
        assert orchestrator.llm_client == mock_llm_client
        assert isinstance(orchestrator.config, OrchestrationConfig)
        assert orchestrator.evaluation_agent is not None
        assert orchestrator.teacher_agent is not None
        assert orchestrator.school_agent is not None
        assert orchestrator.district_agent is not None
    
    def test_orchestrator_with_custom_config(self, mock_llm_client):
        """Test orchestrator with custom configuration."""
        custom_config = OrchestrationConfig(
            max_concurrent_evaluations=5,
            verbose_output=True,
            show_progress=False
        )
        
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client,
            config=custom_config
        )
        
        assert orchestrator.config.max_concurrent_evaluations == 5
        assert orchestrator.config.verbose_output is True
        assert orchestrator.config.show_progress is False


class TestExecutionFlow:
    """Test the execution flow for different user roles."""
    
    @pytest.mark.asyncio
    async def test_principal_execution_flow(
        self, 
        mock_llm_client, 
        sample_user, 
        sample_organization, 
        sample_evaluations
    ):
        """Test execution flow for a principal user."""
        # Setup
        sample_user.class_role = "Principal"
        user_scope = UserScope.from_user(sample_user, sample_organization)
        
        config = OrchestrationConfig(
            max_concurrent_evaluations=2,
            show_progress=False
        )
        
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client,
            config=config
        )
        
        # Mock agent responses
        with patch.object(orchestrator.evaluation_agent, 'execute_with_tracking') as mock_eval_agent:
            mock_eval_agent.return_value = MagicMock(
                success=True,
                data={"evaluation_summary": {
                    "teacher_name": "Test Teacher",
                    "school_name": "Test School", 
                    "evaluation_id": str(uuid4()),
                    "date": datetime.now().isoformat(),
                    "per_domain": {},
                    "evidence_snippets": [],
                    "key_strengths": []
                }}
            )
            
            with patch.object(orchestrator.teacher_agent, 'execute_with_tracking') as mock_teacher_agent:
                mock_teacher_agent.return_value = MagicMock(
                    success=True,
                    data={"teacher_summary": {
                        "teacher_name": "Test Teacher",
                        "school_name": "Test School",
                        "per_domain_overview": {},
                        "overall_short_summary": "Test summary"
                    }}
                )
                
                with patch.object(orchestrator.school_agent, 'execute_with_tracking') as mock_school_agent:
                    mock_school_agent.return_value = MagicMock(
                        success=True,
                        data={"school_summary": {
                            "school_name": "Test School",
                            "num_teachers_analyzed": 1,
                            "domain_stats": {},
                            "domain_percentages": {},
                            "PD_cohorts": [],
                            "priority_domains": [],
                            "school_strengths": [],
                            "school_needs": [],
                            "stories_for_principal": [],
                            "stories_for_supervisor_or_board": [],
                            "exemplar_teachers": [],
                            "teachers_needing_support": [],
                            "overall_performance_level": "yellow",
                            "school_risk_level": "low",
                            "improvement_trend": "stable"
                        }}
                    )
                    
                    # Execute
                    result = await orchestrator.execute_for_user(
                        user_id=sample_user.id,
                        user_scope=user_scope,
                        organization=sample_organization,
                        evaluations=sample_evaluations
                    )
        
        # Verify result
        assert result.success is True
        assert result.user_role == UserRole.PRINCIPAL
        assert result.primary_result_type == "school"
        assert len(result.evaluation_summaries) > 0
        assert len(result.teacher_summaries) > 0
        assert len(result.school_summaries) > 0
        assert result.district_summary is None  # Principals don't get district analysis
        
        # Verify agents were called appropriately
        assert mock_eval_agent.called
        assert mock_teacher_agent.called
        assert mock_school_agent.called
    
    @pytest.mark.asyncio
    async def test_superintendent_execution_flow(
        self,
        mock_llm_client,
        sample_user,
        sample_organization,
        sample_evaluations
    ):
        """Test execution flow for a superintendent user."""
        # Setup
        sample_user.class_role = "Superintendent" 
        sample_user.school_id = []  # Superintendents have access to all schools
        user_scope = UserScope.from_user(sample_user, sample_organization)
        
        config = OrchestrationConfig(
            max_concurrent_evaluations=2,
            show_progress=False
        )
        
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client,
            config=config
        )
        
        # Mock all agent responses
        with patch.object(orchestrator.evaluation_agent, 'execute_with_tracking') as mock_eval:
            mock_eval.return_value = MagicMock(
                success=True,
                data={"evaluation_summary": {
                    "teacher_name": "Test Teacher",
                    "school_name": "Test School",
                    "evaluation_id": str(uuid4()),
                    "date": datetime.now().isoformat(),
                    "per_domain": {},
                    "evidence_snippets": [],
                    "key_strengths": []
                }}
            )
            
            with patch.object(orchestrator.teacher_agent, 'execute_with_tracking') as mock_teacher:
                mock_teacher.return_value = MagicMock(
                    success=True,
                    data={"teacher_summary": {
                        "teacher_name": "Test Teacher",
                        "school_name": "Test School", 
                        "per_domain_overview": {},
                        "overall_short_summary": "Test summary"
                    }}
                )
                
                with patch.object(orchestrator.school_agent, 'execute_with_tracking') as mock_school:
                    mock_school.return_value = MagicMock(
                        success=True,
                        data={"school_summary": {
                            "school_name": "Test School",
                            "num_teachers_analyzed": 1,
                            "domain_stats": {},
                            "domain_percentages": {},
                            "PD_cohorts": [],
                            "priority_domains": [],
                            "school_strengths": [],
                            "school_needs": [],
                            "stories_for_principal": [],
                            "stories_for_supervisor_or_board": [],
                            "exemplar_teachers": [],
                            "teachers_needing_support": [],
                            "overall_performance_level": "yellow",
                            "school_risk_level": "low", 
                            "improvement_trend": "stable"
                        }}
                    )
                    
                    with patch.object(orchestrator.district_agent, 'execute_with_tracking') as mock_district:
                        mock_district.return_value = MagicMock(
                            success=True,
                            data={"district_summary": {
                                "organization_name": "Test District",
                                "num_schools_analyzed": 1,
                                "num_teachers_analyzed": 1,
                                "priority_domains": [],
                                "district_focus_areas": [],
                                "district_strengths": [],
                                "district_needs": [],
                                "school_rankings_by_domain": {},
                                "high_performing_schools": [],
                                "schools_needing_support": [],
                                "board_ready_stories": [],
                                "executive_summary": "Test summary",
                                "recommended_PD_strategy": [],
                                "pilot_opportunities": [],
                                "resource_allocation_priorities": [],
                                "overall_district_health": "yellow",
                                "system_risk_level": "low",
                                "improvement_momentum": "stable",
                                "common_PD_needs": {},
                                "equity_concerns": [],
                                "celebration_opportunities": []
                            }}
                        )
                        
                        # Execute
                        result = await orchestrator.execute_for_user(
                            user_id=sample_user.id,
                            user_scope=user_scope,
                            organization=sample_organization,
                            evaluations=sample_evaluations
                        )
        
        # Verify result
        assert result.success is True
        assert result.user_role == UserRole.SUPERINTENDENT
        assert result.primary_result_type == "district"
        assert result.district_summary is not None  # Superintendents get district analysis
        
        # Verify all agents were called
        assert mock_eval.called
        assert mock_teacher.called 
        assert mock_school.called
        assert mock_district.called
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        mock_llm_client,
        sample_user,
        sample_organization,
        sample_evaluations
    ):
        """Test error handling when agent execution fails."""
        user_scope = UserScope.from_user(sample_user, sample_organization)
        
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client,
            config=OrchestrationConfig(show_progress=False)
        )
        
        # Mock evaluation agent to raise an exception
        with patch.object(orchestrator.evaluation_agent, 'execute_with_tracking') as mock_eval:
            mock_eval.side_effect = Exception("Test error")
            
            result = await orchestrator.execute_for_user(
                user_id=sample_user.id,
                user_scope=user_scope,
                organization=sample_organization,
                evaluations=sample_evaluations
            )
        
        # Verify error handling
        assert result.success is False
        assert len(result.errors) > 0
        assert "Test error" in result.errors[0]
        assert result.execution_time_ms > 0  # Should still track timing


class TestAgentCoordination:
    """Test coordination between different agent layers."""
    
    @pytest.mark.asyncio
    async def test_evaluation_to_teacher_data_flow(
        self,
        mock_llm_client,
        sample_evaluation_summaries
    ):
        """Test that evaluation summaries are properly grouped for teacher analysis."""
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client,
            config=OrchestrationConfig(show_progress=False)
        )
        
        # Create sample organization and user scope
        organization = Organization(
            id=uuid4(),
            name="Test Org",
            created_at=datetime.now()
        )
        
        user_scope = MagicMock()
        user_scope.role = UserRole.PRINCIPAL
        
        # Mock teacher agent
        with patch.object(orchestrator.teacher_agent, 'execute_with_tracking') as mock_teacher:
            mock_teacher.return_value = MagicMock(
                success=True,
                data={"teacher_summary": {
                    "teacher_name": "Test Teacher",
                    "school_name": "Test School",
                    "per_domain_overview": {},
                    "overall_short_summary": "Test"
                }}
            )
            
            # Process teachers
            teacher_summaries = await orchestrator._process_teachers(
                sample_evaluation_summaries,
                organization,
                user_scope
            )
            
            # Verify teacher agent was called with grouped evaluations
            assert mock_teacher.called
            call_args = mock_teacher.call_args[1]  # kwargs
            teacher_input = call_args['teacher_input']
            
            # Should have grouped evaluations by teacher name
            assert teacher_input.teacher_name in ["Teacher 0", "Teacher 1", "Teacher 2"]
            assert len(teacher_input.evaluations) >= 1
    
    @pytest.mark.asyncio 
    async def test_concurrency_limiting(
        self,
        mock_llm_client,
        sample_evaluations
    ):
        """Test that concurrency limits are respected."""
        config = OrchestrationConfig(
            max_concurrent_evaluations=2,  # Small limit for testing
            show_progress=False
        )
        
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client,
            config=config
        )
        
        organization = MagicMock()
        user_scope = MagicMock()
        
        # Track concurrent calls
        active_calls = []
        max_concurrent = 0
        
        async def mock_execute(**kwargs):
            active_calls.append(1)
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, len(active_calls))
            
            # Simulate some work
            await asyncio.sleep(0.1)
            
            active_calls.pop()
            return MagicMock(success=True, data={"evaluation_summary": {}})
        
        with patch.object(orchestrator.evaluation_agent, 'execute_with_tracking', side_effect=mock_execute):
            await orchestrator._process_evaluations(
                sample_evaluations,
                organization,
                user_scope
            )
        
        # Verify concurrency was limited
        assert max_concurrent <= config.max_concurrent_evaluations


class TestRoleBasedExecution:
    """Test that different user roles get appropriate analysis levels."""
    
    def test_determine_primary_result_principal(self):
        """Test primary result determination for principals."""
        orchestrator = HierarchicalOrchestrator(MagicMock())
        
        hierarchy_config = {"primary_output": "school_summary"}
        
        # Create mock summaries
        eval_summaries = [MagicMock()]
        teacher_summaries = [MagicMock()]
        school_summaries = [MagicMock(model_dump=MagicMock(return_value={"school": "data"}))]
        district_summary = None
        
        primary_result, result_type = orchestrator._determine_primary_result(
            hierarchy_config,
            eval_summaries,
            teacher_summaries, 
            school_summaries,
            district_summary
        )
        
        assert result_type == "school"
        assert primary_result is not None
    
    def test_determine_primary_result_superintendent(self):
        """Test primary result determination for superintendents."""
        orchestrator = HierarchicalOrchestrator(MagicMock())
        
        hierarchy_config = {"primary_output": "district_summary"}
        
        # Create mock summaries
        eval_summaries = [MagicMock()]
        teacher_summaries = [MagicMock()]
        school_summaries = [MagicMock()]
        district_summary = MagicMock(model_dump=MagicMock(return_value={"district": "data"}))
        
        primary_result, result_type = orchestrator._determine_primary_result(
            hierarchy_config,
            eval_summaries,
            teacher_summaries,
            school_summaries,
            district_summary
        )
        
        assert result_type == "district"
        assert primary_result is not None


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration tests for end-to-end scenarios."""
    
    async def test_empty_evaluations_handling(
        self,
        mock_llm_client,
        sample_user,
        sample_organization
    ):
        """Test handling of empty evaluation list."""
        user_scope = UserScope.from_user(sample_user, sample_organization)
        
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client,
            config=OrchestrationConfig(show_progress=False)
        )
        
        result = await orchestrator.execute_for_user(
            user_id=sample_user.id,
            user_scope=user_scope,
            organization=sample_organization,
            evaluations=[]  # Empty list
        )
        
        # Should complete successfully but with no data
        assert result.success is True
        assert result.num_evaluations_processed == 0
        assert len(result.evaluation_summaries) == 0
        assert len(result.teacher_summaries) == 0
    
    async def test_partial_agent_failures(
        self,
        mock_llm_client,
        sample_user,
        sample_organization,
        sample_evaluations
    ):
        """Test handling when some but not all agents succeed."""
        user_scope = UserScope.from_user(sample_user, sample_organization)
        
        orchestrator = HierarchicalOrchestrator(
            llm_client=mock_llm_client,
            config=OrchestrationConfig(show_progress=False)
        )
        
        # Mock evaluation agent to succeed, teacher agent to fail
        with patch.object(orchestrator.evaluation_agent, 'execute_with_tracking') as mock_eval:
            mock_eval.return_value = MagicMock(
                success=True,
                data={"evaluation_summary": {
                    "teacher_name": "Test Teacher",
                    "school_name": "Test School",
                    "evaluation_id": str(uuid4()),
                    "date": datetime.now().isoformat(),
                    "per_domain": {},
                    "evidence_snippets": [],
                    "key_strengths": []
                }}
            )
            
            with patch.object(orchestrator.teacher_agent, 'execute_with_tracking') as mock_teacher:
                mock_teacher.side_effect = Exception("Teacher agent failed")
                
                result = await orchestrator.execute_for_user(
                    user_id=sample_user.id,
                    user_scope=user_scope,
                    organization=sample_organization,
                    evaluations=sample_evaluations
                )
        
        # Should still succeed overall with partial results
        assert result.success is True
        assert len(result.evaluation_summaries) > 0  # Evaluation agent succeeded
        assert len(result.teacher_summaries) == 0    # Teacher agent failed
        assert result.primary_result_type == "evaluation"  # Falls back to available data