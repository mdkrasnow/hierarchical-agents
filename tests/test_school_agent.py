"""
Tests for SchoolAgent functionality.

This module tests the third layer of the hierarchical agent system,
verifying that SchoolAgent correctly processes TeacherSummary objects
and produces valid SchoolSummary outputs.
"""

import asyncio
import pytest
import sys
from datetime import datetime
from uuid import uuid4
from unittest.mock import Mock, AsyncMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.school import SchoolAgent, SchoolInput, DomainSchoolMetrics, SchoolRiskAnalysis
from models import TeacherSummary, SchoolSummary, PDCohort, DomainSummary, DomainStatus, RiskLevel, TrendDirection
from utils.llm import LLMClient


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = Mock(spec=LLMClient)
    client.call = AsyncMock()
    return client


@pytest.fixture
def sample_teacher_summaries():
    """Create sample teacher summaries for testing."""
    teacher1 = TeacherSummary(
        teacher_id=uuid4(),
        teacher_name="Sarah Johnson",
        school_name="Lincoln Elementary",
        num_evaluations=3,
        per_domain_overview={
            "I-A": DomainSummary(
                domain_id="I-A",
                score=3.2,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.IMPROVING,
                summary="Strong content knowledge"
            ),
            "II-B": DomainSummary(
                domain_id="II-B", 
                score=2.8,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.STABLE,
                summary="Classroom management developing"
            ),
            "III-C": DomainSummary(
                domain_id="III-C",
                score=3.5,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.IMPROVING,
                summary="Excellent student engagement"
            )
        },
        recommended_PD_domains=["II-B"],
        recommended_PD_focus=["Classroom Management"],
        risk_level=RiskLevel.LOW,
        overall_performance_trend=TrendDirection.IMPROVING,
        is_exemplar=False,
        needs_immediate_support=False,
        domain_distribution={
            DomainStatus.GREEN: 2,
            DomainStatus.YELLOW: 1,
            DomainStatus.RED: 0
        },
        overall_short_summary="Developing teacher with strong engagement skills"
    )
    
    teacher2 = TeacherSummary(
        teacher_id=uuid4(),
        teacher_name="Michael Chen",
        school_name="Lincoln Elementary", 
        num_evaluations=4,
        per_domain_overview={
            "I-A": DomainSummary(
                domain_id="I-A",
                score=3.8,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Exceptional content knowledge"
            ),
            "II-B": DomainSummary(
                domain_id="II-B",
                score=3.6,
                status_color=DomainStatus.GREEN, 
                trend=TrendDirection.IMPROVING,
                summary="Excellent classroom management"
            ),
            "III-C": DomainSummary(
                domain_id="III-C",
                score=3.9,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Outstanding student engagement"
            )
        },
        recommended_PD_domains=[],
        recommended_PD_focus=[],
        risk_level=RiskLevel.LOW,
        overall_performance_trend=TrendDirection.IMPROVING,
        is_exemplar=True,
        needs_immediate_support=False,
        domain_distribution={
            DomainStatus.GREEN: 3,
            DomainStatus.YELLOW: 0,
            DomainStatus.RED: 0
        },
        overall_short_summary="Exemplary teacher demonstrating best practices"
    )
    
    teacher3 = TeacherSummary(
        teacher_id=uuid4(),
        teacher_name="Jennifer Davis",
        school_name="Lincoln Elementary",
        num_evaluations=2,
        per_domain_overview={
            "I-A": DomainSummary(
                domain_id="I-A",
                score=1.8,
                status_color=DomainStatus.RED,
                trend=TrendDirection.DECLINING,
                summary="Content knowledge concerns"
            ),
            "II-B": DomainSummary(
                domain_id="II-B",
                score=2.1,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.STABLE,
                summary="Classroom management struggles"
            ),
            "III-C": DomainSummary(
                domain_id="III-C",
                score=2.3,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.STABLE,
                summary="Student engagement inconsistent"
            )
        },
        recommended_PD_domains=["I-A", "II-B"],
        recommended_PD_focus=["Content Knowledge and Pedagogy", "Classroom Management"],
        risk_level=RiskLevel.HIGH,
        overall_performance_trend=TrendDirection.DECLINING,
        is_exemplar=False,
        needs_immediate_support=True,
        domain_distribution={
            DomainStatus.GREEN: 0,
            DomainStatus.YELLOW: 2,
            DomainStatus.RED: 1
        },
        overall_short_summary="New teacher requiring intensive support and mentoring"
    )
    
    return [teacher1, teacher2, teacher3]


@pytest.fixture
def school_input(sample_teacher_summaries):
    """Create a sample SchoolInput for testing."""
    return SchoolInput(
        teacher_summaries=sample_teacher_summaries,
        school_id=uuid4(),
        school_name="Lincoln Elementary",
        analysis_period_start=datetime(2024, 1, 1),
        analysis_period_end=datetime(2024, 6, 30),
        max_cohort_size=8,
        min_cohort_size=2
    )


class TestSchoolAgent:
    """Test cases for SchoolAgent functionality."""
    
    def test_init(self, mock_llm_client):
        """Test SchoolAgent initialization."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        assert agent.agent_type == "SchoolAgent"
        assert "school-level analysis" in agent.role_description
        assert agent.llm_client == mock_llm_client
        assert len(agent.pd_topics) > 0
    
    @pytest.mark.asyncio
    async def test_execute_empty_input(self, mock_llm_client):
        """Test execution with empty teacher summaries."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        empty_input = SchoolInput(
            teacher_summaries=[],
            school_name="Empty School"
        )
        
        result = await agent.execute(empty_input)
        
        assert not result.success
        assert "No teacher summaries provided" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_llm_client, school_input):
        """Test successful execution with teacher summaries."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        # Mock LLM response for narratives - this will trigger fallback
        agent.llm_call_with_template = AsyncMock(side_effect=Exception("No LLM configured"))
        
        result = await agent.execute(school_input)
        
        assert result.success
        assert "school_summary" in result.data
        
        school_summary = SchoolSummary(**result.data["school_summary"])
        assert school_summary.school_name == "Lincoln Elementary"
        assert school_summary.num_teachers_analyzed == 3
        assert len(school_summary.domain_stats) > 0
        assert len(school_summary.exemplar_teachers) >= 1  # Michael Chen
        assert len(school_summary.teachers_needing_support) >= 1  # Jennifer Davis
    
    def test_compute_domain_statistics(self, mock_llm_client, sample_teacher_summaries):
        """Test domain statistics computation."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        domain_metrics = agent._compute_domain_statistics(sample_teacher_summaries)
        
        # Check that all domains are included
        expected_domains = ["I-A", "II-B", "III-C"]
        for domain in expected_domains:
            assert domain in domain_metrics
        
        # Check I-A domain (should have 1 green, 0 yellow, 1 red)
        ia_metrics = domain_metrics["I-A"]
        assert ia_metrics.status_distribution[DomainStatus.GREEN] == 2  # Sarah and Michael
        assert ia_metrics.status_distribution[DomainStatus.RED] == 1    # Jennifer
        assert ia_metrics.total_teachers == 3
        assert ia_metrics.average_score is not None
        
        # Check percentages
        assert abs(ia_metrics.status_percentages[DomainStatus.GREEN] - 2/3) < 0.01
        assert abs(ia_metrics.status_percentages[DomainStatus.RED] - 1/3) < 0.01
    
    def test_analyze_school_risk(self, mock_llm_client, sample_teacher_summaries):
        """Test school-level risk analysis."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        risk_analysis = agent._analyze_school_risk(sample_teacher_summaries)
        
        # Check teacher categorization
        assert "Michael Chen" in risk_analysis.low_risk_teachers
        assert "Sarah Johnson" in risk_analysis.low_risk_teachers
        assert "Jennifer Davis" in risk_analysis.high_risk_teachers
        assert "Jennifer Davis" in risk_analysis.teachers_needing_immediate_support
        
        # Check overall risk (1/3 high risk, but not >= 0.3 threshold, so should be medium due to immediate support)
        # Actually, immediate support rate = 1/3 ≈ 0.33 which is >= 0.2, so HIGH
        assert risk_analysis.overall_school_risk == RiskLevel.HIGH
        assert risk_analysis.retention_concerns == 2  # high risk + immediate support (same person)
    
    def test_classify_teachers(self, mock_llm_client, sample_teacher_summaries):
        """Test teacher classification logic."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        classifications = agent._classify_teachers(sample_teacher_summaries)
        
        assert "Michael Chen" in classifications["exemplar_teachers"]
        assert "Jennifer Davis" in classifications["teachers_needing_support"]
        # Sarah Johnson has 2 green, 1 yellow, 0 red, so green_rate = 2/3 = 0.67 >= 0.6 and red_rate = 0 <= 0.1
        # So she should be in high_performers, not developing_teachers
        assert "Sarah Johnson" in classifications["high_performers"]
    
    def test_generate_pd_cohorts(self, mock_llm_client, sample_teacher_summaries):
        """Test PD cohort generation."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        pd_cohorts = agent._generate_pd_cohorts(sample_teacher_summaries, max_cohort_size=8, min_cohort_size=2)
        
        # Should create a cohort for II-B (Sarah and Jennifer both need it)
        cohort_domains = [cohort.domain_id for cohort in pd_cohorts]
        assert "II-B" in cohort_domains
        
        # Find the II-B cohort
        iib_cohort = next(c for c in pd_cohorts if c.domain_id == "II-B")
        assert len(iib_cohort.teacher_names) == 2
        assert "Sarah Johnson" in iib_cohort.teacher_names
        assert "Jennifer Davis" in iib_cohort.teacher_names
        assert iib_cohort.focus_area == "Classroom Management"
    
    def test_identify_priority_domains(self, mock_llm_client, sample_teacher_summaries):
        """Test priority domain identification."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        domain_metrics = agent._compute_domain_statistics(sample_teacher_summaries)
        priority_domains = agent._identify_priority_domains(domain_metrics)
        
        # I-A should be high priority (1/3 red rate = 33% >= 30%)
        # II-B should be medium priority and is critical domain
        assert "I-A" in priority_domains or "II-B" in priority_domains
    
    def test_determine_cohort_priority(self, mock_llm_client):
        """Test cohort priority determination logic."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        # Critical domain with many teachers
        assert agent._determine_cohort_priority("III-A", 6) == "high"
        
        # Non-critical domain with medium teachers
        assert agent._determine_cohort_priority("I-E", 4) == "medium"
        
        # Small cohort
        assert agent._determine_cohort_priority("IV-F", 3) == "low"
    
    def test_suggest_pd_duration(self, mock_llm_client):
        """Test PD duration suggestion logic."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        # Complex domain with large cohort
        duration = agent._suggest_pd_duration("I-A", 6)
        assert "6-8 sessions" in duration
        
        # Simple domain with small cohort
        duration = agent._suggest_pd_duration("II-E", 3)
        assert "2-3 sessions" in duration
    
    @pytest.mark.asyncio
    async def test_synthesize_school_narratives_fallback(self, mock_llm_client, school_input):
        """Test narrative synthesis with fallback logic."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        # Mock LLM failure to test fallback
        agent.llm_call_with_template = AsyncMock(side_effect=Exception("LLM failed"))
        
        domain_metrics = agent._compute_domain_statistics(school_input.teacher_summaries)
        risk_analysis = agent._analyze_school_risk(school_input.teacher_summaries)
        teacher_classifications = agent._classify_teachers(school_input.teacher_summaries)
        priority_domains = agent._identify_priority_domains(domain_metrics)
        
        narratives = await agent._synthesize_school_narratives(
            school_input, domain_metrics, risk_analysis, teacher_classifications, priority_domains
        )
        
        assert "school_strengths" in narratives
        assert "school_needs" in narratives
        assert "stories_for_principal" in narratives
        assert "stories_for_supervisor" in narratives
        assert "overall_performance_summary" in narratives
        
        # Check that narratives contain meaningful content
        assert len(narratives["school_strengths"]) > 0
        assert len(narratives["school_needs"]) > 0
        assert len(narratives["overall_performance_summary"]) > 20  # Should be a real summary
    
    def test_build_school_summary(self, mock_llm_client, school_input):
        """Test building the final SchoolSummary."""
        agent = SchoolAgent(llm_client=mock_llm_client)
        
        domain_metrics = agent._compute_domain_statistics(school_input.teacher_summaries)
        risk_analysis = agent._analyze_school_risk(school_input.teacher_summaries)
        teacher_classifications = agent._classify_teachers(school_input.teacher_summaries)
        pd_cohorts = agent._generate_pd_cohorts(school_input.teacher_summaries)
        priority_domains = agent._identify_priority_domains(domain_metrics)
        
        narratives = {
            "school_strengths": ["Strong exemplar teachers"],
            "school_needs": ["Support for struggling teachers"],
            "stories_for_principal": ["Focus on mentoring"],
            "stories_for_supervisor": ["School needs targeted intervention"],
            "overall_performance_summary": "Mixed performance with clear action plan"
        }
        
        school_summary = agent._build_school_summary(
            school_input, domain_metrics, risk_analysis, teacher_classifications,
            pd_cohorts, priority_domains, narratives
        )
        
        # Verify all fields are populated
        assert school_summary.school_name == "Lincoln Elementary"
        assert school_summary.num_teachers_analyzed == 3
        assert len(school_summary.domain_stats) > 0
        assert len(school_summary.exemplar_teachers) > 0
        assert len(school_summary.teachers_needing_support) > 0
        assert len(school_summary.PD_cohorts) >= 0
        assert school_summary.overall_performance_level in [DomainStatus.GREEN, DomainStatus.YELLOW, DomainStatus.RED]
        assert school_summary.school_risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_school_agent.py -v
    pytest.main([__file__, "-v"])