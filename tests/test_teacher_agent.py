"""
Tests for TeacherAgent functionality.

Tests both deterministic aggregation logic and LLM-based narrative synthesis.
"""

import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from agents.teacher import TeacherAgent, TeacherInput, DomainMetrics, RiskAnalysis, PDRecommendations
from models import EvaluationSummary, DomainSummary, DomainStatus, RiskLevel, TrendDirection
from utils.llm import create_llm_client


class TestTeacherAgent:
    """Test suite for TeacherAgent."""
    
    @pytest.fixture
    def llm_client(self):
        """Create mock LLM client for testing."""
        return create_llm_client(provider_type="mock")
    
    @pytest.fixture
    def teacher_agent(self, llm_client):
        """Create TeacherAgent instance for testing."""
        return TeacherAgent(llm_client=llm_client)
    
    @pytest.fixture
    def sample_evaluations(self):
        """Create sample evaluation summaries for testing."""
        # First evaluation - mixed performance
        eval1 = EvaluationSummary(
            teacher_id=uuid4(),
            teacher_name="Test Teacher",
            school_id=uuid4(),
            school_name="Test School",
            evaluation_id=uuid4(),
            date=datetime(2024, 9, 15),
            per_domain={
                "I-A": DomainSummary(
                    domain_id="I-A",
                    score=3.2,
                    status_color=DomainStatus.GREEN,
                    summary="Strong content knowledge",
                    growth_signals=["Deep subject expertise"],
                    concern_signals=[],
                    evidence_quotes=["Demonstrates mastery of curriculum"]
                ),
                "II-B": DomainSummary(
                    domain_id="II-B",
                    score=1.8,
                    status_color=DomainStatus.RED,
                    summary="Classroom management struggles",
                    growth_signals=[],
                    concern_signals=["Frequent disruptions", "Unclear procedures"],
                    evidence_quotes=["Students off-task frequently"]
                ),
                "III-C": DomainSummary(
                    domain_id="III-C",
                    score=2.7,
                    status_color=DomainStatus.YELLOW,
                    summary="Moderate engagement",
                    growth_signals=["Some interactive elements"],
                    concern_signals=["Limited student participation"],
                    evidence_quotes=["Mixed student engagement levels"]
                )
            },
            flags={
                "needs_PD": ["II-B", "III-C"],
                "exemplar": False,
                "risk_of_leaving": False,
                "burnout_signals": True
            },
            evidence_snippets=[
                "Teacher shows strong curriculum knowledge",
                "Classroom management needs improvement",
                "Student engagement varies"
            ],
            key_strengths=["Subject matter expertise"],
            key_concerns=["Classroom management", "Student engagement"],
            relevance_to_question="high",
            evaluation_type="formal"
        )
        
        # Second evaluation - showing improvement
        eval2 = EvaluationSummary(
            teacher_id=uuid4(),
            teacher_name="Test Teacher",
            school_id=uuid4(),
            school_name="Test School",
            evaluation_id=uuid4(),
            date=datetime(2024, 11, 20),
            per_domain={
                "I-A": DomainSummary(
                    domain_id="I-A",
                    score=3.4,
                    status_color=DomainStatus.GREEN,
                    summary="Continued strength in content",
                    growth_signals=["Excellent explanations"],
                    concern_signals=[],
                    evidence_quotes=["Students grasp complex concepts easily"]
                ),
                "II-B": DomainSummary(
                    domain_id="II-B",
                    score=2.3,
                    status_color=DomainStatus.YELLOW,
                    summary="Improvement in management",
                    growth_signals=["Better procedures", "Clearer expectations"],
                    concern_signals=["Still occasional disruptions"],
                    evidence_quotes=["Noticeable improvement in classroom flow"]
                ),
                "III-C": DomainSummary(
                    domain_id="III-C",
                    score=3.1,
                    status_color=DomainStatus.GREEN,
                    summary="Strong engagement strategies",
                    growth_signals=["Active participation", "Variety of methods"],
                    concern_signals=[],
                    evidence_quotes=["Students actively engaged throughout lesson"]
                )
            },
            flags={
                "needs_PD": ["II-B"],
                "exemplar": False,
                "risk_of_leaving": False,
                "burnout_signals": False
            },
            evidence_snippets=[
                "Significant improvement in classroom management",
                "Student engagement much improved",
                "Content expertise remains strong"
            ],
            key_strengths=["Subject knowledge", "Student engagement"],
            key_concerns=["Ongoing management development"],
            relevance_to_question="high",
            evaluation_type="formal"
        )
        
        return [eval1, eval2]
    
    @pytest.fixture
    def teacher_input(self, sample_evaluations):
        """Create TeacherInput for testing."""
        return TeacherInput(
            evaluations=sample_evaluations,
            teacher_name="Test Teacher",
            analysis_period_start=datetime(2024, 9, 1),
            analysis_period_end=datetime(2024, 12, 1),
            pd_focus_limit=3
        )
    
    def test_compute_domain_metrics(self, teacher_agent, sample_evaluations):
        """Test deterministic domain metrics computation."""
        metrics = teacher_agent._compute_domain_metrics(sample_evaluations)
        
        # Should have metrics for all domains
        assert "I-A" in metrics
        assert "II-B" in metrics
        assert "III-C" in metrics
        
        # Check I-A (consistently strong)
        ia_metrics = metrics["I-A"]
        assert ia_metrics.domain_id == "I-A"
        assert len(ia_metrics.scores) == 2
        assert ia_metrics.average_score == pytest.approx(3.3, rel=1e-2)
        assert ia_metrics.score_trend == TrendDirection.IMPROVING  # 3.2 -> 3.4
        assert ia_metrics.status_distribution[DomainStatus.GREEN] == 2
        assert len(ia_metrics.growth_signals) > 0
        
        # Check II-B (improvement from red to yellow)
        iib_metrics = metrics["II-B"]
        assert iib_metrics.average_score == pytest.approx(2.05, rel=1e-2)
        assert iib_metrics.score_trend == TrendDirection.IMPROVING  # 1.8 -> 2.3
        assert iib_metrics.status_distribution[DomainStatus.RED] == 1
        assert iib_metrics.status_distribution[DomainStatus.YELLOW] == 1
        assert len(iib_metrics.concern_signals) > 0
        
        # Check III-C (strong improvement)
        iiic_metrics = metrics["III-C"]
        assert iiic_metrics.average_score == pytest.approx(2.9, rel=1e-2)
        assert iiic_metrics.score_trend == TrendDirection.IMPROVING  # 2.7 -> 3.1
    
    def test_analyze_risk_factors(self, teacher_agent, sample_evaluations):
        """Test risk factor analysis."""
        risk_analysis = teacher_agent._analyze_risk_factors(sample_evaluations)
        
        # Should detect burnout in first evaluation
        assert risk_analysis.overall_risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        assert risk_analysis.risk_trend == TrendDirection.IMPROVING  # burnout -> no burnout
        assert not risk_analysis.needs_immediate_support  # improving trend
    
    def test_analyze_time_trends(self, teacher_agent, sample_evaluations):
        """Test time-based trend analysis."""
        trends = teacher_agent._analyze_time_trends(sample_evaluations)
        
        assert trends["overall_trend"] == TrendDirection.IMPROVING
        assert trends["num_evaluations"] == 2
        assert len(trends["overall_scores"]) == 2
        assert trends["time_span_days"] > 0
    
    def test_generate_pd_recommendations(self, teacher_agent):
        """Test PD recommendation logic."""
        # Create domain metrics with clear weak areas
        domain_metrics = {
            "I-A": DomainMetrics(
                domain_id="I-A",
                scores=[3.2, 3.4],
                average_score=3.3,
                status_distribution={
                    DomainStatus.GREEN: 2,
                    DomainStatus.YELLOW: 0,
                    DomainStatus.RED: 0
                }
            ),
            "II-B": DomainMetrics(
                domain_id="II-B",
                scores=[1.8, 2.3],
                average_score=2.05,
                score_trend=TrendDirection.IMPROVING,
                status_distribution={
                    DomainStatus.GREEN: 0,
                    DomainStatus.YELLOW: 1,
                    DomainStatus.RED: 1
                },
                concern_signals=["Frequent disruptions", "Unclear procedures"]
            ),
            "III-C": DomainMetrics(
                domain_id="III-C",
                scores=[2.7, 3.1],
                average_score=2.9,
                status_distribution={
                    DomainStatus.GREEN: 1,
                    DomainStatus.YELLOW: 1,
                    DomainStatus.RED: 0
                }
            )
        }
        
        recommendations = teacher_agent._generate_pd_recommendations(
            domain_metrics, {}, focus_limit=2
        )
        
        # II-B should be top priority (lowest score, red status, concerns)
        assert "II-B" in recommendations.priority_domains
        assert len(recommendations.priority_domains) <= 2
        assert len(recommendations.specific_topics) == len(recommendations.priority_domains)
        assert recommendations.urgency_level in ["low", "medium", "high"]
        assert len(recommendations.rationale) > 0
    
    def test_aggregate_evidence(self, teacher_agent, sample_evaluations):
        """Test evidence aggregation."""
        evidence = teacher_agent._aggregate_evidence(sample_evaluations)
        
        assert isinstance(evidence, list)
        assert len(evidence) <= 5  # Should limit to top 5
        assert len(evidence) > 0  # Should have some evidence
        
        # Should include both evidence snippets and key strengths
        combined_text = " ".join(evidence)
        assert any(snippet in combined_text for snippet in sample_evaluations[0].evidence_snippets)
    
    @pytest.mark.asyncio
    async def test_synthesize_narratives_fallback(self, teacher_agent):
        """Test narrative synthesis fallback when LLM fails."""
        # Mock LLM to fail
        with patch.object(teacher_agent, 'llm_call_with_template', side_effect=Exception("LLM failed")):
            domain_metrics = {
                "I-A": DomainMetrics(domain_id="I-A", average_score=3.3),
                "II-B": DomainMetrics(domain_id="II-B", average_score=2.0)
            }
            
            risk_analysis = RiskAnalysis(overall_risk_level=RiskLevel.LOW)
            pd_recommendations = PDRecommendations(
                priority_domains=["II-B"],
                specific_topics=["Classroom Management"]
            )
            
            teacher_input = TeacherInput(
                evaluations=[],
                teacher_name="Test Teacher"
            )
            
            narratives = await teacher_agent._synthesize_narratives(
                teacher_input, domain_metrics, risk_analysis, pd_recommendations
            )
            
            assert "overall_summary" in narratives
            assert narratives["overall_summary"] is not None
            assert len(narratives["overall_summary"]) > 0
    
    @pytest.mark.asyncio
    async def test_execute_success(self, teacher_agent, teacher_input):
        """Test successful teacher analysis execution."""
        result = await teacher_agent.execute(teacher_input)
        
        assert result.success
        assert "teacher_summary" in result.data
        
        summary = result.data["teacher_summary"]
        assert summary["teacher_name"] == "Test Teacher"
        assert summary["num_evaluations"] == 2
        assert "per_domain_overview" in summary
        assert "recommended_PD_focus" in summary
        assert "risk_level" in summary
        assert "overall_short_summary" in summary
        
        # Check domain overview structure
        domain_overview = summary["per_domain_overview"]
        assert "I-A" in domain_overview
        assert "II-B" in domain_overview
        assert "III-C" in domain_overview
        
        # Should have PD recommendations for weak areas
        assert len(summary["recommended_PD_focus"]) > 0
        
        # Check metadata
        assert result.metadata["teacher_name"] == "Test Teacher"
        assert result.metadata["evaluations_count"] == 2
    
    @pytest.mark.asyncio
    async def test_execute_no_evaluations(self, teacher_agent):
        """Test handling of empty evaluations list."""
        empty_input = TeacherInput(
            evaluations=[],
            teacher_name="Empty Teacher"
        )
        
        result = await teacher_agent.execute(empty_input)
        
        assert not result.success
        assert "No evaluations provided" in result.error
    
    def test_domain_groups_and_topics(self, teacher_agent):
        """Test that domain groups and PD topics are properly configured."""
        # Check domain groups exist
        assert "planning" in teacher_agent.domain_groups
        assert "environment" in teacher_agent.domain_groups
        assert "instruction" in teacher_agent.domain_groups
        assert "professionalism" in teacher_agent.domain_groups
        
        # Check domains are assigned to groups
        all_domains = set()
        for domains in teacher_agent.domain_groups.values():
            all_domains.update(domains)
        
        assert "I-A" in all_domains
        assert "II-B" in all_domains
        assert "III-C" in all_domains
        assert "IV-A" in all_domains
        
        # Check PD topics mapping
        assert "I-A" in teacher_agent.pd_topics
        assert "II-B" in teacher_agent.pd_topics
        assert teacher_agent.pd_topics["I-A"] == "Content Knowledge and Pedagogy"
        assert teacher_agent.pd_topics["II-B"] == "Classroom Management"
    
    def test_build_teacher_summary(self, teacher_agent, teacher_input):
        """Test teacher summary construction."""
        # Create test data
        domain_metrics = {
            "I-A": DomainMetrics(
                domain_id="I-A",
                average_score=3.3,
                score_trend=TrendDirection.IMPROVING,
                growth_signals=["Strong expertise"],
                evidence_quotes=["Excellent content delivery"]
            ),
            "II-B": DomainMetrics(
                domain_id="II-B",
                average_score=2.0,
                score_trend=TrendDirection.IMPROVING,
                concern_signals=["Management issues"],
                evidence_quotes=["Needs support with procedures"]
            )
        }
        
        risk_analysis = RiskAnalysis(
            overall_risk_level=RiskLevel.LOW,
            needs_immediate_support=False
        )
        
        pd_recommendations = PDRecommendations(
            priority_domains=["II-B"],
            specific_topics=["Classroom Management"]
        )
        
        notable_evidence = ["Strong teaching practices", "Areas for growth identified"]
        
        narratives = {
            "growth_story": "Shows excellent content knowledge",
            "concern_story": "Needs classroom management support",
            "overall_summary": "Developing teacher with clear strengths and growth areas"
        }
        
        time_trends = {"overall_trend": TrendDirection.IMPROVING}
        
        summary = teacher_agent._build_teacher_summary(
            teacher_input,
            domain_metrics,
            risk_analysis,
            pd_recommendations,
            notable_evidence,
            narratives,
            time_trends
        )
        
        assert summary.teacher_name == "Test Teacher"
        assert summary.num_evaluations == 2
        assert summary.risk_level == RiskLevel.LOW
        assert summary.overall_performance_trend == TrendDirection.IMPROVING
        assert summary.growth_story == "Shows excellent content knowledge"
        assert summary.concern_story == "Needs classroom management support"
        assert "II-B" in summary.recommended_PD_domains
        assert "Classroom Management" in summary.recommended_PD_focus
        
        # Check domain distribution
        assert summary.domain_distribution[DomainStatus.GREEN] >= 1  # I-A is green
        assert summary.domain_distribution[DomainStatus.RED] >= 1   # II-B is red
        
        # Should not be exemplar with red domains
        assert not summary.is_exemplar
    
    @pytest.mark.asyncio
    async def test_agent_properties(self, teacher_agent):
        """Test agent type and role properties."""
        assert teacher_agent.agent_type == "TeacherAgent"
        assert "teacher" in teacher_agent.role_description.lower()
        assert "aggregate" in teacher_agent.role_description.lower()


class TestDomainMetrics:
    """Test DomainMetrics model."""
    
    def test_domain_metrics_creation(self):
        """Test creating DomainMetrics instance."""
        metrics = DomainMetrics(
            domain_id="I-A",
            scores=[3.0, 3.2, 3.4],
            average_score=3.2,
            score_trend=TrendDirection.IMPROVING
        )
        
        assert metrics.domain_id == "I-A"
        assert metrics.average_score == 3.2
        assert metrics.score_trend == TrendDirection.IMPROVING
        assert len(metrics.scores) == 3
        
        # Check default values
        assert DomainStatus.GREEN in metrics.status_distribution
        assert metrics.status_distribution[DomainStatus.GREEN] == 0


class TestTeacherInput:
    """Test TeacherInput model validation."""
    
    def test_teacher_input_validation(self):
        """Test TeacherInput model validation."""
        # Minimal valid input
        teacher_input = TeacherInput(
            evaluations=[],
            teacher_name="Test Teacher"
        )
        
        assert teacher_input.teacher_name == "Test Teacher"
        assert teacher_input.evaluations == []
        assert teacher_input.pd_focus_limit == 3  # Default value
        
        # With optional fields
        teacher_input_full = TeacherInput(
            evaluations=[],
            teacher_name="Full Teacher",
            teacher_id=uuid4(),
            analysis_period_start=datetime(2024, 9, 1),
            analysis_period_end=datetime(2024, 12, 1),
            pd_focus_limit=5
        )
        
        assert teacher_input_full.teacher_id is not None
        assert teacher_input_full.analysis_period_start is not None
        assert teacher_input_full.pd_focus_limit == 5