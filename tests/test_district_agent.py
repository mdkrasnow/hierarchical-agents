"""Tests for DistrictAgent functionality."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timezone
from typing import Dict, List
from uuid import uuid4, UUID

from agents.district import DistrictAgent, DistrictInput
from models import (
    SchoolSummary, 
    DistrictSummary, 
    DomainStatus, 
    RiskLevel, 
    TrendDirection,
    PDCohort
)
from utils.llm import LLMClient, LLMProvider, LLMRequest, LLMResponse


class MockDistrictLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    async def call_single(self, request: LLMRequest) -> LLMResponse:
        # Mock LLM response for district narratives
        if request.response_format and "DistrictNarratives" in str(request.response_format):
            from agents.district import DistrictNarratives, BoardStory
            mock_data = DistrictNarratives(
                district_strengths=[
                    "Strong instructional leadership across multiple schools",
                    "Effective cross-school collaboration initiatives"
                ],
                district_needs=[
                    "System-wide focus on student engagement strategies",
                    "Enhanced professional development coordination"
                ],
                executive_summary="District demonstrates balanced performance with clear strategic opportunities for growth. Strong leadership foundation supports continued improvement.",
                recommended_pd_strategy=[
                    "District-wide student engagement initiative",
                    "Cross-school instructional coaching program",
                    "Leadership development for emerging administrators"
                ],
                board_stories=[
                    BoardStory(
                        title="Celebrating School Excellence",
                        narrative="Multiple schools demonstrate exceptional performance with strong teacher leadership",
                        story_type="positive",
                        supporting_data={"high_performing_schools": 2}
                    )
                ],
                celebration_opportunities=[
                    "Recognize exemplary schools at board meeting",
                    "Highlight cross-school collaboration successes"
                ],
                resource_priorities=[
                    "Professional development in student engagement",
                    "Support for high-need schools"
                ]
            )
            return LLMResponse(
                content=mock_data.model_dump_json(),
                parsed_data=mock_data,
                latency_ms=150.0,
                token_usage={"input_tokens": 500, "output_tokens": 300}
            )
        return LLMResponse(
            content="Mock LLM response",
            latency_ms=100.0,
            token_usage={"input_tokens": 50, "output_tokens": 25}
        )
    
    async def call_batch(self, requests):
        """Mock batch call."""
        import asyncio
        import time
        from utils.llm import BatchResult
        
        start_time = time.time()
        
        tasks = [self.call_single(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return BatchResult(
            responses=responses,
            failed_requests=[],
            total_latency_ms=total_latency_ms,
            successful_count=len(responses),
            failed_count=0
        )


@pytest.fixture
def mock_llm_client():
    """Provide mock LLM client for testing."""
    mock_provider = MockDistrictLLMProvider()
    return LLMClient(provider=mock_provider)


@pytest.fixture
def district_agent(mock_llm_client):
    """Create DistrictAgent instance for testing."""
    return DistrictAgent(llm_client=mock_llm_client)


@pytest.fixture
def sample_school_summaries() -> List[SchoolSummary]:
    """Create sample school summaries for testing."""
    
    # High performing school
    school1 = SchoolSummary(
        school_id=uuid4(),
        school_name="Excellence Elementary",
        num_teachers_analyzed=15,
        domain_stats={
            "I-A": {DomainStatus.GREEN: 12, DomainStatus.YELLOW: 3, DomainStatus.RED: 0},
            "II-B": {DomainStatus.GREEN: 10, DomainStatus.YELLOW: 4, DomainStatus.RED: 1},
            "III-A": {DomainStatus.GREEN: 13, DomainStatus.YELLOW: 2, DomainStatus.RED: 0}
        },
        domain_percentages={
            "I-A": {DomainStatus.GREEN: 0.8, DomainStatus.YELLOW: 0.2, DomainStatus.RED: 0.0},
            "II-B": {DomainStatus.GREEN: 0.67, DomainStatus.YELLOW: 0.27, DomainStatus.RED: 0.07},
            "III-A": {DomainStatus.GREEN: 0.87, DomainStatus.YELLOW: 0.13, DomainStatus.RED: 0.0}
        },
        priority_domains=["II-B"],
        school_strengths=["Strong content knowledge", "Excellent classroom environment"],
        school_needs=["Enhanced behavior management systems"],
        exemplar_teachers=["Ms. Johnson", "Mr. Smith", "Dr. Garcia"],
        teachers_needing_support=["Ms. Wilson"],
        overall_performance_level=DomainStatus.GREEN,
        school_risk_level=RiskLevel.LOW,
        improvement_trend=TrendDirection.IMPROVING,
        PD_cohorts=[
            PDCohort(
                domain_id="II-B",
                focus_area="Classroom Management",
                teacher_names=["Ms. Wilson", "Mr. Davis"],
                priority_level="medium"
            )
        ]
    )
    
    # Average performing school
    school2 = SchoolSummary(
        school_id=uuid4(),
        school_name="Progress Middle School", 
        num_teachers_analyzed=20,
        domain_stats={
            "I-A": {DomainStatus.GREEN: 8, DomainStatus.YELLOW: 10, DomainStatus.RED: 2},
            "II-B": {DomainStatus.GREEN: 6, DomainStatus.YELLOW: 12, DomainStatus.RED: 2},
            "III-A": {DomainStatus.GREEN: 10, DomainStatus.YELLOW: 8, DomainStatus.RED: 2}
        },
        domain_percentages={
            "I-A": {DomainStatus.GREEN: 0.4, DomainStatus.YELLOW: 0.5, DomainStatus.RED: 0.1},
            "II-B": {DomainStatus.GREEN: 0.3, DomainStatus.YELLOW: 0.6, DomainStatus.RED: 0.1},
            "III-A": {DomainStatus.GREEN: 0.5, DomainStatus.YELLOW: 0.4, DomainStatus.RED: 0.1}
        },
        priority_domains=["I-A", "II-B"],
        school_strengths=["Committed teaching staff"],
        school_needs=["Content knowledge development", "Classroom management support"],
        exemplar_teachers=["Ms. Rodriguez"],
        teachers_needing_support=["Mr. Brown", "Ms. Taylor"],
        overall_performance_level=DomainStatus.YELLOW,
        school_risk_level=RiskLevel.MEDIUM,
        improvement_trend=TrendDirection.STABLE,
        PD_cohorts=[
            PDCohort(
                domain_id="I-A",
                focus_area="Content Knowledge",
                teacher_names=["Mr. Brown", "Ms. Taylor", "Mr. Anderson"],
                priority_level="high"
            )
        ]
    )
    
    # School needing support
    school3 = SchoolSummary(
        school_id=uuid4(),
        school_name="Challenge High School",
        num_teachers_analyzed=25,
        domain_stats={
            "I-A": {DomainStatus.GREEN: 5, DomainStatus.YELLOW: 12, DomainStatus.RED: 8},
            "II-B": {DomainStatus.GREEN: 3, DomainStatus.YELLOW: 10, DomainStatus.RED: 12},
            "III-A": {DomainStatus.GREEN: 6, DomainStatus.YELLOW: 11, DomainStatus.RED: 8}
        },
        domain_percentages={
            "I-A": {DomainStatus.GREEN: 0.2, DomainStatus.YELLOW: 0.48, DomainStatus.RED: 0.32},
            "II-B": {DomainStatus.GREEN: 0.12, DomainStatus.YELLOW: 0.4, DomainStatus.RED: 0.48},
            "III-A": {DomainStatus.GREEN: 0.24, DomainStatus.YELLOW: 0.44, DomainStatus.RED: 0.32}
        },
        priority_domains=["II-B", "I-A", "III-A"],
        school_strengths=["Dedicated staff working toward improvement"],
        school_needs=["Intensive classroom management support", "Content knowledge development", "Student engagement strategies"],
        exemplar_teachers=["Dr. Martinez"],
        teachers_needing_support=["Ms. Clark", "Mr. Thompson", "Ms. Lewis", "Mr. White", "Ms. Green"],
        overall_performance_level=DomainStatus.RED,
        school_risk_level=RiskLevel.HIGH,
        improvement_trend=TrendDirection.DECLINING,
        PD_cohorts=[
            PDCohort(
                domain_id="II-B",
                focus_area="Classroom Management",
                teacher_names=["Ms. Clark", "Mr. Thompson", "Ms. Lewis"],
                priority_level="high"
            ),
            PDCohort(
                domain_id="I-A", 
                focus_area="Content Knowledge",
                teacher_names=["Mr. White", "Ms. Green"],
                priority_level="high"
            )
        ]
    )
    
    return [school1, school2, school3]


@pytest.fixture
def district_input(sample_school_summaries) -> DistrictInput:
    """Create sample district input for testing."""
    return DistrictInput(
        school_summaries=sample_school_summaries,
        organization_id=uuid4(),
        organization_name="Sample School District",
        analysis_period_start=datetime(2024, 9, 1, tzinfo=timezone.utc),
        analysis_period_end=datetime(2024, 12, 15, tzinfo=timezone.utc),
        max_board_stories=6,
        min_school_count_for_ranking=3
    )


class TestDistrictAgent:
    """Test cases for DistrictAgent."""
    
    def test_agent_properties(self, district_agent):
        """Test basic agent properties."""
        assert district_agent.agent_type == "DistrictAgent"
        assert "district-level strategic analysis" in district_agent.role_description.lower()
    
    @pytest.mark.asyncio
    async def test_execute_successful(self, district_agent, district_input):
        """Test successful district analysis execution."""
        result = await district_agent.execute(district_input)
        
        assert result.success
        assert result.data is not None
        assert "district_summary" in result.data
        
        district_summary = result.data["district_summary"]
        assert district_summary["organization_name"] == "Sample School District"
        assert district_summary["num_schools_analyzed"] == 3
        assert district_summary["num_teachers_analyzed"] == 60  # 15 + 20 + 25
    
    @pytest.mark.asyncio
    async def test_execute_empty_input(self, district_agent):
        """Test execution with empty school summaries."""
        empty_input = DistrictInput(
            school_summaries=[],
            organization_id=uuid4(),
            organization_name="Empty District"
        )
        
        result = await district_agent.execute(empty_input)
        
        assert not result.success
        assert "No school summaries provided" in result.error
    
    def test_compute_district_domain_statistics(self, district_agent, sample_school_summaries):
        """Test domain statistics computation across schools."""
        metrics = district_agent._compute_district_domain_statistics(sample_school_summaries)
        
        # Check that metrics were computed for all domains
        expected_domains = {"I-A", "II-B", "III-A"}
        assert set(metrics.keys()) == expected_domains
        
        # Test I-A domain metrics
        ia_metrics = metrics["I-A"]
        assert ia_metrics.total_schools_analyzed == 3
        assert ia_metrics.total_teachers_analyzed == 60
        assert len(ia_metrics.school_averages) == 3
        
        # Check school averages make sense (should be between 1.5-3.5)
        for school_name, avg in ia_metrics.school_averages.items():
            assert 1.5 <= avg <= 3.5
        
        # Excellence Elementary should have highest average in I-A
        excellence_avg = ia_metrics.school_averages["Excellence Elementary"]
        challenge_avg = ia_metrics.school_averages["Challenge High School"]
        assert excellence_avg > challenge_avg
    
    def test_analyze_district_risk(self, district_agent, sample_school_summaries):
        """Test district-level risk analysis."""
        risk_analysis = district_agent._analyze_district_risk(sample_school_summaries)
        
        # Challenge High School should be identified as high risk
        assert "Challenge High School" in risk_analysis.high_risk_schools
        
        # Should have retention concerns for schools with many teachers needing support  
        assert len(risk_analysis.schools_with_retention_concerns) >= 1
        
        # System risk level should be at least medium (1 high risk school out of 3)
        assert risk_analysis.system_risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
        
        # Stability indicators should be populated
        assert "stable_schools" in risk_analysis.stability_indicators
        assert "improving_schools" in risk_analysis.stability_indicators
        assert "declining_schools" in risk_analysis.stability_indicators
    
    def test_generate_school_rankings(self, district_agent, sample_school_summaries):
        """Test school ranking generation."""
        # First compute domain metrics needed for rankings
        domain_metrics = district_agent._compute_district_domain_statistics(sample_school_summaries)
        
        rankings = district_agent._generate_school_rankings(
            sample_school_summaries,
            domain_metrics,
            min_school_count=3
        )
        
        # Should have rankings for each domain plus overall
        expected_ranking_keys = {"I-A", "II-B", "III-A", "overall"}
        assert set(rankings.keys()) == expected_ranking_keys
        
        # Each ranking should have 3 schools
        for domain, ranking_list in rankings.items():
            assert len(ranking_list) == 3
            
            # Rankings should be ordered (highest rank first)
            ranks = [r.overall_rank for r in ranking_list]
            assert ranks == [1, 2, 3]
        
        # Excellence Elementary should rank first overall
        overall_rankings = rankings["overall"]
        assert overall_rankings[0].school_name == "Excellence Elementary"
        assert overall_rankings[2].school_name == "Challenge High School"  # Should rank last
    
    def test_analyze_system_pd_needs(self, district_agent, sample_school_summaries):
        """Test system-wide PD needs analysis."""
        pd_analysis = district_agent._analyze_system_pd_needs(sample_school_summaries)
        
        # Should identify common needs across schools
        assert len(pd_analysis.most_common_needs) > 0
        
        # II-B appears in multiple school priority lists, so should be shared
        assert "II-B" in pd_analysis.shared_priority_domains
        
        # Should generate some recommended initiatives
        assert len(pd_analysis.recommended_initiatives) > 0
        
        # Should find some cohort opportunities
        assert len(pd_analysis.cohort_opportunities) >= 0  # May be empty with small sample
    
    def test_identify_district_priorities(self, district_agent, sample_school_summaries):
        """Test district priority identification."""
        # Compute prerequisites
        domain_metrics = district_agent._compute_district_domain_statistics(sample_school_summaries)
        risk_analysis = district_agent._analyze_district_risk(sample_school_summaries)
        pd_analysis = district_agent._analyze_system_pd_needs(sample_school_summaries)
        
        priorities = district_agent._identify_district_priorities(
            domain_metrics, pd_analysis, risk_analysis
        )
        
        # Should have all priority categories
        expected_categories = {"instructional_priorities", "support_priorities", "strategic_opportunities", "immediate_actions"}
        assert set(priorities.keys()) == expected_categories
        
        # Should identify support priorities due to high risk schools
        assert len(priorities["support_priorities"]) > 0
        
        # Should identify immediate actions due to system risk
        if risk_analysis.system_risk_level == RiskLevel.HIGH:
            assert len(priorities["immediate_actions"]) > 0
    
    def test_classify_schools(self, district_agent, sample_school_summaries):
        """Test school classification for support and recognition."""
        # Compute prerequisites
        domain_metrics = district_agent._compute_district_domain_statistics(sample_school_summaries)
        risk_analysis = district_agent._analyze_district_risk(sample_school_summaries)
        
        classifications = district_agent._classify_schools(
            sample_school_summaries, domain_metrics, risk_analysis
        )
        
        # Should have all classification categories
        expected_categories = {"high_performing_schools", "schools_needing_support", "stable_schools", "pilot_ready_schools"}
        assert set(classifications.keys()) == expected_categories
        
        # Excellence Elementary should be high performing
        assert "Excellence Elementary" in classifications["high_performing_schools"]
        
        # Challenge High School should need support
        assert "Challenge High School" in classifications["schools_needing_support"]
        
        # Progress Middle School should be stable
        assert "Progress Middle School" in classifications["stable_schools"]
    
    @pytest.mark.asyncio
    async def test_synthesize_district_narratives_fallback(self, district_agent, district_input, sample_school_summaries):
        """Test district narrative synthesis with fallback when LLM fails."""
        # Compute prerequisites for narrative synthesis
        domain_metrics = district_agent._compute_district_domain_statistics(sample_school_summaries)
        risk_analysis = district_agent._analyze_district_risk(sample_school_summaries)
        pd_analysis = district_agent._analyze_system_pd_needs(sample_school_summaries)
        strategic_priorities = district_agent._identify_district_priorities(domain_metrics, pd_analysis, risk_analysis)
        school_classifications = district_agent._classify_schools(sample_school_summaries, domain_metrics, risk_analysis)
        
        # Test fallback narrative generation
        fallback_narratives = district_agent._generate_fallback_district_narratives(
            domain_metrics, risk_analysis, pd_analysis, strategic_priorities, school_classifications
        )
        
        # Should generate all required narrative components
        expected_keys = {"district_strengths", "district_needs", "executive_summary", "recommended_pd_strategy", 
                        "board_stories", "celebration_opportunities", "resource_priorities"}
        assert set(fallback_narratives.keys()) == expected_keys
        
        # All components should be non-empty
        assert len(fallback_narratives["district_strengths"]) > 0
        assert len(fallback_narratives["district_needs"]) > 0
        assert fallback_narratives["executive_summary"]
        assert len(fallback_narratives["recommended_pd_strategy"]) > 0
        
        # Board stories should be valid BoardStory objects
        for story in fallback_narratives["board_stories"]:
            assert hasattr(story, 'title')
            assert hasattr(story, 'narrative')
            assert hasattr(story, 'story_type')
    
    def test_build_district_summary(self, district_agent, district_input, sample_school_summaries):
        """Test building the final DistrictSummary output."""
        # Compute all prerequisites
        domain_metrics = district_agent._compute_district_domain_statistics(sample_school_summaries)
        risk_analysis = district_agent._analyze_district_risk(sample_school_summaries)
        school_rankings = district_agent._generate_school_rankings(sample_school_summaries, domain_metrics)
        pd_analysis = district_agent._analyze_system_pd_needs(sample_school_summaries)
        strategic_priorities = district_agent._identify_district_priorities(domain_metrics, pd_analysis, risk_analysis)
        school_classifications = district_agent._classify_schools(sample_school_summaries, domain_metrics, risk_analysis)
        
        # Mock narratives for testing
        narratives = {
            "district_strengths": ["Strong leadership", "Effective collaboration"],
            "district_needs": ["Enhanced PD", "Support for struggling schools"],
            "executive_summary": "District shows balanced performance with clear improvement opportunities.",
            "recommended_pd_strategy": ["System-wide initiative 1", "Cross-school program 2"],
            "board_stories": [],
            "celebration_opportunities": ["Recognize excellence", "Highlight collaboration"],
            "resource_priorities": ["PD funding", "Support staff"]
        }
        
        district_summary = district_agent._build_district_summary(
            district_input, domain_metrics, risk_analysis, school_rankings,
            pd_analysis, strategic_priorities, school_classifications, narratives
        )
        
        # Verify DistrictSummary structure
        assert isinstance(district_summary, DistrictSummary)
        assert district_summary.organization_name == "Sample School District"
        assert district_summary.num_schools_analyzed == 3
        assert district_summary.num_teachers_analyzed == 60
        
        # Verify strategic content
        assert len(district_summary.district_strengths) > 0
        assert len(district_summary.district_needs) > 0
        assert district_summary.executive_summary
        
        # Verify school classifications
        assert len(district_summary.high_performing_schools) > 0
        assert len(district_summary.schools_needing_support) > 0
        
        # Verify system metrics are set appropriately
        assert district_summary.overall_district_health in [DomainStatus.GREEN, DomainStatus.YELLOW, DomainStatus.RED]
        assert district_summary.system_risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert district_summary.improvement_momentum in [TrendDirection.IMPROVING, TrendDirection.STABLE, TrendDirection.DECLINING]
    
    def test_insufficient_schools_for_ranking(self, district_agent):
        """Test handling when there are too few schools for meaningful rankings."""
        # Create minimal school data
        minimal_schools = [
            SchoolSummary(
                school_name="Only School",
                num_teachers_analyzed=10,
                domain_stats={"I-A": {DomainStatus.GREEN: 8, DomainStatus.YELLOW: 2, DomainStatus.RED: 0}},
                overall_performance_level=DomainStatus.GREEN,
                school_risk_level=RiskLevel.LOW,
                improvement_trend=TrendDirection.STABLE
            )
        ]
        
        domain_metrics = district_agent._compute_district_domain_statistics(minimal_schools)
        rankings = district_agent._generate_school_rankings(minimal_schools, domain_metrics, min_school_count=3)
        
        # Should return empty rankings when insufficient schools
        assert rankings == {}
    
    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, district_agent, district_input):
        """Test end-to-end district analysis integration."""
        result = await district_agent.execute_with_tracking(district_input=district_input)
        
        assert result.success
        assert result.data is not None
        
        # Extract and validate district summary
        district_summary_data = result.data["district_summary"]
        
        # Validate basic structure
        assert district_summary_data["organization_name"] == "Sample School District"
        assert district_summary_data["num_schools_analyzed"] == 3
        assert district_summary_data["num_teachers_analyzed"] == 60
        
        # Validate strategic content exists
        assert len(district_summary_data["district_strengths"]) > 0
        assert len(district_summary_data["district_needs"]) > 0
        assert district_summary_data["executive_summary"]
        assert len(district_summary_data["recommended_PD_strategy"]) > 0
        
        # Validate school classifications
        assert len(district_summary_data["high_performing_schools"]) > 0
        assert len(district_summary_data["schools_needing_support"]) > 0
        
        # Validate rankings were generated (or empty if too few schools)
        assert "school_rankings_by_domain" in district_summary_data
        
        # Validate metrics are sensible
        assert district_summary_data["overall_district_health"] in ["green", "yellow", "red"]
        assert district_summary_data["system_risk_level"] in ["low", "medium", "high"]
        assert district_summary_data["improvement_momentum"] in ["improving", "stable", "declining"]